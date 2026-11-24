"""
Run the probabilistic simulation (many replicates, each with fresh parameter draws and a
fresh synthetic population), evaluate H1-H5 against pre-registered criteria, write CSVs
and figures.

usage: python run.py [--reps 300] [--n 8000] [--seed 2026] [--out results]
"""
import argparse, json, os
import numpy as np, pandas as pd
from params import PRIORS, SCENARIOS, GROUPS, central, sample
from model import make_population, simulate_year, hemoconc, session_log_rr, PATHWAYS

KO_SINGLE = [(k,) for k in PATHWAYS]
KO_ALL = tuple(PATHWAYS)
KNOCKOUTS = [()] + KO_SINGLE + [KO_ALL]
PER = 1e5  # report risk per 100,000 person-years


def one_replicate(r, n, seed):
    rng_p = np.random.default_rng([seed, r, 0])
    p = sample(rng_p)
    rows = []
    for scen, ov in SCENARIOS.items():
        res = {}
        for gi, grp in enumerate(GROUPS):
            rng = np.random.default_rng([seed, r, gi + 1])      # common random numbers across scenarios
            pop = make_population(grp, n, rng, **ov)
            kos = KNOCKOUTS if scen == "baseline" else [()]
            risk, _ = simulate_year(pop, p, rng, kos)
            res[grp] = {ko: risk[ko].mean() for ko in kos}
            # simulated observed cases in a hypothetical cohort of n people (shows rarity)
            res[grp]["cases"] = np.random.default_rng([seed, r, 99, gi]).binomial(1, risk[()]).sum()
        row = dict(rep=r, scenario=scen, **{f"p_{k}": v for k, v in p.items()})
        for grp in GROUPS:
            row[f"risk_{grp}"] = res[grp][()] * PER
            row[f"cases_{grp}"] = res[grp]["cases"]
        row["IRR_int_mod"] = res["intense"][()] / res["moderate"][()]
        row["IRR_int_sed"] = res["intense"][()] / res["sedentary"][()]
        if scen == "baseline":
            full, none = res["intense"][()], res["intense"][KO_ALL]
            row["excess_int_per100k"] = (full - none) * PER
            for (k,) in KO_SINGLE:
                row[f"share_{k}"] = (full - res["intense"][(k,)]) / max(full - none, 1e-15)
            row["IRR_int_mod_noexercise_effects"] = res["intense"][KO_ALL] / res["moderate"][KO_ALL]
        rows.append(row)
    return rows


def prcc(df, y, xs):
    """Partial rank correlation coefficients (global sensitivity)."""
    R = df[xs + [y]].rank().values
    R = (R - R.mean(0)) / R.std(0)
    out = {}
    for i, x in enumerate(xs):
        others = np.delete(R[:, :-1], i, axis=1)
        A = np.c_[np.ones(len(R)), others]
        rx = R[:, i] - A @ np.linalg.lstsq(A, R[:, i], rcond=None)[0]
        ry = R[:, -1] - A @ np.linalg.lstsq(A, R[:, -1], rcond=None)[0]
        out[x] = float(np.corrcoef(rx, ry)[0, 1])
    return pd.Series(out).sort_values(key=abs, ascending=False)


def dose_response(seed, n_draw=300):
    """Single-session-day RR curves for H3, H4, H5 across parameter uncertainty."""
    rng = np.random.default_rng([seed, 7])
    draws = [central()] + [sample(rng) for _ in range(n_draw)]
    bml_grid = np.linspace(0, 6, 25)        # % body-mass loss
    imm_grid = np.linspace(0, 12, 25)       # extra hours immobile
    load_grid = np.linspace(0, 2.4, 25)     # microtrauma load (1 = a marathon of running)
    out = {"bml": bml_grid, "imm": imm_grid, "load": load_grid}
    curves = {"bml": [], "imm": [], "load": []}
    for p in draws:
        # invert: choose hydration so that BML hits each grid value for a 70 kg, Hct 44 person
        dpv = np.clip(p["k_pv"] * bml_grid / 100, 0, 0.35)
        dhct = 100 * 44 / (44 + 56 * (1 - dpv)) - 44
        curves["bml"].append(np.exp(p["beta_hct"] * dhct))
        curves["imm"].append(np.exp(p["beta_stasis"] * imm_grid))
        curves["load"].append(np.exp(p["beta_micro"] * load_grid))
    for k in curves:
        a = np.array(curves[k])
        out[k + "_central"], out[k + "_lo"], out[k + "_hi"] = a[0], np.percentile(a[1:], 5, 0), np.percentile(a[1:], 95, 0)
    return out


def verdict(x, thr=1.0, prob=0.95):
    s = float((x > thr).mean())
    if s >= prob: return "SUPPORTED", s
    if s <= 1 - prob: return "REFUTED", s
    return "INCONCLUSIVE", s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--n", type=int, default=8000, help="synthetic people per group per replicate")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    ap.add_argument("--priors", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.priors == "v2":
        import params as P, model as M
        P.PRIORS.clear(); P.PRIORS.update(P.PRIORS_V2)
        M._COAG_REF = M._coag_shape({k: v[1] for k, v in P.PRIORS.items()})

    rows = []
    for r in range(a.reps):
        rows += one_replicate(r, a.n, a.seed)
        if (r + 1) % 50 == 0: print(f"  replicate {r+1}/{a.reps}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/replicates.csv", index=False)
    base = df[df.scenario == "baseline"].reset_index(drop=True)

    # --- hypothesis evaluation (criteria fixed before looking at results) -----------
    H = {}
    v, s = verdict(base.IRR_int_mod); H["H1 (intense vs moderate risk >1)"] = (v, s)
    v2, s2 = verdict(base.IRR_int_sed); H["H1b (intense vs sedentary risk >1)"] = (v2, s2)
    shares = base[[f"share_{k}" for k in PATHWAYS]]
    triad = shares[["share_hct", "share_coag"]].sum(1)  # hypercoagulability arm
    H["H2 (all 3 triad arms each >=10% of excess)"] = verdict(
        np.minimum.reduce([triad, base.share_stasis, base.share_micro]), thr=0.10)
    dr = dose_response(a.seed)
    i3 = np.argmin(abs(dr["bml"] - 3)); i4 = np.argmin(abs(dr["imm"] - 4)); i5 = np.argmin(abs(dr["load"] - 1))
    H["H3 (3% dehydration raises day-risk >=10%)"] = ("SUPPORTED" if dr["bml_lo"][i3] >= 1.1 else
                                                     "REFUTED" if dr["bml_hi"][i3] < 1.1 else "INCONCLUSIVE",
                                                     float(dr["bml_central"][i3]))
    H["H4 (4h extra immobility raises day-risk >=10%)"] = ("SUPPORTED" if dr["imm_lo"][i4] >= 1.1 else
                                                          "REFUTED" if dr["imm_hi"][i4] < 1.1 else "INCONCLUSIVE",
                                                          float(dr["imm_central"][i4]))
    H["H5 (marathon load raises day-risk >=10%)"] = ("SUPPORTED" if dr["load_lo"][i5] >= 1.1 else
                                                    "REFUTED" if dr["load_hi"][i5] < 1.1 else "INCONCLUSIVE",
                                                    float(dr["load_central"][i5]))

    pcols = [f"p_{k}" for k in PRIORS if PRIORS[k][0] < PRIORS[k][2]]
    base["logIRR"] = np.log(base.IRR_int_mod)
    sens = prcc(base, "logIRR", pcols)
    sens.to_csv(f"{a.out}/sensitivity_prcc.csv")

    q = lambda x: (float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    summary = {
        "config": vars(a),
        "risk_per100k_median_95UI": {g: q(base[f"risk_{g}"]) for g in GROUPS},
        "IRR_int_mod": q(base.IRR_int_mod), "IRR_int_sed": q(base.IRR_int_sed),
        "IRR_int_mod_if_no_acute_effects": q(base.IRR_int_mod_noexercise_effects),
        "excess_share_median": {k: float(base[f"share_{k}"].median()) for k in PATHWAYS},
        "scenarios_intense_risk_median": {s_: float(df[df.scenario == s_].risk_intense.median()) for s_ in SCENARIOS},
        "scenarios_reduction_vs_baseline_median": {
            s_: float(np.median(1 - df[df.scenario == s_].risk_intense.values / base.risk_intense.values)) for s_ in SCENARIOS},
        "hypotheses": H,
        "sensitivity_top": sens.head(6).to_dict(),
    }
    # power: person-years per group to detect the median IRR (80% power, alpha .05)
    l1, l2 = base.risk_intense.median() / PER, base.risk_moderate.median() / PER
    summary["person_years_per_group_for_80pct_power"] = float((1.96 + 0.84) ** 2 * (l1 + l2) / (l1 - l2) ** 2) if l1 != l2 else None
    summary["sim_cases_per_cohort_median"] = {g: float(base[f"cases_{g}"].median()) for g in GROUPS}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2)
    pd.DataFrame(dr).to_csv(f"{a.out}/dose_response.csv", index=False)

    import figures
    figures.make_all(df, base, dr, sens, a.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
