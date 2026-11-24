"""
Step 1 + 2: anchor the clotting pathway to marathon biomarker data, then calibrate the
whole model to prospective cohort data on physical activity and VTE.

Prior sets compared
  v1         original priors (params.py)
  anchored   v1 + biomarker-fitted clotting kinetics, biomarker-derived beta_coag,
             cohort-informed fitness priors
  skeptical  anchored, but the three unmeasured acute effects have their mode at zero
  null       anchored, all acute exercise effects switched off (for the Bayes factor)

Calibration target (importance weighting / ABC)
  Kunutsor et al. 2020 (Eur J Epidemiol), 14 prospective cohorts, 23,753 VTE events:
  most vs least physically active, BMI-adjusted RR 0.87 (95% CI 0.79-0.95).
  Sensitivity target: Wang et al. 2024 (Sci Rep) OR 0.83 (0.77-0.89).
"""
import json, numpy as np, pandas as pd
import params
from params import PRIORS
import model

rng0 = np.random.default_rng(2026)

# ----------------------------------------------------------------------------- Step 2: biomarkers
# Decay of thrombin-generation markers after finishing (fold vs pre-race baseline).
# Western States 100 (24.6 h race): F1+2 3.9x -> 1.5x and TAT 2.4x -> 1.3x at day 1 (~28 h after finish).
# Prisco 1998 marathon: TAT 9.5x, F1+2 7.3x at finish; activation persists to 24 h; baseline by 48 h.
def tau_from(f0, f1, t):  # excess decays exponentially
    return t / np.log((f0 - 1) / (f1 - 1))
tau_f12 = tau_from(3.9, 1.5, 28.0)
tau_tat = tau_from(2.4, 1.3, 28.0)
# Fibrinolytic activity (tPA): 3.7-4.0x at finish, back to baseline within ~15-28 h (Schobersberger 1996;
# WSER coag:tPA ratio +69% at day 1 while F1+2 is 1.5x, which implies tPA ~0.9x baseline). Upper bound:
tau_fib_max = min(tau_from(3.7, 1.3, 15.0), tau_from(4.0, 1.1, 28.0))
prisco_48h = 1 + 6.3 * np.exp(-48 / np.mean([tau_f12, tau_tat]))

# beta_coag from marker-risk associations: log-HR per doubling of a coagulation-activation marker.
#   low  ln(1.27): D-dimer per doubling (Eichinger 2010, recurrence)
#   high ~ln(3.2)/2: ARIC D-dimer Q5 vs Q1 HR 3.2, assuming ~2 doublings between quintile medians
# Marker fold at finish for a ~4 h marathon effort: 3.9 (ultra) .. 7.3 (fast marathon).
N = 200_000
gamma = rng0.uniform(np.log(1.27), np.log(3.2) / 2, N)
fold = rng0.uniform(3.9, 7.3, N)
tau = rng0.uniform(min(tau_f12, tau_tat), max(tau_f12, tau_tat), N)
avg_fold = 1 + (fold - 1) * (tau / 24) * (1 - np.exp(-24 / tau))     # mean fold over the first day
beta_coag_implied = gamma * np.log2(avg_fold)
bc_lo, bc_mid, bc_hi = np.percentile(beta_coag_implied, [5, 50, 95])

ANCHORED = dict(PRIORS)
ANCHORED.update({
    "tau_coag_h":   (min(tau_f12, tau_tat) * 0.8, np.mean([tau_f12, tau_tat]), max(tau_f12, tau_tat) * 1.2, "fit: WSER F1+2/TAT decay"),
    "tau_fib_h":    (1.0, 3.0, tau_fib_max, "fit: tPA back to baseline within a day"),
    # lower bound 0 keeps the possibility that fibrinolysis fully compensates in trained athletes
    "beta_coag":    (0.0, bc_mid, bc_hi, "derived: marker fold x marker-VTE association"),
    "rr_fitness":   (0.60, 0.80, 1.00, "calibrated below to cohort data"),
    "rr_fit_extra": (0.85, 0.95, 1.00, "Wang 2024: benefit curve flattens at higher activity"),
})
SKEPTICAL = dict(ANCHORED)
for k in ("beta_coag", "beta_micro", "beta_syn"):
    SKEPTICAL[k] = (0.0, 0.0, ANCHORED[k][2], "skeptical: mode at zero")
NULL = dict(ANCHORED)
for k in ("beta_hct", "beta_coag", "beta_stasis", "beta_micro", "beta_syn"):
    NULL[k] = (0.0, 0.0, 0.0, "off")
PRIOR_SETS = {"v1": PRIORS, "anchored": ANCHORED, "skeptical": SKEPTICAL, "null": NULL}


def draw(priors, rng):
    return {k: (rng.triangular(v[0], v[1], v[2]) if v[0] < v[2] else v[1]) for k, v in priors.items()}

# The cohort comparison: "most active" category of a general-population cohort vs "least active",
# BMI-adjusted -> same BMI distribution in both groups.
params.GROUPS["cohort_active"] = dict(spw=4.0, inten=(0.55, 0.75), dur=(0.75, 1.25), races=0, race_h=0.0, bmi=25.5, fit=1)
params.GROUPS["cohort_inactive"] = dict(spw=0.5, inten=(0.35, 0.50), dur=(0.3, 0.7), races=0, race_h=0.0, bmi=25.5, fit=0)
model.GROUPS = params.GROUPS
GROUPS = ["sedentary", "moderate", "intense", "cohort_active", "cohort_inactive"]


def run_draw(p, i, n, ref_priors=None):
    # beta_coag is defined per marathon-equivalent window at the prior set's own central kinetics
    ref = {k: v[1] for k, v in (ref_priors or PRIORS).items()}
    model._COAG_REF = model._coag_shape(ref)
    out = {}
    for gi, g in enumerate(GROUPS):
        rng = np.random.default_rng([7, i, gi])
        pop = model.make_population(g, n, rng)
        out[g] = model.simulate_year(pop, p, rng)[0][()].mean()
    return dict(IRR_int_mod=out["intense"] / out["moderate"], IRR_int_sed=out["intense"] / out["sedentary"],
                IRR_cohort=out["cohort_active"] / out["cohort_inactive"],
                risk_int=out["intense"] * 1e5, risk_mod=out["moderate"] * 1e5, risk_sed=out["sedentary"] * 1e5)


def wq(x, w, qs):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return [float(np.interp(q, c, x)) for q in qs]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=1000); ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()

    print(f"tau_coag fit: F1+2 {tau_f12:.1f} h, TAT {tau_tat:.1f} h (v1 assumed 6 h); predicted Prisco 48 h fold {prisco_48h:.2f}")
    print(f"tau_fib upper bound {tau_fib_max:.1f} h; beta_coag implied by markers: p5 {bc_lo:.2f}, median {bc_mid:.2f}, p95 {bc_hi:.2f}")

    # model noise in the cohort IRR at fixed parameters (finite synthetic population)
    pc = draw(ANCHORED, np.random.default_rng(1)); pc.update({k: v[1] for k, v in ANCHORED.items()})
    noise = np.std([np.log(run_draw(pc, 10_000 + s, a.n, ANCHORED)["IRR_cohort"]) for s in range(20)])
    rows = []
    for name, pri in PRIOR_SETS.items():
        rng = np.random.default_rng([2026, len(name)])
        nd = a.draws if name != "null" else a.draws // 2
        for i in range(nd):
            p = draw(pri, rng)
            rows.append(dict(prior=name, draw=i, **{f"p_{k}": v for k, v in p.items()}, **run_draw(p, i, a.n, pri if name != "v1" else PRIORS)))
        print(f"  {name}: {nd} draws done", flush=True)
    df = pd.DataFrame(rows)

    targets = {"Kunutsor2020": (0.87, 0.79, 0.95), "Wang2024": (0.83, 0.77, 0.89)}
    summary = {"biomarker_fit": dict(tau_f12_h=tau_f12, tau_tat_h=tau_tat, tau_fib_max_h=tau_fib_max,
                                     beta_coag_implied_p5_p50_p95=[bc_lo, bc_mid, bc_hi], prisco_48h_fold=prisco_48h),
               "model_noise_logIRR": noise, "results": {}}
    for tname, (m, lo, hi) in targets.items():
        sd = np.sqrt(((np.log(hi) - np.log(lo)) / 3.92) ** 2 + noise ** 2)
        df[f"w_{tname}"] = np.exp(-0.5 * ((np.log(df.IRR_cohort) - np.log(m)) / sd) ** 2)
        res = {}
        for name in PRIOR_SETS:
            d = df[df.prior == name]; w = d[f"w_{tname}"].values
            ess = w.sum() ** 2 / (w ** 2).sum()
            r = dict(mean_likelihood=float(w.mean()), ESS=float(ess))
            for col in ["IRR_int_mod", "IRR_int_sed", "IRR_cohort", "risk_int", "p_rr_fitness", "p_beta_coag", "p_beta_micro"]:
                r[col + "_prior"] = [float(x) for x in np.percentile(d[col], [50, 2.5, 97.5])]
                r[col + "_post"] = wq(d[col], w, [0.5, 0.025, 0.975])
            r["P_IRR_int_mod_gt1_prior"] = float((d.IRR_int_mod > 1).mean())
            r["P_IRR_int_mod_gt1_post"] = float((w * (d.IRR_int_mod > 1)).sum() / w.sum())
            r["P_IRR_int_sed_gt1_post"] = float((w * (d.IRR_int_sed > 1)).sum() / w.sum())
            res[name] = r
        for name in ["v1", "anchored", "skeptical"]:
            res[name]["BayesFactor_vs_null"] = res[name]["mean_likelihood"] / res["null"]["mean_likelihood"]
        summary["results"][tname] = res
    df.to_csv("results/calibration_draws.csv", index=False)
    json.dump(summary, open("results/calibration_summary.json", "w"), indent=2)
    pd.DataFrame({k: {kk: v[0] if isinstance(v, tuple) else v for kk, v in pr.items()} for k, pr in
                  {"v1_low": {k: v[0] for k, v in PRIORS.items()}, "anchored_low": {k: v[0] for k, v in ANCHORED.items()},
                   "anchored_mid": {k: v[1] for k, v in ANCHORED.items()}, "anchored_high": {k: v[2] for k, v in ANCHORED.items()}}.items()}
                 ).to_csv("results/anchored_priors.csv")

    for name in ["v1", "anchored", "skeptical", "null"]:
        r = summary["results"]["Kunutsor2020"][name]
        print(f"\n[{name}] ESS {r['ESS']:.0f}  BF vs null {r.get('BayesFactor_vs_null', float('nan')):.2f}")
        for col in ["IRR_cohort", "IRR_int_mod", "IRR_int_sed", "p_rr_fitness", "p_beta_coag"]:
            print(f"   {col:14s} prior {np.round(r[col+'_prior'],3)}  post {np.round(r[col+'_post'],3)}")
        print(f"   P(IRR_int_mod>1) prior {r['P_IRR_int_mod_gt1_prior']:.3f} post {r['P_IRR_int_mod_gt1_post']:.3f};  P(IRR_int_sed>1) post {r['P_IRR_int_sed_gt1_post']:.3f}")
