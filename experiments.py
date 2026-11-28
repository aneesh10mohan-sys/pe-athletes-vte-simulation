"""
Integrity experiments: rerun the v2 model with one thing changed at a time and
compare the results side by side.

usage:
    python experiments.py                  # all experiments, 100 replicates x 2000 people (~10-15 min)
    python experiments.py --reps 30 --n 1000   # quick version (~3 min)
    python experiments.py --only no_micro double_micro   # run selected experiments

Output: experiments/experiments_summary.csv (one row per experiment) and a printed table.
Each experiment says what result to EXPECT before you look, so you can check whether the
model behaves the way its structure says it should.
"""
import argparse, copy, os
import numpy as np, pandas as pd
import params as P, model as M
import run as R

# Each experiment: name, description, prediction, prior overrides (low, central, high),
# group overrides, seed, people per group.
EXPERIMENTS = [
    # --- A. Reproducibility and noise -------------------------------------------------
    dict(name="reference", what="Calibrated v2 model, unchanged",
         expect="IRR about 1.4 (the paper's result at smaller size)"),
    dict(name="seed_7", what="Different random seed (7)", seed=7,
         expect="Close to reference; differences are random noise only"),
    dict(name="seed_99", what="Different random seed (99)", seed=99,
         expect="Close to reference; differences are random noise only"),
    dict(name="bigger_population", what="Twice as many people per group", n_mult=2,
         expect="Close to reference; more people should not change the answer"),

    # --- B. One mechanism at a time ------------------------------------------------------
    dict(name="no_micro", what="Microtrauma switched off", priors={"beta_micro": (0, 0, 0)},
         expect="IRR drops but stays above 1 (coagulation still raises risk)"),
    dict(name="double_micro", what="Microtrauma effect doubled", priors={"beta_micro": (0, 0.7, 1.4)},
         expect="IRR rises clearly above reference"),
    dict(name="no_coag", what="Coagulation effect switched off", priors={"beta_coag": (0, 0, 0)},
         expect="IRR drops; microtrauma now carries most of the effect"),
    dict(name="no_synergy", what="Virchow synergy term switched off", priors={"beta_syn": (0, 0, 0)},
         expect="Small drop in IRR (synergy is about 14% of the excess)"),
    dict(name="weak_stasis", what="Immobility effect cut by about 40%", priors={"beta_stasis": (0.03, 0.05, 0.07)},
         expect="Small drop in IRR (stasis is about 12% of the excess)"),
    dict(name="strong_dehydration", what="Dehydration effect 3x stronger", priors={"beta_hct": (0.06, 0.135, 0.42)},
         expect="Small rise in IRR (hematocrit only rises 1-3 points)"),

    # --- C. Sanity checks with a known right answer --------------------------------------
    dict(name="all_acute_off", what="All exercise-day effects switched off",
         priors={"beta_micro": (0, 0, 0), "beta_coag": (0, 0, 0), "beta_syn": (0, 0, 0),
                 "beta_stasis": (0, 0, 0), "beta_hct": (0, 0, 0)},
         expect="IRR BELOW 1 (about 0.9): only fitness protection is left"),
    dict(name="no_fitness_benefit", what="Exercise gives no background protection",
         priors={"rr_fitness": (1, 1, 1), "rr_fit_extra": (1, 1, 1)},
         expect="Intense vs sedentary rises toward or above 1"),
    dict(name="baseline_risk_x2", what="Everyone's background risk doubled",
         priors={"base_inc_30": (1.2e-4, 2.0e-4, 3.0e-4)},
         expect="Absolute risks about double, but IRR almost unchanged (ratios cancel)"),
    dict(name="less_training", what="Intense group trains 3 days/week instead of 6",
         groups={"intense": {"spw": 3.0}},
         expect="IRR moves toward 1 (fewer exercise days)"),
    dict(name="more_races", what="Intense group runs 12 races/year instead of 6",
         groups={"intense": {"races": 12}},
         expect="IRR rises (more long efforts and post-race sitting)"),
]


def run_one(exp, reps, n, base_priors, base_groups):
    P.PRIORS.clear(); P.PRIORS.update(copy.deepcopy(base_priors))
    for k, v in exp.get("priors", {}).items():
        P.PRIORS[k] = tuple(v) + (P.PRIORS[k][3],)
    for g in P.GROUPS:
        P.GROUPS[g].clear(); P.GROUPS[g].update(copy.deepcopy(base_groups[g]))
    for g, ov in exp.get("groups", {}).items():
        P.GROUPS[g].update(ov)
    seed = exp.get("seed", 2026)
    n_used = int(n * exp.get("n_mult", 1))
    rows = []
    for r in range(reps):
        rows += [row for row in R.one_replicate(r, n_used, seed) if row["scenario"] == "baseline"]
    b = pd.DataFrame(rows)
    q = lambda x: (np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5))
    m, lo, hi = q(b.IRR_int_mod)
    ms, los, his = q(b.IRR_int_sed)
    return {
        "experiment": exp["name"], "what_changed": exp["what"], "prediction": exp["expect"],
        "IRR_intense_vs_moderate": round(m, 3), "IRR_95_low": round(lo, 3), "IRR_95_high": round(hi, 3),
        "share_runs_IRR_above_1": round(float((b.IRR_int_mod > 1).mean()), 3),
        "IRR_intense_vs_sedentary": round(ms, 3), "IRR_sed_95_low": round(los, 3), "IRR_sed_95_high": round(his, 3),
        "risk_sedentary_per100k": round(float(b.risk_sedentary.median()), 1),
        "risk_moderate_per100k": round(float(b.risk_moderate.median()), 1),
        "risk_intense_per100k": round(float(b.risk_intense.median()), 1),
        "seed": seed, "people_per_group": n_used, "replicates": reps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--only", nargs="*", help="names of experiments to run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Start every experiment from the calibrated v2 model. The coagulation scale stays fixed at
    # the v2 reference, so changing a coagulation parameter changes the result rather than
    # being normalised away.
    base_priors = copy.deepcopy(P.PRIORS_V2)
    base_groups = copy.deepcopy(P.GROUPS)
    M._COAG_REF = M._coag_shape({k: v[1] for k, v in base_priors.items()})
    scenarios_backup = dict(P.SCENARIOS)
    P.SCENARIOS.clear(); P.SCENARIOS["baseline"] = {}      # skip prevention scenarios for speed

    todo = [e for e in EXPERIMENTS if not a.only or e["name"] in a.only]
    results = []
    for i, exp in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {exp['name']}: {exp['what']}", flush=True)
        print(f"        predict: {exp['expect']}", flush=True)
        res = run_one(exp, a.reps, a.n, base_priors, base_groups)
        print(f"        result:  IRR {res['IRR_intense_vs_moderate']} "
              f"({res['IRR_95_low']}-{res['IRR_95_high']}), above 1 in "
              f"{res['share_runs_IRR_above_1']:.0%} of runs; intense vs sedentary "
              f"{res['IRR_intense_vs_sedentary']}", flush=True)
        results.append(res)
        pd.DataFrame(results).to_csv(f"{a.out}/experiments_summary.csv", index=False)

    P.SCENARIOS.clear(); P.SCENARIOS.update(scenarios_backup)
    df = pd.DataFrame(results)
    print("\nSummary (saved to", f"{a.out}/experiments_summary.csv):")
    print(df[["experiment", "IRR_intense_vs_moderate", "IRR_95_low", "IRR_95_high",
              "share_runs_IRR_above_1", "IRR_intense_vs_sedentary",
              "risk_moderate_per100k", "risk_intense_per100k"]].to_string(index=False))


if __name__ == "__main__":
    main()
