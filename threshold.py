"""
Break-even (threshold) analysis: how strong must the two least-evidenced mechanisms
(exercise coagulation activation, microtrauma) be for intense training to out-weigh
the protective effect of fitness? Other parameters held at central values.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import central
from model import make_population, simulate_year
import figures as F

def irr(p, n=3000, seed=11):
    r = {}
    for gi, g in enumerate(["moderate", "intense"]):
        rng = np.random.default_rng([seed, gi])
        pop = make_population(g, n, rng)
        r[g] = simulate_year(pop, p, rng)[0][()].mean()
    return r["intense"] / r["moderate"]

if __name__ == "__main__":
    import sys, params as P, model as M
    V2 = "--priors" in sys.argv and sys.argv[sys.argv.index("--priors") + 1] == "v2"
    if V2:
        P.PRIORS.clear(); P.PRIORS.update(P.PRIORS_V2)
        M._COAG_REF = M._coag_shape(central())
    OUT = "results_v2" if V2 else "results"
    ACUTE = ["beta_hct", "beta_coag", "beta_stasis", "beta_micro", "beta_syn"]
    fit = np.linspace(0.5, 1.0, 11); scale = np.linspace(0, 1.5, 11)
    grid = np.zeros((len(scale), len(fit)))
    for i, s in enumerate(scale):
        for j, f in enumerate(fit):
            p = central(); p["rr_fit_extra"] = f
            for k in ACUTE: p[k] *= s
            grid[i, j] = irr(p)
    pd.DataFrame(grid, index=np.round(scale, 2), columns=np.round(fit, 2)).to_csv(f"{OUT}/threshold_grid.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    cs = ax.contourf(fit, scale, grid, levels=np.arange(0.6, 2.21, 0.1), cmap="Blues")
    ln = ax.contour(fit, scale, grid, levels=[1.0], colors=F.C[1], linewidths=2.5)
    ax.clabel(ln, fmt={1.0: "break-even"}, fontsize=9)
    ax.plot(central()["rr_fit_extra"], 1.0, "o", color=F.INK, ms=7)
    ax.annotate("central assumptions", (central()["rr_fit_extra"], 1.0), xytext=(-115, 8), textcoords="offset points", fontsize=8)
    fig.colorbar(cs, label="IRR: intense vs moderate")
    ax.set_xlabel("Extra fitness protection of intense vs moderate training (RR; lower = more)")
    ax.set_ylabel("Strength of all acute exercise effects\n(1 = central assumption)")
    ax.set_title("Break-even: when does intense training raise VTE risk?")
    ax.text(0.52, 0.08, "intense training\nis net protective", fontsize=8, color=F.INK)
    ax.text(0.84, 1.3, "intense training\nraises risk", fontsize=8, color="white")
    ax.grid(False); fig.tight_layout(); fig.savefig(f"{OUT}/fig6_breakeven.png", dpi=200)
    # break-even acute strength for each fitness value
    for j, f in enumerate(fit):
        col = grid[:, j]; k = np.argmax(col >= 1.0)
        be = np.interp(1.0, col, scale) if col[0] < 1 <= col[-1] else (0.0 if col[0] >= 1 else np.nan)
        print(f"extra-fitness RR {f:.2f}: break-even acute strength = {be:.2f}x central")
