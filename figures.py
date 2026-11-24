"""Publication figures (static PNG, light surface, validated categorical palette)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
})
PATH_LABEL = {"hct": "Hemoconcentration\n(dehydration)", "coag": "Coagulation\nactivation",
              "stasis": "Stasis\n(immobility)", "micro": "Endothelial\nmicrotrauma", "syn": "Triad\nsynergy"}


def fig_irr(base, out):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(min(0.5, base.IRR_int_mod.min()), max(base.IRR_int_mod.max(), base.IRR_int_sed.max()), 40)
    ax.hist(base.IRR_int_mod, bins=bins, color=C[0], alpha=0.85, label="Intense vs moderate", rwidth=0.9)
    ax.hist(base.IRR_int_sed, bins=bins, color=C[1], alpha=0.65, label="Intense vs sedentary", rwidth=0.9)
    ax.axvline(1, color=INK, lw=1, ls="--"); ax.text(1.01, ax.get_ylim()[1] * 0.92, "no effect", color=INK2, fontsize=9)
    ax.set_xlabel("Incidence rate ratio of VTE (per replicate)"); ax.set_ylabel("Replicates")
    ax.set_title("H1: Does intense training raise annual VTE risk?"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_H1_irr.png", dpi=200); plt.close(fig)


def fig_shares(base, out):
    keys = list(PATH_LABEL)
    data = [base[f"share_{k}"].values * 100 for k in keys]
    med = [np.median(d) for d in data]
    lo = [np.percentile(d, 10) for d in data]; hi = [np.percentile(d, 90) for d in data]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(keys))
    ax.bar(x, med, color=C[0], width=0.55, edgecolor=SURF, linewidth=2)
    ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor=INK2, capsize=4, lw=1)
    for i, m in enumerate(med): ax.text(i, hi[i] + 1.5, f"{m:.0f}%", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [PATH_LABEL[k] for k in keys], fontsize=9)
    ax.set_ylabel("Share of exercise-attributable excess risk (%)")
    ax.set_title("H2: Which Virchow pathway drives the excess? (median, 10–90th pct)")
    fig.tight_layout(); fig.savefig(f"{out}/fig2_H2_pathway_shares.png", dpi=200); plt.close(fig)


def fig_dose(dr, out):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    spec = [("bml", "Body-mass loss from dehydration (%)", "H3: Dehydration"),
            ("imm", "Extra hours immobile after exercise", "H4: Post-exercise immobility"),
            ("load", "Repetitive load (1 = one marathon run)", "H5: Microtrauma load")]
    for ax, (k, xl, t) in zip(axs, spec):
        ax.fill_between(dr[k], dr[k + "_lo"], dr[k + "_hi"], color=C[0], alpha=0.18, lw=0, label="90% uncertainty")
        ax.plot(dr[k], dr[k + "_central"], color=C[0], label="central estimate")
        ax.axhline(1.1, color=INK2, lw=1, ls=":"); ax.set_xlabel(xl); ax.set_title(t)
    axs[0].set_ylabel("Relative risk on that day"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[0].text(0.1, 1.115, "10% threshold", color=INK2, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig3_H3-5_dose_response.png", dpi=200); plt.close(fig)


def fig_scen(df, base, out):
    scen = [s for s in df.scenario.unique()]
    red = [100 * np.median(1 - df[df.scenario == s].risk_intense.values / base.risk_intense.values) for s in scen]
    risk = [df[df.scenario == s].risk_intense.median() for s in scen]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(scen))[::-1]
    ax.barh(y, risk, color=C[0], height=0.55, edgecolor=SURF, linewidth=2)
    ax.axvline(df[df.scenario == "baseline"].risk_moderate.median(), color=C[1], lw=2, ls="--")
    ax.text(df[df.scenario == "baseline"].risk_moderate.median(), y[0] + 0.45, " moderate exerciser", color=INK2, fontsize=8)
    for yi, r, rd in zip(y, risk, red):
        ax.text(r, yi, f"  {r:.1f}" + (f"  (−{rd:.0f}%)" if rd > 0.5 else ""), va="center", fontsize=9)
    ax.set_yticks(y, [s.replace("_", " ") for s in scen]); ax.set_xlabel("VTE per 100,000 intense athletes per year (median)")
    ax.set_title("Prevention scenarios for intense athletes"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(risk) * 1.3)
    fig.tight_layout(); fig.savefig(f"{out}/fig4_prevention_scenarios.png", dpi=200); plt.close(fig)


def fig_sens(sens, out):
    s = sens.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(range(len(s)), s.values, color=[C[0] if v > 0 else C[1] for v in s.values], height=0.6)
    ax.set_yticks(range(len(s)), [i.replace("p_", "") for i in s.index], fontsize=9)
    ax.axvline(0, color=INK, lw=0.8); ax.set_xlabel("Partial rank correlation with log IRR (intense vs moderate)")
    ax.set_title("Which assumptions decide H1?"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_sensitivity.png", dpi=200); plt.close(fig)


def make_all(df, base, dr, sens, out):
    fig_irr(base, out); fig_shares(base, out); fig_dose(dr, out); fig_scen(df, base, out); fig_sens(sens, out)
