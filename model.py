"""
Mechanistic Monte Carlo model: exercise exposure -> Virchow's triad -> VTE risk.

Pipeline for each synthetic person and each exercise session:
  sweat loss & fluid intake -> body-mass loss -> plasma-volume drop -> hematocrit rise   (H3)
  intensity x duration      -> coagulation activation, partly offset by fibrinolysis     (H1/H2)
  post-exercise sitting     -> venous stasis hours                                        (H4)
  repetitive load           -> cumulative endothelial microtrauma (decays between days)   (H5)
  all three arms together   -> synergy term (Virchow interaction)                         (H2)
Each session multiplies the person's VTE hazard for that day; on non-exercise days the
person carries their baseline hazard (age, BMI, thrombophilia, OCP, fitness protection).
"""
import numpy as np
from params import GROUPS, SPORTS, central

REF_INTEN, REF_DUR = 0.85, 4.0          # "one marathon" reference effort
PATHWAYS = ("hct", "coag", "stasis", "micro", "syn")


# ----------------------------------------------------------------------------- population
def make_population(group, n, rng, hydr_mean=0.60, active_recovery_share=0.30, race_immob_h=5.0):
    g = GROUPS[group]
    female = rng.random(n) < 0.5
    pop = dict(
        group=group, n=n,
        age=rng.uniform(18, 50, n),
        female=female,
        bmi=np.clip(rng.normal(g["bmi"], 3.0, n), 17, 42),
        mass=np.clip(rng.normal(np.where(female, 62, 75), 8, n), 45, 110),
        hct0=rng.normal(np.where(female, 40, 44), 2.5, n),
        fvl=rng.random(n) < 0.05,
        ocp=female & (rng.random(n) < 0.30),
        fit=np.full(n, g["fit"]),
        active_rec=rng.random(n) < active_recovery_share,
    )
    conc = 8.0
    pop["hydr"] = rng.beta(hydr_mean * conc, (1 - hydr_mean) * conc, n)
    if group == "intense":
        names = list(SPORTS); probs = np.array([SPORTS[s][1] for s in names])
        idx = rng.choice(len(names), n, p=probs / probs.sum())
        pop["sport"] = np.array(names)[idx]
        pop["rep"] = np.array([SPORTS[s][0] for s in names])[idx]
    else:
        pop["sport"] = np.full(n, "mixed")
        pop["rep"] = np.full(n, 0.6)
    pop["race_immob_h"] = race_immob_h
    return pop


# ----------------------------------------------------------------------------- physiology
def hemoconc(inten, dur, hydr, mass, hct0, p):
    """H3: returns (body-mass loss %, hematocrit rise in points)."""
    sweat = (0.2 + 1.4 * inten) * dur                         # litres
    bml = np.clip(sweat * (1 - hydr) / mass * 100, 0, 8)      # % body mass
    dpv = np.clip(p["k_pv"] * bml / 100, 0, 0.35)             # fractional plasma-volume drop
    hct = 100 * hct0 / (hct0 + (100 - hct0) * (1 - dpv))
    return bml, hct - hct0


def _coag_shape(p):
    """Area of the net procoagulant window (coag decays slowly, fibrinolysis fast)."""
    tc, tf, r = p["tau_coag_h"], p["tau_fib_h"], p["fib_ratio"]
    tstar = max(0.0, np.log(r) / (1 / tf - 1 / tc)) if r > 1 else 0.0
    return tc * np.exp(-tstar / tc) - r * tf * np.exp(-tstar / tf)


_COAG_REF = _coag_shape(central())


def coag_net(inten, dur, p):
    """H1/H2 hypercoagulability: net coagulation window, 1.0 = marathon w/ central params."""
    amp = (inten / REF_INTEN) ** 2 * (dur / REF_DUR)
    return amp * max(_coag_shape(p), 0) / _COAG_REF


def session_log_rr(dhct, cnet, stasis_h, micro, p, off=()):
    """Per-pathway log relative risk for the day of a session. `off` knocks out pathways."""
    terms = {
        "hct":    p["beta_hct"] * dhct,
        "coag":   p["beta_coag"] * cnet,
        "stasis": p["beta_stasis"] * stasis_h,
        "micro":  p["beta_micro"] * micro,
    }
    hyper = (dhct / 2.0 + cnet) / 2.0
    terms["syn"] = p["beta_syn"] * np.cbrt(np.maximum(hyper, 0) * (stasis_h / 4.0) * micro)
    return sum(0 if k in off else v for k, v in terms.items())


def person_rr(pop, p):
    rr = p["rr_age_decade"] ** ((pop["age"] - 30) / 10)
    rr *= p["rr_bmi_5"] ** (np.maximum(pop["bmi"] - 25, 0) / 5)
    rr *= np.where(pop["fvl"], p["rr_fvl"], 1.0)
    rr *= np.where(pop["ocp"], p["rr_ocp"], 1.0)
    rr *= np.where(pop["fit"] >= 1, p["rr_fitness"], 1.0)
    rr *= np.where(pop["fit"] >= 2, p["rr_fit_extra"], 1.0)
    return rr


# ----------------------------------------------------------------------------- one year
def simulate_year(pop, p, rng, knockouts=((),)):
    """
    Returns dict: knockout-tuple -> per-person annual risk array,
    plus session-level summaries for dose-response diagnostics.
    """
    g = GROUPS[pop["group"]]
    n = pop["n"]
    n_s = max(int(round(52 * g["spw"])), 1)
    race_idx = set(np.linspace(0, n_s - 1, g["races"] + 2, dtype=int)[1:-1]) if g["races"] else set()
    dt = 365.0 / n_s
    decay = np.exp(-dt / p["tau_micro_d"])

    excess = {k: np.zeros(n) for k in knockouts}
    micro = np.zeros(n)
    ar = pop["active_rec"]
    log_bml, log_imm, log_load = [], [], []

    for k in range(n_s):
        is_race = k in race_idx
        if is_race:
            inten = np.full(n, 0.85)
            dur = g["race_h"] * rng.uniform(0.7, 1.5, n)
            immob = rng.lognormal(np.log(pop["race_immob_h"]), 0.5, n)
        else:
            inten = rng.uniform(*g["inten"], n)
            dur = rng.uniform(*g["dur"], n)
            immob = rng.lognormal(np.log(1.0), 0.6, n)
        immob = immob * np.where(ar, 1 - p["active_recov"], 1.0)
        stasis = np.maximum(immob - 1.0, 0)                        # hours beyond normal sitting

        bml, dhct = hemoconc(inten, dur, pop["hydr"], pop["mass"], pop["hct0"], p)
        cnet = coag_net(inten, dur, p)
        load = pop["rep"] * inten * dur / (1.0 * REF_INTEN * REF_DUR)
        micro = micro * decay + load

        for ko in knockouts:
            excess[ko] += np.expm1(session_log_rr(dhct, cnet, stasis, micro, p, off=ko))
        if k % 4 == 0:
            log_bml.append(bml[:2000]); log_imm.append(stasis[:2000]); log_load.append(micro[:2000])

    h_day = p["base_inc_30"] / 365.0
    prr = person_rr(pop, p)
    risk = {ko: -np.expm1(-h_day * prr * (365.0 + excess[ko])) for ko in knockouts}
    diag = dict(bml=np.concatenate(log_bml), stasis=np.concatenate(log_imm), micro=np.concatenate(log_load))
    return risk, diag
