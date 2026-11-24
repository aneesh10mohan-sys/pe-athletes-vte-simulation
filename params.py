"""
Parameter priors for the Exercise -> Virchow's triad -> VTE (PE/DVT) simulation.

Every parameter has (low, central, high) and a note on where the value comes from.
"VERIFY" = a literature-anchored placeholder that should be checked against the
cited source before publication. "ASSUMPTION" = no good data exists; the wide
range is deliberate and the sensitivity analysis tells you how much it matters.
"""

PRIORS = {
    # ---- Baseline epidemiology -------------------------------------------------
    "base_inc_30":   (0.6e-4, 1.0e-4, 1.5e-4,
                      "Annual VTE incidence at age 30, no risk factors. VERIFY: Heit 2015 (~1/10,000/yr young adults)."),
    "rr_age_decade": (1.4, 1.7, 2.0,   "VTE risk multiplier per decade of age. VERIFY: Heit 2015."),
    "rr_bmi_5":      (1.3, 1.5, 1.8,   "RR per 5 kg/m2 above 25. VERIFY: Hotoleanu 2025 / obesity meta-analyses."),
    "rr_fvl":        (3.0, 4.5, 7.0,   "Factor V Leiden heterozygote RR. VERIFY (standard thrombophilia literature)."),
    "rr_ocp":        (2.0, 3.0, 4.5,   "Combined oral contraceptive RR. VERIFY."),
    "rr_fitness":    (0.65, 0.80, 0.95,"Protective RR of regular activity on baseline risk. VERIFY: Kabrhel 2011 (inactivity & PE)."),
    "rr_fit_extra":  (0.70, 0.90, 1.00,"Extra protective multiplier for high-volume vs moderate training (dose-response of fitness). ASSUMPTION."),

    # ---- H3: dehydration -> hemoconcentration ------------------------------------
    "k_pv":          (1.0, 1.6, 2.5,   "% plasma-volume drop per 1% body-mass loss. ASSUMPTION (exercise-physiology range)."),
    "beta_hct":      (0.019, 0.045, 0.14,
                      "log-RR per +1 hematocrit point (acute). Low end = chronic Hct HR 1.1/5pts; high = 2.0/5pts. ASSUMPTION/VERIFY (Tromso Hct-VTE studies)."),

    # ---- H1/H2: exercise-induced coagulation vs fibrinolysis ------------------
    "beta_coag":     (0.0, 0.5, 1.1,   "log-RR for net procoagulant window equal to one marathon (=1.0). ASSUMPTION; Hanke 2010 shows activation, not risk."),
    "fib_ratio":     (0.6, 0.9, 1.3,   "Peak fibrinolysis / peak coagulation activation. ASSUMPTION (Hanke 2010: both activated)."),
    "tau_coag_h":    (4.0, 6.0, 10.0,  "Decay time (h) of coagulation activation. ASSUMPTION."),
    "tau_fib_h":     (0.8, 1.5, 3.0,   "Decay time (h) of fibrinolytic response (fast). ASSUMPTION."),

    # ---- H4: post-exercise immobility -> stasis ---------------------------------
    "beta_stasis":   (0.05, 0.083, 0.12,
                      "log-RR per hour of immobility. Central = +18% per 2 h of travel. VERIFY: travel-VTE meta-analyses / CDC Yellow Book 2025."),
    "active_recov":  (0.3, 0.5, 0.7,   "Fraction of immobility removed by active recovery. ASSUMPTION."),

    # ---- H5: repetitive load -> vascular microtrauma ------------------------
    "beta_micro":    (0.0, 0.35, 0.7,  "log-RR for microtrauma load equal to one marathon (=1.0). ASSUMPTION (mechanism only, Miele 2024)."),
    "tau_micro_d":   (1.0, 2.0, 4.0,   "Endothelial recovery time constant (days). ASSUMPTION."),

    # ---- H2: Virchow synergy ----------------------------------------------------
    "beta_syn":      (0.0, 0.3, 0.8,   "log-RR interaction when all three triad arms are elevated together. ASSUMPTION."),
}


def central():
    return {k: v[1] for k, v in PRIORS.items()}


def sample(rng):
    """Draw one parameter set from triangular(low, central, high) priors."""
    return {k: rng.triangular(v[0], v[1], v[2]) if v[0] < v[2] else v[1]
            for k, v in PRIORS.items()}


# Synthetic population / exposure profiles (inputs a user would change)
GROUPS = {
    #                  sessions/wk, intensity (frac VO2max), hours/session, races/yr, race hours, BMI mean, active
    "sedentary":   dict(spw=0.5, inten=(0.35, 0.50), dur=(0.3, 0.7), races=0, race_h=0.0, bmi=27.5, fit=0),
    "moderate":    dict(spw=3.0, inten=(0.50, 0.65), dur=(0.5, 1.0), races=0, race_h=0.0, bmi=24.0, fit=1),
    "intense":     dict(spw=6.0, inten=(0.65, 0.85), dur=(1.0, 2.0), races=6, race_h=4.0, bmi=22.5, fit=2),
}

SPORTS = {  # repetitiveness of the movement (microtrauma weight), share in intense group
    "running":   (1.0, 0.45),
    "triathlon": (0.8, 0.20),
    "cycling":   (0.6, 0.25),
    "swimming":  (0.35, 0.10),
}

SCENARIOS = {
    # name: overrides applied to behaviour
    "baseline":         {},
    "hydration":        {"hydr_mean": 0.90},            # replace ~90% of sweat loss
    "active_recovery":  {"active_recovery_share": 1.0},  # everyone does active recovery
    "no_travel_sitting":{"race_immob_h": 1.5},          # avoid long sitting after races
    "combined":         {"hydr_mean": 0.90, "active_recovery_share": 1.0, "race_immob_h": 1.5},
}


# ---------------------------------------------------------------------------- v2 (calibrated)
# Produced by calibrate.py (2026-11-14). Clotting kinetics fitted to marathon/ultramarathon
# biomarker time courses; beta_coag derived from marker-VTE associations; fitness protection =
# posterior after weighting to Kunutsor 2020 (most vs least active RR 0.87, 0.79-0.95).
PRIORS_V2 = dict(PRIORS)
PRIORS_V2.update({
    "tau_coag_h":   (12.7, 17.1, 21.8, "Fitted: F1+2 and TAT decay after Western States 100; consistent with Prisco 1998."),
    "tau_fib_h":    (1.0, 3.0, 6.8,    "Fitted: tPA back to baseline within a day (Schobersberger 1996; WSER ratios)."),
    "beta_coag":    (0.0, 0.71, 1.08,  "Derived: marathon marker rise x D-dimer/F1+2-VTE associations (Eichinger 2010; ARIC). Lower bound 0 = full fibrinolytic compensation."),
    "rr_fitness":   (0.67, 0.78, 0.89, "Calibrated posterior to Kunutsor 2020."),
    "rr_fit_extra": (0.85, 0.95, 1.00, "Informed: dose-response flattens at higher activity (Wang 2024)."),
})
