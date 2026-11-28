# Exercise-Associated VTE Risk: A Mechanistic Simulation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24517304.svg)](https://doi.org/10.5281/zenodo.24517304)

Companion code for **"Pulmonary Embolism in Athletes: A Review and Simulation Study of the Thrombotic Risks of Intense Exercise"** by Aneesh A. Mohan.

The model turns the paper's hypotheses into numbers. It simulates a year of exercise sessions for synthetic sedentary, moderate and intense-endurance adults aged 18–50. Each session raises that day's clot risk through four pathways mapped to Virchow's triad:

| Pathway | Virchow arm | Hypothesis |
| --- | --- | --- |
| Dehydration → hemoconcentration | Hypercoagulability | H3 |
| Coagulation activation (net of fibrinolysis) | Hypercoagulability | H1, H2 |
| Post-exercise immobility | Venous stasis | H4 |
| Repetitive-movement microtrauma | Endothelial injury | H5 |

Risk is computed with a hazard model. Each person has a background daily hazard set by age, BMI, Factor V Leiden, estrogen-containing contraception and fitness. On an exercise day that hazard is multiplied by e^L, where

```
L = β_hct·ΔHct + β_coag·C_net + β_stasis·S + β_micro·E + β_syn·∛(H·(S/4)·E)
annual risk = 1 − exp(−h·(365 + Σ_sessions (e^L − 1)))
```

Every parameter has a (low, central, high) range with its source in `params.py`. Version 2 priors are calibrated to biomarker data and to a meta-analysis of 14 cohort studies (Kunutsor et al., 2020).

## Main results (version 2 priors, 250 replicates)

- Intense vs moderate exercisers: annual VTE incidence rate ratio **1.41** (95% interval 1.07–1.94)
- Intense athletes vs sedentary adults: **0.96** (0.69–1.38)
- Annual incidence per 100,000: sedentary 30.1, moderate 20.5, intense 28.5

## Reproduce

Requires Python 3 with `numpy`, `pandas` and `matplotlib`:

```bash
pip install -r requirements.txt
python run.py --priors v2 --reps 250 --n 4000 --seed 2026 --out results_v2   # main results
python calibrate.py --draws 1200 --n 1500                                    # cohort calibration
python threshold.py                                                          # break-even analysis
```

Seeds are fixed, so results are reproducible.

## Integrity experiments

`experiments.py` reruns the calibrated model with one thing changed at a time (seeds, population size, each mechanism switched off or strengthened, and sanity checks with a known right answer) and saves a comparison table to `experiments/experiments_summary.csv`.

```bash
python experiments.py                     # all 15 experiments, about 5 minutes
python experiments.py --reps 30 --n 1000  # quick version, about 1 minute
python experiments.py --only no_micro all_acute_off
```

## Files

- `params.py`: parameter ranges and sources (v1 original, v2 calibrated), groups, prevention scenarios
- `model.py`: synthetic population, session-level physiology, hazard model
- `run.py`: replicates, hypothesis tests, sensitivity analysis (PRCC), power calculation, outputs
- `calibrate.py`: biomarker fit and cohort calibration
- `threshold.py`: break-even analysis
- `figures.py`: figures
- `experiments.py`: integrity experiments (one change at a time)
- `results/`, `results_v2/`: outputs for v1 and v2 priors
- `web_calculator/`: browser version of the model for a single person (`index.html`), with a parity test against the Python model

All data are synthetic, generated from published estimates. No individual-level human data were used.

## Limitations

The size of the microtrauma effect has no direct data and is an assumption. The coagulation effect applies long-term marker associations to a short post-race spike. The model has not been validated against observed VTE incidence in athletes. See the paper and its supplementary material for details.

## Use of AI tools

The research question, literature review, Virchow's triad framework and hypotheses are the author's own work. The simulation code was developed with the assistance of an AI model (Claude, Anthropic). The author reviewed the model and takes responsibility for it.

## Citation

Archived on Zenodo: https://doi.org/10.5281/zenodo.24517304. See `CITATION.cff`. License: MIT (see `LICENSE`).
