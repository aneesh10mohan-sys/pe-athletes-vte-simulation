// JavaScript port of vte_sim/model.py + params.PRIORS_V2 (calibrated v2, 2026-11-14).
// Keep in sync with the Python model; test_parity.js checks they agree.

const PRIORS = {
  base_inc_30:   [0.6e-4, 1.0e-4, 1.5e-4],
  rr_age_decade: [1.4, 1.7, 2.0],
  rr_bmi_5:      [1.3, 1.5, 1.8],
  rr_fvl:        [3.0, 4.5, 7.0],
  rr_ocp:        [2.0, 3.0, 4.5],
  rr_fitness:    [0.67, 0.78, 0.89],   // v2: calibrated to Kunutsor 2020
  rr_fit_extra:  [0.85, 0.95, 1.00],   // v2: Wang 2024 dose-response flattens
  k_pv:          [1.0, 1.6, 2.5],
  beta_hct:      [0.019, 0.045, 0.14],
  beta_coag:     [0.0, 0.71, 1.08],    // v2: derived from marker-VTE associations
  fib_ratio:     [0.6, 0.9, 1.3],
  tau_coag_h:    [12.7, 17.1, 21.8],   // v2: fitted to WSER / Prisco 1998
  tau_fib_h:     [1.0, 3.0, 6.8],      // v2: tPA clears within a day
  beta_stasis:   [0.05, 0.083, 0.12],
  active_recov:  [0.3, 0.5, 0.7],
  beta_micro:    [0.0, 0.35, 0.7],
  tau_micro_d:   [1.0, 2.0, 4.0],
  beta_syn:      [0.0, 0.3, 0.8],
};
const SPORT_REP = { running: 1.0, triathlon: 0.8, cycling: 0.6, mixed: 0.6, swimming: 0.35 };
const REF_INTEN = 0.85, REF_DUR = 4.0;

// --- random numbers (seeded, so results are stable for the same inputs) ---
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normal(rng) { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }
function triangular(rng, a, c, b) {
  if (a >= b) return c;
  const u = rng(), f = (c - a) / (b - a);
  return u < f ? a + Math.sqrt(u * (b - a) * (c - a)) : b - Math.sqrt((1 - u) * (b - a) * (b - c));
}
function centralParams() { const p = {}; for (const k in PRIORS) p[k] = PRIORS[k][1]; return p; }
function sampleParams(rng) { const p = {}; for (const k in PRIORS) p[k] = triangular(rng, ...PRIORS[k]); return p; }

// --- physiology (same equations as model.py) ---
function coagShape(p) {
  const tc = p.tau_coag_h, tf = p.tau_fib_h, r = p.fib_ratio;
  const tstar = r > 1 ? Math.max(0, Math.log(r) / (1 / tf - 1 / tc)) : 0;
  return tc * Math.exp(-tstar / tc) - r * tf * Math.exp(-tstar / tf);
}
const COAG_REF = coagShape(centralParams());

function fitnessLevel(person) {
  const weeklyHours = person.spw * person.dur + (person.races * person.raceH) / 52;
  return weeklyHours < 1 ? 0 : weeklyHours < 5 ? 1 : 2;
}

// Simulate one year for one person under parameter set p.
// off: set of pathways to switch off ("hct","coag","stasis","micro","syn").
function simulateYear(person, p, rng, off = new Set()) {
  const nS = Math.max(Math.round(52 * person.spw), 1);
  const races = Math.min(person.races, nS);
  const raceIdx = new Set();
  if (races > 0) for (let j = 1; j <= races; j++) raceIdx.add(Math.floor(j * (nS - 1) / (races + 1)));
  const dt = 365 / nS, decay = Math.exp(-dt / p.tau_micro_d);
  const coagScale = Math.max(coagShape(p), 0) / COAG_REF;
  const rep = SPORT_REP[person.sport] ?? 0.6;
  const hct0 = person.hct0;
  let micro = 0, excess = 0;
  const contrib = { hct: 0, coag: 0, stasis: 0, micro: 0, syn: 0 };

  for (let k = 0; k < nS; k++) {
    let inten, dur, immob;
    if (raceIdx.has(k)) {
      inten = 0.85;
      dur = person.raceH * (0.7 + 0.8 * rng());
      immob = Math.exp(Math.log(Math.max(person.raceSit, 0.1)) + 0.5 * normal(rng));
    } else {
      inten = Math.min(Math.max(person.inten + (rng() - 0.5) * 0.15, 0.2), 1.0);
      dur = person.dur * (0.67 + 0.66 * rng());
      immob = Math.exp(Math.log(Math.max(person.sitAfter, 0.1)) + 0.6 * normal(rng));
    }
    if (person.activeRecovery) immob *= 1 - p.active_recov;
    const stasis = Math.max(immob - 1, 0);

    const sweat = (0.2 + 1.4 * inten) * dur;
    const bml = Math.min(Math.max(sweat * (1 - person.hydr) / person.mass * 100, 0), 8);
    const dpv = Math.min(Math.max(p.k_pv * bml / 100, 0), 0.35);
    const dhct = 100 * hct0 / (hct0 + (100 - hct0) * (1 - dpv)) - hct0;
    const cnet = Math.pow(inten / REF_INTEN, 2) * (dur / REF_DUR) * coagScale;
    const load = rep * inten * dur / (REF_INTEN * REF_DUR);
    micro = micro * decay + load;

    const hyper = (dhct / 2 + cnet) / 2;
    const t = {
      hct: p.beta_hct * dhct,
      coag: p.beta_coag * cnet,
      stasis: p.beta_stasis * stasis,
      micro: p.beta_micro * micro,
      syn: p.beta_syn * Math.cbrt(Math.max(hyper, 0) * (stasis / 4) * micro),
    };
    let lr = 0;
    for (const key in t) if (!off.has(key)) { lr += t[key]; contrib[key] += t[key]; }
    excess += Math.expm1(lr);
  }

  let prr = Math.pow(p.rr_age_decade, (person.age - 30) / 10);
  prr *= Math.pow(p.rr_bmi_5, Math.max(person.bmi - 25, 0) / 5);
  if (person.fvl) prr *= p.rr_fvl;
  if (person.ocp) prr *= p.rr_ocp;
  const fit = fitnessLevel(person);
  if (fit >= 1) prr *= p.rr_fitness;
  if (fit >= 2) prr *= p.rr_fit_extra;
  const hDay = p.base_inc_30 / 365;
  return { risk: -Math.expm1(-hDay * prr * (365 + excess)), contrib };
}

// Full calculator: central estimate + uncertainty band + pathway breakdown.
function calculate(person, { draws = 400, seed = 12345 } = {}) {
  const rngC = mulberry32(seed);
  const reps = 20;
  const avg = (fn) => { let s = 0; for (let i = 0; i < reps; i++) s += fn(); return s / reps; };
  const pc = centralParams();
  const central = avg(() => simulateYear(person, pc, rngC).risk);
  const allOff = new Set(["hct", "coag", "stasis", "micro", "syn"]);
  const noExercise = avg(() => simulateYear(person, pc, rngC, allOff).risk);
  const share = {};
  const excess = central - noExercise;
  for (const k of allOff) {
    const without = avg(() => simulateYear(person, pc, rngC, new Set([k])).risk);
    share[k] = excess > 0 ? (central - without) / excess : 0;
  }
  const rngU = mulberry32(seed + 1);
  const samples = [];
  for (let i = 0; i < draws; i++) samples.push(simulateYear(person, sampleParams(rngU), rngU).risk);
  samples.sort((a, b) => a - b);
  const q = (f) => samples[Math.min(samples.length - 1, Math.floor(f * samples.length))];
  return { central, low: q(0.05), median: q(0.5), high: q(0.95), noExercise, share, fitness: fitnessLevel(person) };
}

if (typeof module !== "undefined") module.exports = { calculate, simulateYear, centralParams, mulberry32, fitnessLevel };
