const { simulateYear, centralParams, mulberry32, fitnessLevel } = require("./model.js");
const { execSync } = require("child_process");
const base = { age: 35, sex: "male", bmi: 23, mass: 72, hct0: 44, fvl: false, ocp: false, spw: 6, inten: 0.75, dur: 1.5,
  races: 6, raceH: 4, raceSit: 5, sitAfter: 1, hydr: 0.6, activeRecovery: false, sport: "running", rep: 1.0 };
const cases = [
  base,
  { ...base, age: 25, sex: "female", ocp: true, mass: 60, hct0: 40, sport: "cycling", rep: 0.6, hydr: 0.4 },
  { ...base, fvl: true, races: 0, spw: 3, inten: 0.6, dur: 0.75, activeRecovery: true },
  { ...base, age: 48, bmi: 29, spw: 0.5, inten: 0.45, dur: 0.5, races: 0 },
].map(c => ({ ...c, fitLevel: fitnessLevel(c) }));
require("fs").writeFileSync("cases.json", JSON.stringify(cases));
const py = JSON.parse(execSync("python3 parity.py").toString());
const p = centralParams(), rng = mulberry32(7);
cases.forEach((c, i) => {
  let s = 0; const N = 20000; for (let j = 0; j < N; j++) s += simulateYear(c, p, rng).risk;
  const js = s / N;
  console.log(`case ${i}: JS ${(js*1e5).toFixed(2)}  PY ${(py[i]*1e5).toFixed(2)} per 100k  diff ${(100*(js/py[i]-1)).toFixed(2)}%`);
});
