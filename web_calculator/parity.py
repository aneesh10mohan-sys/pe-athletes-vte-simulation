import sys, json, numpy as np
sys.path.insert(0, "/home/claude/vte_sim")
import params, model
from params import central
params.PRIORS.clear(); params.PRIORS.update(params.PRIORS_V2)
model._COAG_REF = model._coag_shape(central())
cases = json.load(open("cases.json"))
out = []
for c in cases:
    params.GROUPS["test"] = dict(spw=c["spw"], inten=(c["inten"]-0.075, c["inten"]+0.075), dur=(0.67*c["dur"], 1.33*c["dur"]),
                                 races=c["races"], race_h=c["raceH"], bmi=c["bmi"], fit=0)
    model.GROUPS = params.GROUPS
    n = 20000; rng = np.random.default_rng(1)
    pop = model.make_population("test", n, rng, race_immob_h=c["raceSit"])
    pop.update(age=np.full(n, c["age"]), bmi=np.full(n, c["bmi"]), mass=np.full(n, c["mass"]), hct0=np.full(n, c["hct0"]),
               fvl=np.full(n, c["fvl"]), ocp=np.full(n, c["ocp"]), fit=np.full(n, c["fitLevel"]),
               active_rec=np.full(n, c["activeRecovery"]), hydr=np.full(n, c["hydr"]), rep=np.full(n, c["rep"]))
    risk, _ = model.simulate_year(pop, central(), rng)
    out.append(float(risk[()].mean()))
print(json.dumps(out))
