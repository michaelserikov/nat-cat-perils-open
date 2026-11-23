"""Wildfire hazard from EFFIS burnt-area polygons 2016-2025: country totals, fire size tail and the artificial-surface share."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perils import stats as S  # noqa: E402
from perils import wildfires as W  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
df = W.load(Path.home() / "z_data" / "z_sre" / "effis")
df = df[df.year.between(2016, 2025)]
MIN_HA = 30.0
d = df[df.AREA_HA >= MIN_HA]
years = sorted(d.year.unique())
out = {"fires_ge_30ha": int(len(d)), "years": [int(years[0]), int(years[-1])], "burnt_ha_total": float(d.AREA_HA.sum())}

yr = W.yearly(df, MIN_HA)
yr.to_csv(RES / "wildfire_by_year.csv")
out["by_year"] = {int(k): {"fires": int(v.fires), "burnt_kha": float(round(v.burnt_ha / 1000, 0)), "largest_ha": float(round(v.max_ha, 0))} for k, v in yr.iterrows()}

# countries (EFFIS codes): totals, annual mean, largest fire, coefficient of variation of the annual burnt area
covered = d.groupby("COUNTRY").year.nunique()
full = set(covered[covered == len(years)].index)
out["countries_excluded_incomplete_coverage"] = {c: int(n) for c, n in covered.items() if n < len(years)}
rows = []
for c, g in d[d.COUNTRY.isin(full)].groupby("COUNTRY"):
    a = g.groupby("year").AREA_HA.sum().reindex(years, fill_value=0.0)
    rows.append({"country": c, "fires": int(len(g)), "mean_burnt_kha_per_yr": round(a.mean() / 1000, 1), "cv_annual": round(float(a.std(ddof=1) / a.mean()), 2),
                 "largest_fire_ha": round(g.AREA_HA.max(), 0), "worst_year": int(a.idxmax()), "worst_year_share": round(float(a.max() / a.sum()), 2)})
ct = pd.DataFrame(rows).sort_values("mean_burnt_kha_per_yr", ascending=False)
ct.to_csv(RES / "wildfire_by_country.csv", index=False)
out["top_countries"] = ct.head(8).to_dict("records")

# fire size tail for the Mediterranean EU group: fires above 500 ha
med = d[d.COUNTRY.isin(["PT", "ES", "FR", "IT", "EL", "GR", "HR", "BG", "RO", "CY"])]
big = med[med.AREA_HA >= 500]
cnt = big.groupby("year").size().reindex(years, fill_value=0).values
out["med_eu"] = {"fires_ge_30ha": int(len(med)), "fires_ge_500ha": int(len(big)), "per_year_ge_500ha": round(float(cnt.mean()), 1), "dispersion_ge_500ha": S.dispersion(cnt)}
out["med_eu"]["shape_by_threshold"] = {}
for thr in (500.0, 1000.0, 2000.0, 5000.0):
    g = S.fit_gpd(med.AREA_HA.values, thr)
    out["med_eu"]["shape_by_threshold"][str(int(thr))] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in g.items()}
# tail model: fires above 5000 ha as Poisson-GPD, smaller fires taken as observed
THR = 5000.0
gp = S.fit_gpd(med.AREA_HA.values, THR)
rate = (med.AREA_HA > THR).sum() / len(years)
small_by_year = med[med.AREA_HA <= THR].groupby("year").AREA_HA.sum().reindex(years, fill_value=0.0)
from scipy import stats as sps  # noqa: E402
cnt5 = med[med.AREA_HA > THR].groupby("year").size().reindex(years, fill_value=0).values
m5, v5 = cnt5.mean(), cnt5.var(ddof=1)
rng = np.random.default_rng(1)
NSIM = 20000
if v5 > m5:                                   # negative binomial with the observed mean and variance
    nb_p = m5 / v5
    nb_r = m5 * nb_p / (1 - nb_p)
    ncount = rng.negative_binomial(nb_r, nb_p, NSIM)
else:
    ncount = rng.poisson(m5, NSIM)
tot = np.array([(THR + sps.genpareto.rvs(gp["shape"], scale=gp["scale"], size=k, random_state=rng)).sum() if k else 0.0 for k in ncount])
sim_total = tot + rng.choice(small_by_year.values, NSIM)        # smaller fires: resampled years
out["med_eu"]["count_above_threshold_by_year"] = [int(x) for x in cnt5]
obs_total = med.groupby("year").AREA_HA.sum().reindex(years, fill_value=0.0)
out["med_eu"]["tail_model"] = {"threshold_ha": THR, "fires_per_year_above": round(float(rate), 1), "shape": round(gp["shape"], 2),
                               "shape_90pct_range": [round(gp["shape_lo"], 2), round(gp["shape_hi"], 2)], "scale_ha": round(gp["scale"], 0)}
out["med_eu"]["annual_burnt_ha"] = {"observed_mean": round(float(obs_total.mean()), 0), "observed_max": round(float(obs_total.max()), 0),
                                    "simulated_mean": round(float(sim_total.mean()), 0), "simulated_p95": round(float(np.percentile(sim_total, 95)), 0),
                                    "simulated_p99": round(float(np.percentile(sim_total, 99)), 0)}
out["med_eu"]["largest_fire_return_level_ha"] = {str(T): round(S.gpd_return_level(rate, THR, gp["shape"], gp["scale"], T), 0) for T in (10, 50, 100)}
out["med_eu"]["largest_observed_ha"] = float(med.AREA_HA.max())

# exposure indicator: artificial-surface share of burnt area, by country group
out["artificial_share_of_burnt_area"] = {"all_fires": round(W.artificial_share(d), 4), "mediterranean_eu": round(W.artificial_share(med), 4)}
(RES / "wildfire_summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
