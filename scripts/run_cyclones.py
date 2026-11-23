"""Tropical cyclones affecting EMEA from IBTrACS v04r01 since 1980: landfalls in two boxes and North Atlantic storms reaching Europe."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perils import cyclones as C  # noqa: E402
from perils import stats as S  # noqa: E402

SRC = Path.home() / "z_data" / "z_sre" / "ibtracs_full.csv"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
df = C.load(SRC)
df = df[df.SEASON <= 2025]
Y0, Y1 = 1980, 2025
years = Y1 - Y0 + 1
out = {"steps": int(len(df)), "storms": int(df.SID.nunique()), "period": [Y0, Y1]}
rng = np.random.default_rng(0)
rows = []
for name, (basin, lo0, lo1, la0, la1) in C.BOXES.items():
    lf = C.landfalls(df, basin, lo0, lo1, la0, la1)
    lf = lf[(lf.SEASON >= Y0) & (lf.SEASON <= Y1)]
    cnt = lf.groupby("SEASON").size().reindex(range(Y0, Y1 + 1), fill_value=0).values
    disp = S.dispersion(cnt)
    w = lf.wind_lf_kt.dropna().values
    r = {"region": name, "landfalling_storms": int(len(lf)), "per_year": round(len(lf) / years, 2), "var_over_mean": round(disp["var_over_mean"], 2),
         "p_overdispersed": round(disp["p_overdispersed"], 2), "share_ge_64kt": round(float((w >= 64).mean()), 2), "n_ge_64kt": int((w >= 64).sum()),
         "max_wind_kt": float(w.max()), "strongest": str(lf.loc[lf.wind_lf_kt.idxmax(), "NAME"]) + " " + str(lf.loc[lf.wind_lf_kt.idxmax(), "SEASON"])}
    # trend in the annual count (Poisson regression slope per decade), first half against second half
    half = years // 2
    r["per_year_first_half"], r["per_year_second_half"] = round(float(cnt[:half].mean()), 2), round(float(cnt[half:].mean()), 2)
    # landfall wind return levels: GPD above 50 kt where enough exceedances, bootstrap over storms
    thr = 50.0
    g = S.fit_gpd(w, thr)
    r["gpd_threshold_kt"], r["gpd_n_exc"] = thr, g["n_exc"]
    if g["n_exc"] >= 15:
        rate = (w > thr).sum() / years
        lv = {}
        for T in (10, 25, 50):
            lv[T] = S.gpd_return_level(rate, thr, g["shape"], g["scale"], T)
        bs = []
        for _ in range(300):
            s = rng.choice(w, len(w))
            gb = S.fit_gpd(s, thr, boot=0) if (s > thr).sum() >= 15 else None
            if gb and np.isfinite(gb["shape"]):
                bs.append(S.gpd_return_level((s > thr).sum() / years, thr, gb["shape"], gb["scale"], 25))
        r["gpd_shape"] = round(g["shape"], 2)
        r["return_level_kt"] = {str(k): round(v, 0) for k, v in lv.items()}
        if bs:
            r["return_level_25yr_kt_90pct_range"] = [round(float(np.percentile(bs, 5)), 0), round(float(np.percentile(bs, 95)), 0)]
    rows.append(r)
    lf.to_csv(RES / ("landfalls_" + basin + ".csv"), index=False)
out["landfalls"] = rows

eu = C.europe_reach(df[(df.SEASON >= Y0) & (df.SEASON <= Y1)])
eu.to_csv(RES / "north_atlantic_storms_in_europe_box.csv", index=False)
out["europe"] = {"storms_in_box": int(len(eu)), "per_year": round(len(eu) / years, 2),
                 "with_wind_ge_64kt_in_box": int((eu.max_wind_kt >= 64).sum()), "with_wind_ge_50kt_in_box": int((eu.max_wind_kt >= 50).sum()),
                 "top5_by_wind": eu.sort_values("max_wind_kt", ascending=False).head(5)[["name", "season", "max_wind_kt", "nature"]].astype(str).values.tolist(),
                 "first_half_per_year": round(float((eu.season < Y0 + years // 2).sum() / (years // 2)), 2),
                 "second_half_per_year": round(float((eu.season >= Y0 + years // 2).sum() / (years - years // 2)), 2)}
(RES / "cyclone_summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
