"""Earthquake hazard for five EMEA regions from the USGS catalogue 1980-2025 (M >= 5)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perils import earthquakes as eq  # noqa: E402

DATA = Path.home() / "z_data" / "z_sre" / "usgs"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
df = eq.load(sorted(DATA.glob("eq_*.csv")))
df = df[(df.time >= "1980-01-01") & (df.time < "2026-01-01")]
YEARS = 46.0
out = {"catalogue": {"events_M5_plus_world": int(len(df)), "period": "1980-2025"}}
rows = []
for name, box in eq.REGIONS.items():
    for kind in ("gk", "uhr"):
        s = eq.region_stats(df, box, YEARS, kind=kind)
        s.update(region=name, window=kind)
        rows.append(s)
tab = pd.DataFrame(rows)
tab.to_csv(RES / "earthquake_regions.csv", index=False)
out["regions"] = rows
# stability of the largest events: M >= 7 observed in the box (all events, not declustered)
big = {}
for name, box in eq.REGIONS.items():
    b = eq.in_box(df, box)
    big[name] = {"M7_plus": int((b.mag >= 7).sum()), "largest": float(b.mag.max()), "largest_time": str(b.loc[b.mag.idxmax(), "time"].date())}
out["observed_largest"] = big
(RES / "earthquake_summary.json").write_text(json.dumps(out, indent=1))
print(tab[["region", "window", "events_all", "mainshocks", "mc", "b", "b_se", "rate_per_yr_mc", "obs_M7", "return_period_M7_yr", "return_period_M6.5_yr"]].to_string())
print(json.dumps(big, indent=1))
