import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perils import cyclones as C  # noqa: E402
from perils import wildfires as W  # noqa: E402


def _track(sid, dists, winds, lat=-18.0, lon=45.0, basin="SI", season=2000):
    n = len(dists)
    return pd.DataFrame({"SID": sid, "SEASON": season, "BASIN": basin, "NAME": "X", "NATURE": "TS",
                         "ISO_TIME": pd.date_range("2000-01-01", periods=n, freq="6h"), "LAT": lat, "LON": lon,
                         "WMO_WIND": winds, "USA_WIND": np.nan, "DIST2LAND": dists, "wind_kt": winds})


def test_landfall_is_first_step_over_land_after_sea():
    sea_land_sea_land = _track("A", [200, 100, 0, 0, 50, 0], [60, 80, 90, 70, 50, 40])
    df = pd.concat([sea_land_sea_land, _track("B", [300, 250, 200], [50, 50, 50])], ignore_index=True)
    lf = C.landfalls(df, "SI", 30, 52, -28, -10)
    # one landfall per storm, the strongest of its landfalls: the first one has wind max(90, 80) = 90
    assert len(lf) == 1 and lf.SID.iloc[0] == "A" and lf.wind_lf_kt.iloc[0] == 90


def test_landfall_outside_box_is_ignored():
    df = _track("C", [100, 0], [70, 80], lon=70.0)
    assert len(C.landfalls(df, "SI", 30, 52, -28, -10)) == 0


def test_europe_reach_takes_peak_wind_inside_box():
    a = _track("N1", [900, 900, 900], [40, 80, 60], lat=50.0, lon=-10.0, basin="NA")
    b = _track("N2", [900, 900], [100, 100], lat=20.0, lon=-60.0, basin="NA")
    eu = C.europe_reach(pd.concat([a, b], ignore_index=True))
    assert list(eu.SID) == ["N1"] and eu.max_wind_kt.iloc[0] == 80


def test_artificial_share_is_area_weighted():
    df = pd.DataFrame({"AREA_HA": [100.0, 300.0], **{c: [0.0, 0.0] for c in W.LC}})
    df["ARTIFSURF"] = [10.0, 0.0]
    df["OTHERLC"] = [90.0, 100.0]
    assert np.isclose(W.artificial_share(df), 0.1 * 100 / 400)
