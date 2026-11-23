"""Tropical cyclone hazard for EMEA coasts from IBTrACS v04r01 (since 1980): landfalls by box and tracks reaching Europe."""
from __future__ import annotations

import numpy as np
import pandas as pd

KT_TO_MS = 0.514444
BOXES = {  # lon_min, lon_max, lat_min, lat_max
    "South-West Indian Ocean (Mozambique, Madagascar)": ("SI", 30.0, 52.0, -28.0, -10.0),
    "Arabian Sea (Oman, Yemen, Somalia)": ("NI", 40.0, 62.0, 8.0, 28.0),
}
EUROPE_BOX = (-30.0, 15.0, 35.0, 62.0)   # North Atlantic to western Europe


def load(path) -> pd.DataFrame:
    cols = ["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND", "USA_WIND", "DIST2LAND"]
    # keep_default_na is off because the basin code 'NA' (North Atlantic) would otherwise be read as missing
    df = pd.read_csv(path, usecols=cols, skiprows=[1], low_memory=False, keep_default_na=False, na_values=["", " "])
    for c in ["LAT", "LON", "WMO_WIND", "USA_WIND", "DIST2LAND", "SEASON"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["wind_kt"] = df["WMO_WIND"].fillna(df["USA_WIND"])
    return df.dropna(subset=["LAT", "LON", "ISO_TIME"]).sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def landfalls(df: pd.DataFrame, basin: str, lon0, lon1, lat0, lat1) -> pd.DataFrame:
    """First step over land (distance to land 0) after a step over sea inside the box; wind is the larger of the two steps around it.

    Wind is the agency (WMO) value, with the USA value where there is none; steps are three-hourly or six-hourly, so the value is
    the maximum sustained wind of the nearest recorded steps, not at the instant of landfall.
    """
    d = df[df.BASIN == basin].copy()
    d["prev_d"] = d.groupby("SID")["DIST2LAND"].shift(1)
    d["prev_w"] = d.groupby("SID")["wind_kt"].shift(1)
    lf = d[(d.DIST2LAND == 0) & (d.prev_d > 0)]
    lf = lf[(lf.LON >= lon0) & (lf.LON <= lon1) & (lf.LAT >= lat0) & (lf.LAT <= lat1)].copy()
    lf["wind_lf_kt"] = lf[["wind_kt", "prev_w"]].max(axis=1)
    # one landfall per storm: the strongest
    lf = lf.sort_values("wind_lf_kt", ascending=False).drop_duplicates("SID")
    return lf[["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "wind_lf_kt"]].sort_values("ISO_TIME").reset_index(drop=True)


def europe_reach(df: pd.DataFrame, box=EUROPE_BOX) -> pd.DataFrame:
    """Tropical-origin storms of the North Atlantic with a track step inside the Europe box: peak wind inside the box and nature there."""
    lo0, lo1, la0, la1 = box
    d = df[df.BASIN == "NA"]
    inside = d[(d.LON >= lo0) & (d.LON <= lo1) & (d.LAT >= la0) & (d.LAT <= la1)]
    g = inside.groupby("SID")
    out = g.agg(season=("SEASON", "first"), name=("NAME", "first"), first_time=("ISO_TIME", "min"), max_wind_kt=("wind_kt", "max"),
                min_lat=("LAT", "min"), max_lat=("LAT", "max"), nature=("NATURE", lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    return out.reset_index().sort_values("first_time").reset_index(drop=True)


def annual_counts(events: pd.DataFrame, season_col: str, y0: int, y1: int, min_wind: float | None = None) -> np.ndarray:
    e = events if min_wind is None else events[events.iloc[:, events.columns.get_loc([c for c in events.columns if "wind" in c][0])] >= min_wind]
    c = e.groupby(season_col).size().reindex(range(y0, y1 + 1), fill_value=0)
    return c.values
