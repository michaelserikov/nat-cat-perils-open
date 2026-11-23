"""Earthquake hazard from the USGS catalogue: declustering, completeness, Gutenberg-Richter and return periods for regional boxes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import aki_b

# Declustering windows. "gk": Gardner and Knopoff (1974) as usually coded (distance 10^(0.1238 M + 0.983) km; time 10^(0.032 M + 2.7389) days
# for M >= 6.5, else 10^(0.5409 M - 0.547) days). "uhr": distance exp(0.804 M - 1.024) km after Uhrhammer (1986) with the same time window.
# The coefficients are written from the literature as I know it, not read from the original papers; the choice of window is run as a sensitivity.
REGIONS = {  # lon_min, lon_max, lat_min, lat_max (boxes, not political borders)
    "Turkey": (26.0, 45.0, 36.0, 42.0),
    "Italy": (6.5, 18.6, 36.0, 47.5),
    "Greece and Aegean": (19.0, 28.5, 34.0, 41.5),
    "Iran": (44.0, 63.5, 25.0, 40.0),
    "Albania, Montenegro, Serbia": (18.6, 23.0, 39.0, 46.0),
}


def load(files) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, usecols=["time", "latitude", "longitude", "depth", "mag", "type"]) for f in files], ignore_index=True)
    df = df[df["type"] == "earthquake"].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    return df.sort_values("time").reset_index(drop=True)


def _km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def windows(mag, kind="gk"):
    mag = np.asarray(mag, float)
    tw = np.where(mag >= 6.5, 10 ** (0.032 * mag + 2.7389), 10 ** (0.5409 * mag - 0.547))
    dw = 10 ** (0.1238 * mag + 0.983) if kind == "gk" else np.exp(0.804 * mag - 1.024)
    return tw, dw


def decluster(df: pd.DataFrame, kind="gk") -> pd.DataFrame:
    """Gardner-Knopoff declustering: keep the largest event of each window cluster (mainshocks); the input must be sorted by time."""
    t = (df["time"].values.astype("datetime64[s]").astype("int64") / 86400.0)
    lat, lon, mag = df["latitude"].values, df["longitude"].values, df["mag"].values
    order = np.argsort(-mag)
    removed = np.zeros(len(df), bool)
    tw, dw = windows(mag, kind)
    for i in order:
        if removed[i]:
            continue
        near = np.where((np.abs(t - t[i]) <= tw[i]) & (mag < mag[i]) & ~removed)[0]
        if len(near):
            d = _km(lat[i], lon[i], lat[near], lon[near])
            removed[near[d <= dw[i]]] = True
    return df[~removed].reset_index(drop=True)


def completeness_maxc(mags, dm: float = 0.1) -> float:
    """Magnitude of completeness by maximum curvature plus 0.2 (a common rule of thumb; Woessner and Wiemer 2005 give the bias)."""
    m = np.round(np.asarray(mags, float) / dm) * dm
    vals, cnt = np.unique(np.round(m, 2), return_counts=True)
    return float(vals[np.argmax(cnt)] + 0.2)


def in_box(df: pd.DataFrame, box) -> pd.DataFrame:
    lo0, lo1, la0, la1 = box
    return df[(df.longitude >= lo0) & (df.longitude <= lo1) & (df.latitude >= la0) & (df.latitude <= la1)]


def region_stats(df: pd.DataFrame, box, years: float, m_ref=(6.0, 6.5, 7.0), kind="gk") -> dict:
    """Declustered mainshock rate above the completeness magnitude, b value and return periods of the reference magnitudes."""
    sub = decluster(in_box(df, box).reset_index(drop=True), kind)
    mc = max(completeness_maxc(sub["mag"]), 5.0)
    b = aki_b(sub["mag"], mc)
    rate_mc = (sub["mag"] >= mc).sum() / years
    out = {"events_all": int(len(in_box(df, box))), "mainshocks": int(len(sub)), "mc": round(mc, 1), "b": round(b["b"], 2), "b_se": round(b["se"], 2),
           "rate_per_yr_mc": round(float(rate_mc), 2)}
    for m in m_ref:
        obs = int((sub["mag"] >= m).sum())
        gr = rate_mc * 10 ** (-b["b"] * (m - mc))
        out[f"obs_M{m:g}"] = obs
        out[f"gr_rate_M{m:g}"] = round(float(gr), 3)
        out[f"return_period_M{m:g}_yr"] = round(float(1.0 / gr), 1)
    return out
