"""Wildfire hazard in Europe and the Mediterranean from EFFIS burnt-area polygons (MODIS, 2016-2025): size, frequency and exposure share."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LC = ["BROADLEA", "CONIFER", "MIXED", "SCLEROPH", "TRANSIT", "OTHERNATLC", "AGRIAREAS", "ARTIFSURF", "OTHERLC"]
EU_MED = ["PT", "ES", "FR", "IT", "EL", "GR", "HR", "BG", "RO", "CY", "AL", "ME", "RS", "BA", "MK", "SI", "TR", "MA", "DZ", "TN", "IL", "LB", "SY"]


def load(folder) -> pd.DataFrame:
    fs = sorted(Path(folder).glob("ba_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["date"] = pd.to_datetime(df["FIREDATE"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["AREA_HA"] = pd.to_numeric(df["AREA_HA"], errors="coerce")
    for c in LC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["AREA_HA", "year"]).drop_duplicates("id")


def yearly(df: pd.DataFrame, min_ha: float = 30.0) -> pd.DataFrame:
    d = df[df.AREA_HA >= min_ha]
    return d.groupby("year").agg(fires=("id", "size"), burnt_ha=("AREA_HA", "sum"), max_ha=("AREA_HA", "max"))


def artificial_share(df: pd.DataFrame) -> float:
    """Area-weighted share of burnt area classed as artificial surfaces by EFFIS (the land cover split sums to about 100 per fire)."""
    w = df["AREA_HA"]
    tot = df[LC].sum(axis=1).replace(0, np.nan)
    sh = df["ARTIFSURF"] / tot
    ok = sh.notna()
    return float((sh[ok] * w[ok]).sum() / w[ok].sum())
