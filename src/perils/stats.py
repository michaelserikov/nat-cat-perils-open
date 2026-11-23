"""Shared statistics for frequency-severity hazard studies: count dispersion, generalised Pareto tails, compound Poisson simulation."""
from __future__ import annotations

import numpy as np
from scipy import stats


def dispersion(counts) -> dict:
    """Variance to mean ratio and a chi-square dispersion test against a Poisson count (one-sided, large ratio rejects)."""
    c = np.asarray(counts, float)
    n, m, v = len(c), c.mean(), c.var(ddof=1)
    stat = (n - 1) * v / m if m > 0 else np.nan
    return {"years": int(n), "mean": float(m), "var_over_mean": float(v / m) if m > 0 else np.nan,
            "p_overdispersed": float(1 - stats.chi2.cdf(stat, n - 1)) if m > 0 else np.nan}


def fit_gpd(x, threshold: float, boot: int = 300) -> dict:
    """Maximum likelihood GPD fit of the excesses above `threshold` (location fixed at 0), with a bootstrap interval for the shape."""
    x = np.asarray(x, float)
    ex = x[x > threshold] - threshold
    if len(ex) < 15:
        return {"n_exc": int(len(ex)), "shape": np.nan, "scale": np.nan, "shape_lo": np.nan, "shape_hi": np.nan}
    xi, _, sc = stats.genpareto.fit(ex, floc=0)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(boot):
        s = rng.choice(ex, len(ex))
        try:
            bs.append(stats.genpareto.fit(s, floc=0)[0])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [5, 95]) if bs else (np.nan, np.nan)
    return {"n_exc": int(len(ex)), "shape": float(xi), "scale": float(sc), "shape_lo": float(lo), "shape_hi": float(hi)}


def gpd_return_level(rate: float, threshold: float, shape: float, scale: float, years: float) -> float:
    """Level exceeded on average once in `years`, for `rate` threshold exceedances a year."""
    p = 1.0 / (rate * years)
    if p >= 1:
        return float(threshold)
    return float(threshold + stats.genpareto.ppf(1 - p, shape, loc=0, scale=scale))


def compound_poisson(rate: float, shape: float, scale: float, threshold: float, n_years: int = 20000, seed: int = 0, cap: float | None = None):
    """Annual sum and annual maximum of threshold exceedances: Poisson number of events, GPD excesses (optionally capped)."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate, n_years)
    tot, mx = np.zeros(n_years), np.zeros(n_years)
    for i, k in enumerate(n):
        if k:
            v = threshold + stats.genpareto.rvs(shape, scale=scale, size=k, random_state=rng)
            if cap is not None:
                v = np.minimum(v, cap)
            tot[i], mx[i] = v.sum(), v.max()
    return tot, mx


def aki_b(mags, mc: float, dm: float = 0.1) -> dict:
    """Maximum likelihood Gutenberg-Richter b value (Aki 1965 with the binning correction of Utsu) and its standard error."""
    m = np.asarray(mags, float)
    m = m[m >= mc]
    n = len(m)
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    return {"n": int(n), "b": float(b), "se": float(b / np.sqrt(n))}
