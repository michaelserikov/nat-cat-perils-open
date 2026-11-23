import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perils import earthquakes as eq  # noqa: E402
from perils import stats as S  # noqa: E402


def test_aki_recovers_b_value():
    rng = np.random.default_rng(1)
    b = 1.0
    m = 4.95 + rng.exponential(1 / (b * np.log(10)), 20000)
    m = m[m >= 4.95]
    m = np.round(m, 1)
    r = S.aki_b(m, 5.0)
    assert abs(r["b"] - b) < 0.05


def test_gpd_fit_and_return_level():
    rng = np.random.default_rng(2)
    x = 10 + S.stats.genpareto.rvs(0.3, scale=5, size=3000, random_state=rng)
    f = S.fit_gpd(x, 10)
    assert abs(f["shape"] - 0.3) < 0.08
    lvl = S.gpd_return_level(rate=len(x) / 100, threshold=10, shape=f["shape"], scale=f["scale"], years=100)
    assert np.isclose(lvl, np.quantile(x, 1 - 1 / len(x)), rtol=0.5)


def test_dispersion_flags_overdispersed_counts():
    rng = np.random.default_rng(3)
    pois = rng.poisson(5, 60)
    nb = rng.negative_binomial(2, 2 / 7, 60)
    assert S.dispersion(pois)["p_overdispersed"] > 0.05
    assert S.dispersion(nb)["p_overdispersed"] < 0.05


def test_decluster_removes_aftershocks_and_keeps_mainshock():
    t0 = pd.Timestamp("2020-01-01")
    rows = [{"time": t0, "latitude": 40.0, "longitude": 30.0, "mag": 7.0},
            {"time": t0 + pd.Timedelta(days=1), "latitude": 40.1, "longitude": 30.1, "mag": 5.5},
            {"time": t0 + pd.Timedelta(days=2), "latitude": 40.0, "longitude": 30.0, "mag": 5.2},
            {"time": t0 + pd.Timedelta(days=1500), "latitude": 40.0, "longitude": 30.0, "mag": 5.4},
            {"time": t0 + pd.Timedelta(days=1), "latitude": 10.0, "longitude": -60.0, "mag": 5.1}]
    d = eq.decluster(pd.DataFrame(rows).sort_values("time").reset_index(drop=True))
    assert len(d) == 3 and d.mag.max() == 7.0 and 5.5 not in set(d.mag)


def test_compound_poisson_mean_matches_theory():
    tot, mx = S.compound_poisson(rate=2.0, shape=0.1, scale=1.0, threshold=5.0, n_years=20000, seed=4)
    mean_event = 5.0 + 1.0 / (1 - 0.1)
    assert abs(tot.mean() - 2.0 * mean_event) < 0.4
    assert (mx >= tot * 0).all()
