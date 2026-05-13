from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plurel.topology_metrics import (
    _classify_cardinality,
    _fit_beta_mom,
    _pa_exponent_alpha,
    _timestamps_to_float,
    compute_edge_metrics,
)


def test_compute_edge_metrics_returns_complete_record():
    parent_idx = np.array([0, 1, 0, 2, 1, 0], dtype=np.int64)
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    metrics = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=4,
        timestamps=timestamps,
    )

    expected_keys = {
        "fanout_p05",
        "fanout_p25",
        "fanout_p50",
        "fanout_p75",
        "fanout_p95",
        "fanout_p99",
        "fanout_max",
        "fanout_gini",
        "fanout_ks_to_poisson",
        "fanout_ks_to_powerlaw",
        "powerlaw_gamma",
        "powerlaw_xmin",
        "powerlaw_pvalue",
        "powerlaw_fit_n",
        "powerlaw_plausible",
        "isolated_parent_rate",
        "null_rate",
        "degree_assortativity",
        "temporal_growth_alpha",
    }
    assert expected_keys.issubset(metrics)
    assert metrics["fanout_max"] == 3
    assert metrics["isolated_parent_rate"] == pytest.approx(0.25)
    assert metrics["null_rate"] == pytest.approx(0.0)
    assert 0.0 <= metrics["fanout_gini"] <= 1.0
    assert -1.0 <= metrics["degree_assortativity"] <= 1.0 or math.isnan(
        metrics["degree_assortativity"]
    )
    assert isinstance(metrics["powerlaw_plausible"], bool)
    assert metrics["cardinality_kind"] in {"no_data", "one_to_one", "many_to_one"}
    for key, value in metrics.items():
        if key in {"powerlaw_plausible", "cardinality_kind"}:
            continue
        assert value is None or math.isnan(value) or math.isfinite(value)


def test_null_mask_is_applied_before_fanout_counts():
    parent_idx = np.array([0, 99, 1, 1, -1, 2], dtype=np.int64)
    null_mask = np.array([False, True, False, False, True, False], dtype=bool)
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    metrics = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=3,
        null_mask=null_mask,
        timestamps=timestamps,
    )

    assert metrics["null_rate"] == pytest.approx(2 / 6)
    assert metrics["fanout_max"] == 2
    assert metrics["isolated_parent_rate"] == pytest.approx(0.0)
    assert metrics["fanout_p50"] == pytest.approx(1.0)
    assert np.isfinite(metrics["temporal_growth_alpha"])


def test_uniform_assignments_have_zero_concentration():
    parent_idx = np.tile(np.arange(5, dtype=np.int64), 20)

    metrics = compute_edge_metrics(parent_idx=parent_idx, num_parents=5)

    assert metrics["fanout_gini"] == pytest.approx(0.0)
    assert metrics["fanout_max"] == 20
    assert metrics["isolated_parent_rate"] == pytest.approx(0.0)
    assert metrics["fanout_ks_to_poisson"] >= 0.0


def test_concentrated_assignments_show_high_gini():
    parent_idx = np.zeros(16, dtype=np.int64)

    metrics = compute_edge_metrics(parent_idx=parent_idx, num_parents=4)

    assert metrics["fanout_max"] == 16
    assert metrics["isolated_parent_rate"] == pytest.approx(0.75)
    assert metrics["fanout_gini"] > 0.7
    assert math.isnan(metrics["powerlaw_gamma"])


def test_temporal_growth_alpha_is_positive_for_prefix_growth():
    parent_idx = np.array(
        [0] * 24 + [1] * 16 + [2] * 8 + [3] * 4 + [4] * 2 + [5] * 2,
        dtype=np.int64,
    )
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    metrics = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=6,
        timestamps=timestamps,
    )

    assert np.isfinite(metrics["temporal_growth_alpha"])
    assert metrics["temporal_growth_alpha"] > 0.0


def test_scale_free_like_tail_produces_reasonable_powerlaw_fit():
    num_parents = 128
    counts = np.maximum(
        1,
        (1000 / np.power(np.arange(1, num_parents + 1, dtype=float), 1.5)).astype(int),
    )
    parent_idx = np.repeat(np.arange(num_parents, dtype=np.int64), counts)
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    metrics = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=num_parents,
        timestamps=timestamps,
    )

    assert np.isfinite(metrics["powerlaw_gamma"])
    assert metrics["powerlaw_gamma"] > 1.0
    assert metrics["powerlaw_xmin"] >= 1.0
    assert metrics["fanout_ks_to_powerlaw"] >= 0.0
    assert metrics["fanout_gini"] > 0.2
    assert metrics["powerlaw_fit_n"] >= 1
    assert isinstance(metrics["powerlaw_plausible"], bool)


def test_powerlaw_fit_is_skipped_for_small_samples():
    parent_idx = np.arange(50, dtype=np.int64)

    metrics = compute_edge_metrics(parent_idx=parent_idx, num_parents=50)

    assert math.isnan(metrics["powerlaw_gamma"])
    assert math.isnan(metrics["powerlaw_xmin"])
    assert math.isnan(metrics["fanout_ks_to_powerlaw"])
    assert math.isnan(metrics["powerlaw_fit_n"])
    assert metrics["powerlaw_plausible"] is False


def test_powerlaw_subsample_preserves_gamma_within_tolerance():
    rng = np.random.default_rng(0)
    gamma_true = 2.5
    num_parents = 8000
    fanouts = np.maximum(1, (rng.pareto(gamma_true - 1.0, size=num_parents) + 1.0).astype(int))
    parent_idx = np.repeat(np.arange(num_parents, dtype=np.int64), fanouts)
    rng.shuffle(parent_idx)

    full = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=num_parents,
        max_powerlaw_sample=0,
    )
    subsampled = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=num_parents,
        max_powerlaw_sample=1000,
    )

    assert full["powerlaw_fit_n"] == num_parents
    assert subsampled["powerlaw_fit_n"] == 1000
    full_gamma = full["powerlaw_gamma"]
    sub_gamma = subsampled["powerlaw_gamma"]
    assert np.isfinite(full_gamma) and np.isfinite(sub_gamma)
    asymptotic_std = max(full_gamma - 1.0, 0.1) / math.sqrt(subsampled["powerlaw_fit_n"])
    assert abs(full_gamma - sub_gamma) < 8.0 * asymptotic_std


def test_powerlaw_plausible_filters_pathological_fits():
    parent_idx = np.tile(np.arange(200, dtype=np.int64), 4)

    metrics = compute_edge_metrics(parent_idx=parent_idx, num_parents=200)

    if np.isfinite(metrics["powerlaw_gamma"]):
        if metrics["powerlaw_gamma"] > 8.0 or metrics["fanout_ks_to_powerlaw"] > 0.3:
            assert metrics["powerlaw_plausible"] is False


def test_compute_edge_metrics_emits_new_dcsbm_pa_cardinality_fields():
    rng = np.random.default_rng(0)
    fanouts = np.maximum(1, (rng.pareto(2.0, size=200) + 1).astype(int))
    parent_idx = np.repeat(np.arange(200, dtype=np.int64), fanouts)
    rng.shuffle(parent_idx)
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    metrics = compute_edge_metrics(parent_idx=parent_idx, num_parents=200, timestamps=timestamps)

    assert "theta_beta_alpha" in metrics
    assert "theta_beta_beta" in metrics
    assert "pa_exponent_alpha" in metrics
    assert "cardinality_kind" in metrics
    assert metrics["cardinality_kind"] == "many_to_one"


def test_fit_beta_mom_recovers_known_shape():
    rng = np.random.default_rng(0)
    alpha_true, beta_true = 2.0, 5.0
    samples = rng.beta(alpha_true, beta_true, size=10_000)
    fanout = (samples * 1000).astype(np.int64)

    alpha_est, beta_est = _fit_beta_mom(fanout)

    assert np.isfinite(alpha_est) and np.isfinite(beta_est)
    assert abs(alpha_est - alpha_true) / alpha_true < 0.3
    assert abs(beta_est - beta_true) / beta_true < 0.3


def test_fit_beta_mom_returns_nan_on_degenerate_input():
    assert all(math.isnan(x) for x in _fit_beta_mom(np.zeros(10, dtype=np.int64)))
    assert all(math.isnan(x) for x in _fit_beta_mom(np.full(10, 5, dtype=np.int64)))
    assert all(math.isnan(x) for x in _fit_beta_mom(np.array([], dtype=np.int64)))


def test_pa_exponent_alpha_recovers_linear_pa():
    rng = np.random.default_rng(0)
    num_parents = 200
    parent_idx_list: list[int] = []
    degrees = np.ones(num_parents, dtype=np.float64)
    for _ in range(8000):
        probs = degrees / degrees.sum()
        chosen = int(rng.choice(num_parents, p=probs))
        parent_idx_list.append(chosen)
        degrees[chosen] += 1.0
    parent_idx = np.asarray(parent_idx_list, dtype=np.int64)
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    alpha_est = _pa_exponent_alpha(
        active_parent_idx=parent_idx,
        timestamps=timestamps,
        num_parents=num_parents,
    )

    assert np.isfinite(alpha_est)
    assert 0.6 < alpha_est < 1.4


def test_pa_exponent_alpha_recovers_uniform_attachment_near_zero():
    rng = np.random.default_rng(1)
    num_parents = 200
    parent_idx = rng.integers(0, num_parents, size=8000, dtype=np.int64)
    timestamps = np.arange(parent_idx.size, dtype=np.int64)

    alpha_est = _pa_exponent_alpha(
        active_parent_idx=parent_idx,
        timestamps=timestamps,
        num_parents=num_parents,
    )

    assert np.isfinite(alpha_est)
    assert abs(alpha_est) < 0.5


def test_pa_exponent_alpha_returns_nan_without_timestamps():
    parent_idx = np.zeros(1000, dtype=np.int64)
    assert math.isnan(
        _pa_exponent_alpha(active_parent_idx=parent_idx, timestamps=None, num_parents=100)
    )


def test_classify_cardinality_returns_three_categories():
    one_to_one = np.array([1, 1, 1, 0, 1], dtype=np.int64)
    many_to_one = np.array([5, 2, 1, 0, 0], dtype=np.int64)
    no_data = np.zeros(5, dtype=np.int64)
    empty = np.array([], dtype=np.int64)

    assert _classify_cardinality(one_to_one) == "one_to_one"
    assert _classify_cardinality(many_to_one) == "many_to_one"
    assert _classify_cardinality(no_data) == "no_data"
    assert _classify_cardinality(empty) == "no_data"


def test_empty_input_returns_defaults():
    metrics = compute_edge_metrics(parent_idx=np.array([], dtype=np.int64), num_parents=3)

    assert metrics["fanout_max"] == 0
    assert metrics["isolated_parent_rate"] == pytest.approx(1.0)
    assert metrics["fanout_gini"] == pytest.approx(0.0)
    assert metrics["null_rate"] == pytest.approx(0.0)
    assert math.isnan(metrics["powerlaw_gamma"])
    assert math.isnan(metrics["degree_assortativity"])
    assert math.isnan(metrics["temporal_growth_alpha"])


def test_timestamps_to_float_preserves_length_with_nat_and_string_dates():
    s = pd.Series(
        [pd.Timestamp("2020-01-01", tz="UTC"), pd.NA, "2020-01-03"],
        dtype="datetime64[ns, UTC]",
    )
    arr = s.to_numpy()
    out = _timestamps_to_float(arr)
    assert out.shape == (3,)
    assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])


def test_compute_edge_metrics_accepts_object_timestamps_with_pd_na():
    parent_idx = np.array([0, 1, 0, 1, 0] * 40, dtype=np.int64)
    ts = pd.Series(
        [pd.Timestamp("2020-01-01")] * 100 + [pd.NA] * 100,
        dtype="datetime64[ns]",
    )
    metrics = compute_edge_metrics(
        parent_idx=parent_idx,
        num_parents=2,
        timestamps=ts.to_numpy(),
    )
    assert metrics["fanout_max"] >= 1
    for key, value in metrics.items():
        if key in {"powerlaw_plausible", "cardinality_kind"}:
            continue
        assert value is None or math.isnan(value) or math.isfinite(value)
