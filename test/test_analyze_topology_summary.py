from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_topology_summary import (
    Quantiles,
    _suggest_prior_mixture,
    categorize_edges,
    derive_prior_recommendations,
    finite_quantiles,
    load_summary_as_frame,
    null_rate_breakdown,
    render_markdown,
)


def test_finite_quantiles_drops_nan_and_inf():
    series = pd.Series([1.0, 2.0, np.nan, np.inf, 3.0, -np.inf])

    result = finite_quantiles(series)

    assert isinstance(result, Quantiles)
    assert result.count == 3
    assert result.n_missing == 3
    assert result.p50 == pytest.approx(2.0)
    assert result.min == pytest.approx(1.0)
    assert result.max == pytest.approx(3.0)


def test_finite_quantiles_handles_empty_input():
    series = pd.Series([np.nan, np.inf, -np.inf])

    result = finite_quantiles(series)

    assert result.count == 0
    assert result.n_missing == 3
    assert result.p50 is None
    assert result.mean is None


def test_categorize_edges_splits_into_five_buckets():
    frame = pd.DataFrame(
        [
            {
                "m_powerlaw_gamma": 2.5,
                "m_fanout_ks_to_powerlaw": 0.05,
                "m_powerlaw_plausible": True,
            },
            {
                "m_powerlaw_gamma": 2.8,
                "m_fanout_ks_to_powerlaw": 0.10,
                "m_powerlaw_plausible": True,
            },
            {
                "m_powerlaw_gamma": 15.0,
                "m_fanout_ks_to_powerlaw": 0.50,
                "m_powerlaw_plausible": False,
            },
            {
                "m_powerlaw_gamma": 1.2,
                "m_fanout_ks_to_powerlaw": 0.20,
                "m_powerlaw_plausible": False,
            },
            {
                "m_powerlaw_gamma": 5.0,
                "m_fanout_ks_to_powerlaw": 0.50,
                "m_powerlaw_plausible": False,
            },
            {
                "m_powerlaw_gamma": math.nan,
                "m_fanout_ks_to_powerlaw": math.nan,
                "m_powerlaw_plausible": False,
            },
        ]
    )

    categories = categorize_edges(frame)

    assert categories["total"] == 6
    assert categories["plausible_powerlaw"] == 2
    assert categories["near_uniform_high_gamma"] == 1
    assert categories["extreme_low_gamma"] == 1
    assert categories["heavy_non_powerlaw"] == 1
    assert categories["no_fit_or_degenerate"] == 1


def test_suggest_prior_mixture_respects_powerlaw_majority():
    mixture = _suggest_prior_mixture(
        {"plausible_powerlaw": 70, "near_uniform_high_gamma": 15, "heavy_non_powerlaw": 15}
    )

    assert mixture[0] == "tpa"
    assert mixture.count("chung_lu") + mixture.count("tpa") >= 5
    assert "dcsbm" in mixture
    assert "erdos_renyi" in mixture or "hsbm" in mixture


def test_null_rate_breakdown_buckets_distribution():
    frame = pd.DataFrame(
        {
            "m_null_rate": [
                0.0,
                0.0,
                0.0,
                0.01,
                0.04,
                0.10,
                0.20,
                0.40,
                0.99,
                math.nan,
            ]
        }
    )

    breakdown = null_rate_breakdown(frame)

    assert breakdown["zero"] == 4
    assert breakdown["small_(0_0.05]"] == 2
    assert breakdown["medium_(0.05_0.3]"] == 2
    assert breakdown["high_(0.3_0.9]"] == 1
    assert breakdown["extreme_(0.9_1.0]"] == 1
    assert breakdown["nonzero_p50"] is not None


def test_load_summary_as_frame_round_trip(tmp_path: Path):
    payload = {
        "rows": [
            {
                "db_name": "rel-test",
                "child_table": "child",
                "fkey_col": "fk",
                "parent_table": "parent",
                "num_children": 100,
                "num_parents": 50,
                "num_non_null_edges": 100,
                "metrics": {
                    "fanout_gini": 0.6,
                    "powerlaw_gamma": 2.4,
                    "powerlaw_plausible": True,
                    "temporal_growth_alpha": 1.3,
                    "null_rate": 0.0,
                },
            }
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    frame = load_summary_as_frame(summary_path)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["db_name"] == "rel-test"
    assert row["benchmark_family"] == "relbench"
    assert row["m_powerlaw_gamma"] == pytest.approx(2.4)
    assert bool(row["m_powerlaw_plausible"]) is True


def test_derive_prior_recommendations_produces_six_fields():
    frame = pd.DataFrame(
        [
            {
                "db_name": "rel-a",
                "m_powerlaw_gamma": gamma,
                "m_fanout_ks_to_powerlaw": 0.05,
                "m_powerlaw_plausible": True,
                "m_temporal_growth_alpha": ta,
                "m_pa_exponent_alpha": pa,
                "m_theta_beta_alpha": tb_a,
                "m_theta_beta_beta": tb_b,
                "m_null_rate": null_rate,
            }
            for gamma, ta, pa, tb_a, tb_b, null_rate in zip(
                [2.0, 2.5, 3.0, 3.5, 4.0],
                [1.0, 1.2, 1.4, 1.6, 1.8],
                [0.8, 0.9, 1.0, 1.1, 1.2],
                [0.5, 0.6, 0.7, 0.8, 0.9],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [0.0, 0.0, 0.0, 0.05, 0.3],
            )
        ]
    )

    recs = derive_prior_recommendations(frame)

    fields = {rec.config_field for rec in recs}
    assert {
        "chung_lu_gamma_choices",
        "tpa_alpha_choices",
        "dcsbm_theta_alpha_choices",
        "dcsbm_theta_beta_choices",
        "edge_prior_null_rate_choices",
        "topology_prior_choices",
    } == fields
    for rec in recs:
        if rec.config_field == "chung_lu_gamma_choices":
            assert rec.suggested_kind == "range"
            assert rec.suggested_value is not None
            low, high = rec.suggested_value
            assert low <= 3.0 <= high
        if rec.config_field == "tpa_alpha_choices":
            assert rec.source_metric == "pa_exponent_alpha"
        if rec.config_field == "dcsbm_theta_alpha_choices":
            assert rec.source_metric == "theta_beta_alpha"


def test_render_markdown_is_self_contained(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "db_name": "rel-a",
                "child_table": "c",
                "fkey_col": "fk",
                "parent_table": "p",
                "num_children": 100,
                "num_parents": 20,
                "num_non_null_edges": 100,
                "m_fanout_gini": 0.6,
                "m_powerlaw_gamma": 2.5,
                "m_fanout_ks_to_powerlaw": 0.05,
                "m_powerlaw_plausible": True,
                "m_temporal_growth_alpha": 1.4,
                "m_pa_exponent_alpha": 1.0,
                "m_theta_beta_alpha": 0.7,
                "m_theta_beta_beta": 4.0,
                "m_null_rate": 0.0,
                "benchmark_family": "relbench",
            }
        ]
    )
    recs = derive_prior_recommendations(frame)
    rendered = render_markdown(
        frame=frame,
        recommendations=recs,
        summary_path=tmp_path / "summary.json",
        timestamp=datetime(2026, 5, 11, 12, 0, tzinfo=UTC),
    )

    assert "Phase 0c" in rendered
    assert "chung_lu_gamma_choices" in rendered
    assert "tpa_alpha_choices" in rendered
    assert "dcsbm_theta_alpha_choices" in rendered
    assert "pa_exponent_alpha" in rendered
    assert "Single benchmark family" in rendered
    assert "Per-recommendation rationale" in rendered
