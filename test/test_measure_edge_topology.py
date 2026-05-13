from __future__ import annotations

import math

import pytest

from plurel.topology_measure import SUMMARY_PERCENTILES, summarize_metrics
from scripts.measure_edge_topology import DatabaseTopologyStats, EdgeTopologyRecord

_summarize_metrics = summarize_metrics


def _make_edge(metrics: dict[str, float | bool | None]) -> EdgeTopologyRecord:
    return EdgeTopologyRecord(
        db_name="db",
        child_table="child",
        fkey_col="fk",
        parent_table="parent",
        num_children=0,
        num_parents=0,
        num_non_null_edges=0,
        metrics=metrics,
    )


def _make_db(edges: list[EdgeTopologyRecord]) -> DatabaseTopologyStats:
    return DatabaseTopologyStats(
        db_name="db",
        num_tables=0,
        num_edges=len(edges),
        total_child_rows=0,
        total_non_null_edges=0,
        edges=edges,
    )


def test_summarize_metrics_filters_nan_and_inf_before_aggregating():
    edges = [
        _make_edge({"fanout_gini": 0.10, "powerlaw_gamma": math.nan}),
        _make_edge({"fanout_gini": 0.50, "powerlaw_gamma": math.inf}),
        _make_edge({"fanout_gini": 0.90, "powerlaw_gamma": 2.5}),
    ]
    summary = _summarize_metrics([_make_db(edges)])

    gamma = summary["powerlaw_gamma"]
    assert gamma["count"] == 1
    assert gamma["n_missing"] == 2
    assert gamma["mean"] == pytest.approx(2.5)
    assert gamma["min"] == pytest.approx(2.5)
    assert gamma["max"] == pytest.approx(2.5)

    gini = summary["fanout_gini"]
    assert gini["count"] == 3
    assert gini["n_missing"] == 0
    assert gini["mean"] == pytest.approx(0.5)
    assert gini["min"] == pytest.approx(0.10)
    assert gini["max"] == pytest.approx(0.90)


def test_summarize_metrics_emits_percentiles_for_all_present_metrics():
    edges = [_make_edge({"fanout_gini": v / 10.0}) for v in range(11)]
    summary = _summarize_metrics([_make_db(edges)])

    record = summary["fanout_gini"]
    for percentile in SUMMARY_PERCENTILES:
        assert f"p{percentile:02d}" in record
    assert record["p50"] == pytest.approx(0.5)
    assert record["p10"] == pytest.approx(0.1)
    assert record["p90"] == pytest.approx(0.9)
    assert record["std"] == pytest.approx(record["std"])
    assert record["std"] > 0.0


def test_summarize_metrics_handles_metric_with_all_values_missing():
    edges = [_make_edge({"fanout_gini": 0.5, "powerlaw_gamma": math.nan})]
    summary = _summarize_metrics([_make_db(edges)])

    gamma = summary["powerlaw_gamma"]
    assert gamma["count"] == 0
    assert gamma["n_missing"] == 1
    assert "p50" not in gamma


def test_summarize_metrics_plausible_only_restricts_population():
    edges = [
        _make_edge({"powerlaw_gamma": 2.0, "powerlaw_plausible": True, "fanout_gini": 0.8}),
        _make_edge({"powerlaw_gamma": 9.0, "powerlaw_plausible": False, "fanout_gini": 0.2}),
        _make_edge({"powerlaw_gamma": 3.0, "powerlaw_plausible": True, "fanout_gini": 0.6}),
    ]
    full = _summarize_metrics([_make_db(edges)])
    plausible = _summarize_metrics([_make_db(edges)], plausible_only=True)

    assert full["powerlaw_gamma"]["count"] == 3
    assert plausible["powerlaw_gamma"]["count"] == 2
    assert plausible["powerlaw_gamma"]["mean"] == pytest.approx(2.5)
    assert plausible["fanout_gini"]["mean"] == pytest.approx(0.7)


def test_summarize_metrics_coerces_bools_for_aggregation():
    edges = [
        _make_edge({"powerlaw_plausible": True}),
        _make_edge({"powerlaw_plausible": False}),
        _make_edge({"powerlaw_plausible": True}),
    ]
    summary = _summarize_metrics([_make_db(edges)])

    record = summary["powerlaw_plausible"]
    assert record["count"] == 3
    assert record["mean"] == pytest.approx(2 / 3)
    assert record["min"] == 0.0
    assert record["max"] == 1.0
