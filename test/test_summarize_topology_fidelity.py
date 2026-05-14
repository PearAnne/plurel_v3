from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.summarize_topology_fidelity import (
    PRIMARY_METRICS,
    build_cohort_rank,
    build_per_metric_rank,
    population_name_from_path,
    write_outputs,
)


def _write_report(path: Path) -> None:
    rows = []
    for cohort, base in [("topology_stats.G0_hsbm", 0.3), ("topology_stats.G7_realistic_mix", 0.1)]:
        for metric_name in PRIMARY_METRICS:
            rows.append(
                {
                    "cohort": cohort,
                    "metric_name": metric_name,
                    "num_real_edges": 10,
                    "num_synth_edges": 10,
                    "real_mean": 1.0,
                    "synth_mean": 1.0 + base,
                    "mean_abs_error": base,
                    "ks_distance": base,
                    "wasserstein": base,
                }
            )
        rows.append(
            {
                "cohort": cohort,
                "metric_name": "null_rate",
                "num_real_edges": 10,
                "num_synth_edges": 10,
                "real_mean": 0.05,
                "synth_mean": 0.0,
                "mean_abs_error": 0.05,
                "ks_distance": 0.2,
                "wasserstein": 0.05,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_population_name_from_path_strips_fidelity_prefix() -> None:
    assert population_name_from_path(Path("fidelity_relbench_only.parquet")) == "relbench_only"


def test_build_cohort_rank_prefers_lower_metric_distances(tmp_path: Path) -> None:
    report_path = tmp_path / "fidelity_relbench_only.parquet"
    _write_report(report_path)

    per_metric_rank = build_per_metric_rank(reports=[report_path])
    cohort_rank = build_cohort_rank(per_metric_rank)

    relbench = cohort_rank[cohort_rank["population"] == "relbench_only"]
    assert relbench.iloc[0]["cohort"] == "G7_realistic_mix"
    assert set(per_metric_rank["metric_name"]) == set(PRIMARY_METRICS)


def test_write_outputs_creates_markdown_and_csvs(tmp_path: Path) -> None:
    report_path = tmp_path / "fidelity_relbench_only.parquet"
    output_dir = tmp_path / "report"
    _write_report(report_path)

    write_outputs(reports=[report_path], output_dir=output_dir)

    assert (
        (output_dir / "summary.md")
        .read_text(encoding="utf-8")
        .startswith("# Edge Prior Topology Fidelity Summary")
    )
    assert (output_dir / "per_metric_rank.csv").exists()
    assert (output_dir / "cohort_rank.csv").exists()
    assert (output_dir / "secondary_metrics.csv").exists()
