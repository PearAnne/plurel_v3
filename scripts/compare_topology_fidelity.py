from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_METRICS = [
    "fanout_gini",
    "fanout_ks_to_poisson",
    "fanout_ks_to_powerlaw",
    "powerlaw_gamma",
    "isolated_parent_rate",
    "null_rate",
    "degree_assortativity",
    "temporal_growth_alpha",
]


def load_metric_frame(summary_path: Path, cohort: str) -> pd.DataFrame:
    payload = json.loads(summary_path.expanduser().read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("rows", []):
        metrics = row.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "db_name": row["db_name"],
                    "child_table": row["child_table"],
                    "fkey_col": row["fkey_col"],
                    "parent_table": row["parent_table"],
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                }
            )
    return pd.DataFrame(rows)


def build_fidelity_report(
    real_summary_path: Path,
    synthetic_summary_paths: list[Path],
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    metric_names = metric_names or DEFAULT_METRICS
    real = load_metric_frame(real_summary_path, cohort="real")
    report_rows: list[dict[str, Any]] = []

    for synthetic_summary_path in synthetic_summary_paths:
        cohort = _cohort_name_from_summary_path(synthetic_summary_path)
        synthetic = load_metric_frame(synthetic_summary_path, cohort=cohort)
        for metric_name in metric_names:
            real_values = _finite_metric_values(real, metric_name)
            synth_values = _finite_metric_values(synthetic, metric_name)
            if real_values.size == 0 or synth_values.size == 0:
                continue
            report_rows.append(
                {
                    "cohort": cohort,
                    "metric_name": metric_name,
                    "num_real_edges": int(real_values.size),
                    "num_synth_edges": int(synth_values.size),
                    "real_mean": float(real_values.mean()),
                    "synth_mean": float(synth_values.mean()),
                    "mean_abs_error": float(abs(real_values.mean() - synth_values.mean())),
                    "ks_distance": _ks_distance(real_values, synth_values),
                    "wasserstein": _wasserstein_1d(real_values, synth_values),
                }
            )

    return pd.DataFrame(report_rows)


def _cohort_name_from_summary_path(summary_path: Path) -> str:
    parent_name = summary_path.expanduser().parent.name
    if parent_name:
        return parent_name
    return summary_path.stem


def _finite_metric_values(frame: pd.DataFrame, metric_name: str) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=float)
    values = frame.loc[frame["metric_name"] == metric_name, "metric_value"].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(left)
    right = np.sort(right)
    grid = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, grid, side="right") / left.size
    right_cdf = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(left)
    right = np.sort(right)
    quantiles = np.linspace(0.0, 1.0, num=max(left.size, right.size), endpoint=True)
    left_q = np.quantile(left, quantiles)
    right_q = np.quantile(right, quantiles)
    return float(np.mean(np.abs(left_q - right_q)))


def write_report(report: pd.DataFrame, output_path: Path) -> None:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(output_path, index=False)


def main(
    real_summary_path: Path,
    synthetic_summary_paths: list[Path],
    output_path: Path,
    metric_names: list[str] | None,
) -> None:
    report = build_fidelity_report(
        real_summary_path=real_summary_path,
        synthetic_summary_paths=synthetic_summary_paths,
        metric_names=metric_names,
    )
    write_report(report=report, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare real and synthetic topology fidelity.")
    parser.add_argument("--real_summary", type=Path, required=True)
    parser.add_argument("--synthetic_summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to compare. Defaults to topology fidelity metrics.",
    )
    args = parser.parse_args()

    main(
        real_summary_path=args.real_summary,
        synthetic_summary_paths=args.synthetic_summaries,
        output_path=args.output_path,
        metric_names=args.metrics,
    )
