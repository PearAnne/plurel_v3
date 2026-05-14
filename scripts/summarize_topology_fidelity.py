from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRIMARY_METRICS = [
    "fanout_gini",
    "fanout_ks_to_poisson",
    "fanout_ks_to_powerlaw",
    "isolated_parent_rate",
    "degree_assortativity",
]

SECONDARY_METRICS = [
    "null_rate",
    "powerlaw_gamma",
    "temporal_growth_alpha",
]

DISTANCE_COLUMNS = ["mean_abs_error", "ks_distance", "wasserstein"]


def population_name_from_path(report_path: Path) -> str:
    stem = report_path.stem
    if stem.startswith("fidelity_"):
        return stem.removeprefix("fidelity_")
    return stem


def load_report(report_path: Path, population: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(report_path.expanduser())
    required = {"cohort", "metric_name", *DISTANCE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{report_path} is missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["population"] = population or population_name_from_path(report_path)
    result["cohort"] = result["cohort"].str.replace("topology_stats.", "", regex=False)
    return result


def build_per_metric_rank(
    reports: list[Path],
    primary_metrics: list[str] | None = None,
) -> pd.DataFrame:
    metric_names = primary_metrics or PRIMARY_METRICS
    frames = [load_report(report_path) for report_path in reports]
    combined = pd.concat(frames, ignore_index=True)
    ranked = combined[combined["metric_name"].isin(metric_names)].copy()
    for distance_col in DISTANCE_COLUMNS:
        ranked[f"{distance_col}_rank"] = ranked.groupby(["population", "metric_name"])[
            distance_col
        ].rank(method="min")
    return ranked


def build_cohort_rank(per_metric_rank: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (population, cohort), group in per_metric_rank.groupby(["population", "cohort"]):
        row: dict[str, float | str | int] = {
            "population": population,
            "cohort": cohort,
            "num_metrics": int(group["metric_name"].nunique()),
        }
        for distance_col in DISTANCE_COLUMNS:
            rank_col = f"{distance_col}_rank"
            row[f"{distance_col}_mean_rank"] = float(group[rank_col].mean())
            row[f"{distance_col}_median_rank"] = float(group[rank_col].median())
            row[f"{distance_col}_mean"] = float(group[distance_col].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["population", "ks_distance_mean_rank", "wasserstein_mean_rank", "mean_abs_error_mean_rank"]
    ).reset_index(drop=True)


def build_secondary_metrics(reports: list[Path]) -> pd.DataFrame:
    frames = [load_report(report_path) for report_path in reports]
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["metric_name"].isin(SECONDARY_METRICS)].copy()


def render_markdown(
    cohort_rank: pd.DataFrame,
    per_metric_rank: pd.DataFrame,
    secondary_metrics: pd.DataFrame,
    reports: list[Path],
) -> str:
    lines = [
        "# Edge Prior Topology Fidelity Summary",
        "",
        "This report is topology-only. It ranks synthetic cohorts by distance to real FK-edge topology distributions and does not claim downstream RT/RFM improvement.",
        "",
        "## Inputs",
        "",
    ]
    for report_path in reports:
        lines.append(f"- `{report_path}`")
    lines.extend(
        [
            "",
            "## Primary Ranking",
            "",
            "Primary ranking uses per-metric ranks over: "
            + ", ".join(f"`{metric}`" for metric in PRIMARY_METRICS)
            + ". Lower rank is better.",
            "",
        ]
    )
    for population, group in cohort_rank.groupby("population", sort=False):
        display_cols = [
            "cohort",
            "ks_distance_mean_rank",
            "wasserstein_mean_rank",
            "mean_abs_error_mean_rank",
            "num_metrics",
        ]
        lines.extend([f"### {population}", "", _markdown_table(group[display_cols]), ""])

    lines.extend(["## Per-Metric Winners", ""])
    for population, group in per_metric_rank.groupby("population", sort=False):
        lines.extend([f"### {population}", ""])
        winners = []
        for metric_name, metric_group in group.groupby("metric_name"):
            best_idx = metric_group["wasserstein"].idxmin()
            best = metric_group.loc[best_idx]
            winners.append(
                {
                    "metric_name": metric_name,
                    "cohort": best["cohort"],
                    "real_mean": best.get("real_mean"),
                    "synth_mean": best.get("synth_mean"),
                    "wasserstein": best["wasserstein"],
                    "ks_distance": best["ks_distance"],
                }
            )
        lines.extend([_markdown_table(pd.DataFrame(winners)), ""])

    lines.extend(
        [
            "## Secondary Diagnostics",
            "",
            "`null_rate`, `powerlaw_gamma`, and `temporal_growth_alpha` are reported as diagnostics rather than included in the primary rank because they are either not mature generation targets or have unstable scale/missingness in the current synthetic runs.",
            "",
        ]
    )
    if secondary_metrics.empty:
        lines.append("No secondary metrics were available.")
    else:
        diagnostic = secondary_metrics[
            [
                "population",
                "cohort",
                "metric_name",
                "real_mean",
                "synth_mean",
                "mean_abs_error",
                "ks_distance",
                "wasserstein",
                "num_synth_edges",
            ]
        ].sort_values(["population", "metric_name", "wasserstein"])
        lines.append(_markdown_table(diagnostic))

    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "A high topology-fidelity rank only justifies moving a cohort into controlled RT pretraining. It does not justify changing the default generator or claiming downstream improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = []
    for _, row in frame.iterrows():
        rows.append([_format_markdown_cell(row[column]) for column in columns])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_outputs(
    reports: list[Path],
    output_dir: Path,
    primary_metrics: list[str] | None = None,
) -> None:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_metric_rank = build_per_metric_rank(reports=reports, primary_metrics=primary_metrics)
    cohort_rank = build_cohort_rank(per_metric_rank=per_metric_rank)
    secondary_metrics = build_secondary_metrics(reports=reports)

    per_metric_rank.to_csv(output_dir / "per_metric_rank.csv", index=False)
    cohort_rank.to_csv(output_dir / "cohort_rank.csv", index=False)
    secondary_metrics.to_csv(output_dir / "secondary_metrics.csv", index=False)
    (output_dir / "summary.md").write_text(
        render_markdown(
            cohort_rank=cohort_rank,
            per_metric_rank=per_metric_rank,
            secondary_metrics=secondary_metrics,
            reports=reports,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize edge-prior topology fidelity reports.")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--primary_metrics",
        nargs="+",
        default=None,
        help="Optional primary metrics for rank aggregation.",
    )
    args = parser.parse_args()

    write_outputs(
        reports=args.reports,
        output_dir=args.output_dir,
        primary_metrics=args.primary_metrics,
    )


if __name__ == "__main__":
    main()
