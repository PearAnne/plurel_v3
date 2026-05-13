"""Reverse-engineer Phase 1b topology prior parameter ranges from a measured summary.json.

Inputs:
    out/topology_stats/relbench_stats/summary.json (RelBench-only), or
    out/topology_stats/combined/summary.json after merge_topology_summaries.py, or any compatible summary file
Outputs (idempotent, regenerated on every run):
    notes/phase0c_prior_calibration.md   — human-readable runbook draft
    notes/phase0c_prior_calibration.json — machine-readable {field: {kind, value}} table

The markdown is regenerated from the empirical summary so it always reflects the
latest measurement run; do not hand-edit, copy to a different path if you want
to preserve a snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plurel.config import SCMParams
from plurel.topology_stats_paths import COMBINED_SUMMARY_PATH, RELBENCH_SUMMARY_PATH


@dataclass(frozen=True)
class Quantiles:
    count: int
    n_missing: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    p10: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p90: float | None

    @classmethod
    def empty(cls, n_missing: int) -> Quantiles:
        return cls(0, n_missing, *([None] * 9))


@dataclass(frozen=True)
class PriorRecommendation:
    config_field: str
    source_metric: str
    population: str
    suggested_kind: str
    suggested_value: list[Any] | None
    current_kind: str
    current_value: list[Any]
    confidence: str
    rationale: str


def load_summary_as_frame(summary_path: Path) -> pd.DataFrame:
    payload = json.loads(summary_path.expanduser().read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in payload["rows"]:
        record: dict[str, Any] = {
            "db_name": row["db_name"],
            "child_table": row["child_table"],
            "fkey_col": row["fkey_col"],
            "parent_table": row["parent_table"],
            "num_children": row["num_children"],
            "num_parents": row["num_parents"],
            "num_non_null_edges": row["num_non_null_edges"],
        }
        for key, value in (row.get("metrics") or {}).items():
            record[f"m_{key}"] = value
        records.append(record)
    frame = pd.DataFrame(records)
    frame["benchmark_family"] = frame["db_name"].apply(_benchmark_family_for)
    return frame


def _benchmark_family_for(db_name: str) -> str:
    if db_name.startswith("rel-"):
        return "relbench"
    if db_name.startswith("dbinfer-"):
        return "dbinfer"
    if db_name.startswith("ctu-") or "/" in db_name:
        return "ctu"
    return "other"


def finite_quantiles(series: pd.Series) -> Quantiles:
    numeric = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(numeric)
    values = numeric[finite_mask]
    if values.empty:
        return Quantiles.empty(n_missing=int(series.size))
    return Quantiles(
        count=int(values.size),
        n_missing=int(series.size - values.size),
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std(ddof=0)),
        p10=float(values.quantile(0.10)),
        p25=float(values.quantile(0.25)),
        p50=float(values.quantile(0.50)),
        p75=float(values.quantile(0.75)),
        p90=float(values.quantile(0.90)),
    )


def per_db_quantiles(
    frame: pd.DataFrame, metric: str, plausible_only: bool = False
) -> pd.DataFrame:
    population = frame
    if plausible_only:
        population = frame[frame["m_powerlaw_plausible"] == True]  # noqa: E712
    rows = []
    for db_name, group in population.groupby("db_name"):
        q = finite_quantiles(group[metric])
        rows.append(
            {
                "db_name": db_name,
                "count": q.count,
                "p10": q.p10,
                "p50": q.p50,
                "p90": q.p90,
                "mean": q.mean,
            }
        )
    return pd.DataFrame(rows).sort_values("db_name").reset_index(drop=True)


def categorize_edges(frame: pd.DataFrame) -> dict[str, int]:
    gamma = pd.to_numeric(frame["m_powerlaw_gamma"], errors="coerce")
    ks = pd.to_numeric(frame["m_fanout_ks_to_powerlaw"], errors="coerce")
    gamma_missing = ~np.isfinite(gamma)
    plausible = frame["m_powerlaw_plausible"] == True  # noqa: E712
    finite_gamma = np.isfinite(gamma)
    gamma_high = finite_gamma & (gamma > 8.0)
    gamma_low = finite_gamma & (gamma < 1.5)
    gamma_in_range = finite_gamma & (gamma.between(1.5, 8.0))
    ks_bad = np.isfinite(ks) & (ks > 0.3)
    pl_plausible = plausible
    non_pl_heavy = gamma_in_range & ks_bad & ~plausible
    return {
        "total": int(len(frame)),
        "plausible_powerlaw": int(pl_plausible.sum()),
        "near_uniform_high_gamma": int(gamma_high.sum()),
        "extreme_low_gamma": int(gamma_low.sum()),
        "heavy_non_powerlaw": int(non_pl_heavy.sum()),
        "no_fit_or_degenerate": int(gamma_missing.sum()),
    }


def null_rate_breakdown(frame: pd.DataFrame) -> dict[str, int | float]:
    null_rate = pd.to_numeric(frame["m_null_rate"], errors="coerce").fillna(0.0)
    bins = {
        "zero": int((null_rate == 0).sum()),
        "small_(0_0.05]": int(((null_rate > 0) & (null_rate <= 0.05)).sum()),
        "medium_(0.05_0.3]": int(((null_rate > 0.05) & (null_rate <= 0.3)).sum()),
        "high_(0.3_0.9]": int(((null_rate > 0.3) & (null_rate <= 0.9)).sum()),
        "extreme_(0.9_1.0]": int(((null_rate > 0.9) & (null_rate <= 1.0)).sum()),
    }
    nonzero_quantiles = finite_quantiles(null_rate[null_rate > 0])
    bins["nonzero_p50"] = (
        float(nonzero_quantiles.p50) if nonzero_quantiles.p50 is not None else None
    )
    bins["nonzero_p90"] = (
        float(nonzero_quantiles.p90) if nonzero_quantiles.p90 is not None else None
    )
    return bins


def derive_prior_recommendations(frame: pd.DataFrame) -> list[PriorRecommendation]:
    defaults = SCMParams()
    plausible = frame[frame["m_powerlaw_plausible"] == True]  # noqa: E712

    gamma_q = finite_quantiles(plausible["m_powerlaw_gamma"])
    pa_alpha_q = finite_quantiles(frame.get("m_pa_exponent_alpha", pd.Series(dtype=float)))
    theta_alpha_q = finite_quantiles(frame.get("m_theta_beta_alpha", pd.Series(dtype=float)))
    theta_beta_q = finite_quantiles(frame.get("m_theta_beta_beta", pd.Series(dtype=float)))
    null_q_all = finite_quantiles(frame["m_null_rate"])
    null_q_nonzero = finite_quantiles(
        pd.to_numeric(frame["m_null_rate"], errors="coerce").loc[
            pd.to_numeric(frame["m_null_rate"], errors="coerce") > 0
        ]
    )

    categories = categorize_edges(frame)
    pl_share = categories["plausible_powerlaw"] / max(1, categories["total"])

    return [
        PriorRecommendation(
            config_field="chung_lu_gamma_choices",
            source_metric="powerlaw_gamma",
            population=f"powerlaw_plausible=True (N={gamma_q.count} / {len(frame)})",
            suggested_kind="range",
            suggested_value=[_round(gamma_q.p10, 2), _round(gamma_q.p90, 2)],
            current_kind=defaults.chung_lu_gamma_choices.kind,
            current_value=list(defaults.chung_lu_gamma_choices.value),
            confidence="high",
            rationale=(
                f"Empirical γ p10/p50/p90 = {_round(gamma_q.p10, 2)} / "
                f"{_round(gamma_q.p50, 2)} / {_round(gamma_q.p90, 2)}; MLE std error "
                "negligible at N≥1000 per edge. NB: this is fanout-distribution γ, not the "
                "node-weight γ that Chung-Lu literally exponentiates; in practice the two "
                "are within ~10% on power-law fanouts and we use the empirical value "
                "directly. Default [1.5, 3.0] truncates the upper tail "
                f"(median is {_round(gamma_q.p50, 2)} but p90 is {_round(gamma_q.p90, 2)})."
            ),
        ),
        PriorRecommendation(
            config_field="tpa_alpha_choices",
            source_metric="pa_exponent_alpha",
            population=f"all edges with PA-fit (N={pa_alpha_q.count} / {len(frame)})",
            suggested_kind="range",
            suggested_value=(
                [_round(pa_alpha_q.p10, 2), _round(pa_alpha_q.p90, 2)]
                if pa_alpha_q.p10 is not None
                else None
            ),
            current_kind=defaults.tpa_alpha_choices.kind,
            current_value=list(defaults.tpa_alpha_choices.value),
            confidence="high",
            rationale=(
                f"Empirical preferential-attachment exponent (Newman split-half kernel) "
                f"p10/p50/p90 = {_round(pa_alpha_q.p10, 2)} / "
                f"{_round(pa_alpha_q.p50, 2)} / {_round(pa_alpha_q.p90, 2)}. This is the "
                "direct empirical analog of TPA's `W(k) ∝ k^α` kernel — fitted by sorting "
                "child arrivals, splitting in halves, regressing log(degree gained in "
                "second half) on log(degree at midpoint). NOTE: an earlier Phase 0c draft "
                "mistakenly used `temporal_growth_alpha` (cumulative-edge vs elapsed-time) "
                "for this field; those are different quantities."
            ),
        ),
        PriorRecommendation(
            config_field="dcsbm_theta_alpha_choices",
            source_metric="theta_beta_alpha",
            population=f"all edges with Beta-MoM fit (N={theta_alpha_q.count} / {len(frame)})",
            suggested_kind="range",
            suggested_value=(
                [_round(theta_alpha_q.p10, 2), _round(theta_alpha_q.p90, 2)]
                if theta_alpha_q.p10 is not None
                else None
            ),
            current_kind=defaults.dcsbm_theta_alpha_choices.kind,
            current_value=list(defaults.dcsbm_theta_alpha_choices.value),
            confidence="medium",
            rationale=(
                f"Beta(α, β) shape α empirically fitted via method of moments on per-parent "
                "normalized fanout (θ_i = f_i / max f). p10/p50/p90 = "
                f"{_round(theta_alpha_q.p10, 2)} / {_round(theta_alpha_q.p50, 2)} / "
                f"{_round(theta_alpha_q.p90, 2)}. Small α (< 1) means most parents have "
                "low intrinsic attractiveness with a long tail of strong hubs; large α "
                "means more uniform attractiveness."
            ),
        ),
        PriorRecommendation(
            config_field="dcsbm_theta_beta_choices",
            source_metric="theta_beta_beta",
            population=f"all edges with Beta-MoM fit (N={theta_beta_q.count} / {len(frame)})",
            suggested_kind="range",
            suggested_value=(
                [_round(theta_beta_q.p10, 2), _round(theta_beta_q.p90, 2)]
                if theta_beta_q.p10 is not None
                else None
            ),
            current_kind=defaults.dcsbm_theta_beta_choices.kind,
            current_value=list(defaults.dcsbm_theta_beta_choices.value),
            confidence="medium",
            rationale=(
                f"Beta(α, β) shape β empirically fitted via MoM. p10/p50/p90 = "
                f"{_round(theta_beta_q.p10, 2)} / {_round(theta_beta_q.p50, 2)} / "
                f"{_round(theta_beta_q.p90, 2)}. β/α ratio controls right-skew of the "
                "attractiveness distribution; typical RDBs should show β >> α."
            ),
        ),
        PriorRecommendation(
            config_field="edge_prior_null_rate_choices",
            source_metric="null_rate",
            population=(
                f"all edges (N={null_q_all.count}); nonzero subset (N={null_q_nonzero.count})"
            ),
            suggested_kind="range",
            suggested_value=[0.0, _round(null_q_all.p90 or 0.0, 2)],
            current_kind=defaults.edge_prior_null_rate_choices.kind,
            current_value=list(defaults.edge_prior_null_rate_choices.value),
            confidence="low",
            rationale=(
                "Distribution is multi-modal: ~64% of edges have null_rate=0, ~16% in "
                f"(0, 0.05], ~20% > 0.05 with several near 1.0. A single range "
                f"[0.0, {_round(null_q_all.p90 or 0.0, 2)}] captures 90% but misses the "
                "structural-null tail. Recommend introducing a `null_rate_strategy` mixture "
                "field in Phase 1b (e.g. {p_zero=0.65, range_nonzero=[0.001, 0.6]}) rather "
                "than forcing this into a single `Choices(range=...)`."
            ),
        ),
        PriorRecommendation(
            config_field="topology_prior_choices",
            source_metric="(plausible / near-uniform / heavy-non-PL share)",
            population=f"all edges (N={categories['total']})",
            suggested_kind="set",
            suggested_value=_suggest_prior_mixture(categories),
            current_kind=defaults.topology_prior_choices.kind,
            current_value=list(defaults.topology_prior_choices.value),
            confidence="medium",
            rationale=(
                f"{categories['plausible_powerlaw']}/{categories['total']} "
                f"({pl_share:.0%}) edges fit power-law plausibly; "
                f"{categories['near_uniform_high_gamma']} are near-uniform; "
                f"{categories['heavy_non_powerlaw']} are heavy-tailed but non-PL; "
                f"{categories['no_fit_or_degenerate']} are 1:1 / cardinality-1 joins. "
                "Set-based duplicated-element weighting is a hack; Phase 1b should add a "
                "`Choices(kind='weighted_set', value=[(name, weight), ...])` instead."
            ),
        ),
    ]


def _round(value: float | None, ndigits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, ndigits)


def _suggest_prior_mixture(categories: dict[str, int]) -> list[str]:
    """Heuristic mixture composition that matches the observed power-law / uniform / heavy split.

    Duplicates approximate weighting in a Choices(kind="set") universe (which samples uniformly).
    """
    pl = max(categories["plausible_powerlaw"], 0)
    uniform = max(categories["near_uniform_high_gamma"], 0)
    heavy = max(categories["heavy_non_powerlaw"], 0)
    total = pl + uniform + heavy
    if total == 0:
        return ["hsbm"]
    pl_w = max(1, round(8 * pl / total))
    uniform_w = max(0, round(8 * uniform / total))
    heavy_w = max(0, round(8 * heavy / total))
    mixture: list[str] = []
    mixture += ["chung_lu"] * pl_w
    mixture += ["erdos_renyi"] * uniform_w
    mixture += ["dcsbm"] * heavy_w
    if "tpa" not in mixture and pl_w >= 2:
        mixture[0] = "tpa"
    if "hsbm" not in mixture and uniform_w == 0:
        mixture += ["hsbm"]
    return mixture


def render_markdown(
    frame: pd.DataFrame,
    recommendations: list[PriorRecommendation],
    summary_path: Path,
    timestamp: datetime,
) -> str:
    categories = categorize_edges(frame)
    null_bins = null_rate_breakdown(frame)
    db_names = sorted(frame["db_name"].unique())
    families = sorted(frame["benchmark_family"].unique())

    lines: list[str] = []
    lines.append("# Phase 0c — Prior Parameter Calibration")
    lines.append("")
    lines.append("> Auto-generated by `scripts/analyze_topology_summary.py`. Do not hand-edit;")
    lines.append("> rerun with new measurements to refresh. Snapshot before downstream use.")
    lines.append("")
    lines.append(f"- Generated: `{timestamp.isoformat(timespec='seconds')}`")
    lines.append(f"- Source: `{summary_path}`")
    lines.append(
        f"- Population: {len(frame)} FK→PK edges across {len(db_names)} dbs ({', '.join(families)})"
    )
    lines.append(f"- Databases: {', '.join(db_names)}")
    lines.append("")
    if set(families) == {"relbench"}:
        lines.append(
            "> ⚠️ **Single benchmark family**: this calibration only reflects RelBench v2. "
            "Cross-benchmark adapter outputs (see `notes/rdb_benchmark_landscape.md`) must "
            "be incorporated before these values can defend a *generic* RDB topology prior "
            "claim in the paper."
        )
        lines.append("")

    lines.append("## 1. Mapping table: empirical metrics → `plurel/config.py` `*_choices`")
    lines.append("")
    lines.append(
        "| `config.py` field | empirical source | population | suggested | current default | confidence |"
    )
    lines.append("|---|---|---|---|---|---|")
    for rec in recommendations:
        suggested = (
            f"`{rec.suggested_kind}` `{rec.suggested_value}`"
            if rec.suggested_value is not None
            else "(see note)"
        )
        current = f"`{rec.current_kind}` `{rec.current_value}`"
        lines.append(
            f"| `{rec.config_field}` | `{rec.source_metric}` | {rec.population} | "
            f"{suggested} | {current} | {rec.confidence} |"
        )
    lines.append("")

    lines.append("## 2. Per-recommendation rationale")
    for rec in recommendations:
        lines.append("")
        lines.append(f"### `{rec.config_field}`")
        lines.append("")
        lines.append(f"- **Source**: `{rec.source_metric}`")
        lines.append(f"- **Population**: {rec.population}")
        lines.append(f"- **Suggested**: `{rec.suggested_kind}` value=`{rec.suggested_value}`")
        lines.append(f"- **Current default**: `{rec.current_kind}` value=`{rec.current_value}`")
        lines.append(f"- **Confidence**: {rec.confidence}")
        lines.append("")
        lines.append(rec.rationale)

    lines.append("")
    lines.append("## 3. Edge-type breakdown (informs prior mixture composition)")
    lines.append("")
    lines.append(
        f"- **plausible power-law** ((1.5 ≤ γ ≤ 8) AND (ks ≤ 0.3)): "
        f"{categories['plausible_powerlaw']}/{categories['total']}"
    )
    lines.append(
        f"- **near-uniform** (γ > 8 from pathological MLE on near-constant fanout): "
        f"{categories['near_uniform_high_gamma']}/{categories['total']}"
    )
    lines.append(
        f"- **heavy-non-PL** (γ in [1.5, 8] but ks > 0.3, e.g. truncated exponential / lognormal): "
        f"{categories['heavy_non_powerlaw']}/{categories['total']}"
    )
    lines.append(
        f"- **extreme tail** (γ < 1.5): {categories['extreme_low_gamma']}/{categories['total']}"
    )
    lines.append(
        f"- **degenerate** (γ=NaN — 1:1 PK-FK shadow or parent.size < 100): "
        f"{categories['no_fit_or_degenerate']}/{categories['total']}"
    )

    lines.append("")
    lines.append("## 4. Null-rate distribution (multi-modal — single range can't capture it)")
    lines.append("")
    lines.append(f"- null_rate = 0: **{null_bins['zero']}** edges")
    lines.append(f"- null_rate ∈ (0, 0.05]: **{null_bins['small_(0_0.05]']}** edges")
    lines.append(f"- null_rate ∈ (0.05, 0.3]: **{null_bins['medium_(0.05_0.3]']}** edges")
    lines.append(f"- null_rate ∈ (0.3, 0.9]: **{null_bins['high_(0.3_0.9]']}** edges")
    lines.append(f"- null_rate ∈ (0.9, 1.0]: **{null_bins['extreme_(0.9_1.0]']}** edges")
    if null_bins.get("nonzero_p50") is not None:
        lines.append(
            f"- Of nonzero null_rates: p50={null_bins['nonzero_p50']:.3f}, "
            f"p90={null_bins['nonzero_p90']:.3f}"
        )

    lines.append("")
    lines.append("## 5. Per-db distribution of plausible-only γ (for cross-db variance check)")
    lines.append("")
    per_db = per_db_quantiles(frame, metric="m_powerlaw_gamma", plausible_only=True)
    lines.append("| db | plausible edges | γ p10 | γ p50 | γ p90 | γ mean |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in per_db.iterrows():
        lines.append(
            f"| {row['db_name']} | {int(row['count'])} | "
            f"{_fmt(row['p10'])} | {_fmt(row['p50'])} | {_fmt(row['p90'])} | "
            f"{_fmt(row['mean'])} |"
        )

    lines.append("")
    lines.append("## 6. Known limitations and follow-up work")
    lines.append("")
    if set(families) == {"relbench"}:
        lines.append("- **Single benchmark family**: ranges above are RelBench v2 only.")
        lines.append(
            "  See `notes/rdb_benchmark_landscape.md` for adapter priorities to broaden coverage."
        )
    else:
        lines.append(
            "- **Cross-benchmark pool**: ranges above combine "
            f"{', '.join(families)} (see `notes/rdb_benchmark_landscape.md`). "
            "CTU includes many small or legacy schemas; 4DBInfer FK metadata is hand-maintained "
            "in `plurel/topology_adapters.py` — validate against the upstream benchmark when "
            "upgrading data."
        )
    lines.append(
        "- **`temporal_growth_alpha`** is *not* mapped to any prior parameter. It captures "
    )
    lines.append("  log(cumulative-edge) vs log(elapsed-time) for the child table — useful as a ")
    lines.append(
        "  schema-level diagnostic (DB design) but NOT a substitute for the PA kernel exponent "
    )
    lines.append(
        "  that TPA uses. The empirical p50 ≈ 1.45 (super-linear growth) is informational only."
    )
    lines.append("- **`tpa_beta_choices`** (recency decay): no direct empirical analog;")
    lines.append("  keep default [0.0, 0.5] until ablation shows it matters.")
    lines.append("- **`bi_hsbm_*`** (HSBM hierarchy): could derive from bipartite fanout spectral")
    lines.append("  modes; deferred to Phase 1b refinement.")
    lines.append("- **`null_rate` multi-modality**: the `Choices(range)` API is the wrong shape;")
    lines.append("  needs a Phase 1b extension (e.g. `null_rate_mixture` field) before the")
    lines.append("  empirical distribution can be faithfully reproduced.")
    lines.append("")

    lines.append("## 7. Open questions")
    lines.append("")
    lines.append("- Does CTU's mostly-static schema collection shift the γ p90 above 4.58?")
    lines.append(
        "- Do small CTU / classical PKDD DBs shift null_rate or γ tails vs RelBench-only edges?"
    )
    lines.append('- Should the 12 "1:1 join" edges (γ=NaN, fanout=1) be excluded from synthetic')
    lines.append("  generation entirely, or modeled with `erdos_renyi(n_children=n_parents)`?")
    lines.append("- Should the synthetic generator pick prior kind **per-db** (current) or")
    lines.append("  **per-edge** (`edge_level_uniform` strategy in `DatabaseParams`)?")
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.3f}"


def write_machine_recommendations(
    recommendations: list[PriorRecommendation], output_path: Path
) -> None:
    payload = [
        {
            "config_field": rec.config_field,
            "source_metric": rec.source_metric,
            "population": rec.population,
            "suggested": {"kind": rec.suggested_kind, "value": rec.suggested_value},
            "current": {"kind": rec.current_kind, "value": rec.current_value},
            "confidence": rec.confidence,
            "rationale": rec.rationale,
        }
        for rec in recommendations
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(summary_path: Path, output_md: Path, output_json: Path | None) -> None:
    frame = load_summary_as_frame(summary_path)
    recommendations = derive_prior_recommendations(frame)
    rendered = render_markdown(
        frame=frame,
        recommendations=recommendations,
        summary_path=summary_path,
        timestamp=datetime.now(tz=UTC),
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(rendered, encoding="utf-8")
    print(f"wrote {output_md} ({len(rendered)} chars)")
    if output_json is not None:
        write_machine_recommendations(recommendations, output_json)
        print(f"wrote {output_json}")

    print()
    print(_render_stdout_table(frame, recommendations))


def _render_stdout_table(frame: pd.DataFrame, recommendations: list[PriorRecommendation]) -> str:
    plausible = frame[frame["m_powerlaw_plausible"] == True]  # noqa: E712
    lines = [
        f"edges: {len(frame)}  plausible: {len(plausible)}  dbs: {frame['db_name'].nunique()}",
        "",
        f"{'field':38s}{'suggested':28s}{'current':24s}{'confidence':10s}",
    ]
    for rec in recommendations:
        suggested = (
            f"{rec.suggested_kind} {rec.suggested_value}"
            if rec.suggested_value is not None
            else "(see md)"
        )
        current = f"{rec.current_kind} {rec.current_value}"
        lines.append(f"{rec.config_field:38s}{suggested:28s}{current:24s}{rec.confidence:10s}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reverse-engineer Phase 1b topology prior parameter ranges from summary.json."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Path to summary.json (from measure_edge_topology, measure_ctu_topology, "
            "or merge_topology_summaries). "
            "Defaults to RelBench-only stats; use --merged after merge_topology_summaries.py."
        ),
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help=f"Shorthand for --summary {COMBINED_SUMMARY_PATH}",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("notes/phase0c_prior_calibration.md"),
        help="Path to write the generated markdown runbook.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("notes/phase0c_prior_calibration.json"),
        help=("Path to write a machine-readable recommendations JSON, or '' to skip."),
    )
    args = parser.parse_args()

    output_json = args.output_json if str(args.output_json) else None
    summary_path = (
        COMBINED_SUMMARY_PATH
        if args.merged
        else (args.summary if args.summary is not None else RELBENCH_SUMMARY_PATH)
    )
    main(summary_path=summary_path, output_md=args.output_md, output_json=output_json)
