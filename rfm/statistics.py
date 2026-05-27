from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from rfm.types import SyntheticDataset


@dataclass(frozen=True)
class PriorStatisticsReport:
    sample_count: int
    prior_counts: dict[str, int]
    task_counts: dict[str, int]
    graph_layout_counts: dict[str, int]
    source_kind_counts: dict[str, int]
    aggregation_kind_counts: dict[str, int]
    mechanism_kind_counts: dict[str, int]
    mean_rows: float
    mean_features: float
    categorical_feature_fraction: float
    temporal_fraction: float
    ood_fraction: float
    dynamic_dataset_fraction: float
    dynamic_node_fraction: float
    mean_edge_density: float
    mean_in_degree: float
    mean_target_ancestor_fraction: float
    mean_abs_feature_corr: float
    mean_max_feature_corr: float
    mean_feature_autocorr_lag1: float
    mean_feature_spectral_entropy: float
    mean_target_autocorr_lag1: float
    mean_train_test_feature_shift: float
    mean_class_entropy: float
    mean_class_imbalance: float
    mean_num_classes: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PriorQualityThresholds:
    min_graph_layouts: int = 5
    min_source_kinds: int = 6
    min_aggregation_kinds: int = 5
    min_mechanism_kinds: int = 5
    min_dynamic_dataset_fraction: float = 0.15
    min_dynamic_node_fraction: float = 0.05
    min_mean_abs_feature_corr: float = 0.03
    max_mean_abs_feature_corr: float = 0.85
    min_mean_feature_autocorr_lag1_when_temporal: float = 0.05
    min_mean_feature_spectral_entropy: float = 0.15
    min_mean_train_test_feature_shift_when_ood: float = 0.05
    min_mean_class_entropy: float = 0.45
    max_mean_class_imbalance: float = 0.92


@dataclass(frozen=True)
class PriorQualityGate:
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_datasets(datasets: Sequence[SyntheticDataset]) -> PriorStatisticsReport:
    if len(datasets) == 0:
        raise ValueError("datasets must be non-empty")

    prior_counts = Counter(dataset.meta.prior_type for dataset in datasets)
    task_counts = Counter(dataset.meta.task_type for dataset in datasets)
    graph_layout_counts = Counter(dataset.meta.graph_layout or "" for dataset in datasets)
    graph_layout_counts.pop("", None)
    source_kind_counts = Counter(kind for dataset in datasets for kind in dataset.meta.source_kinds)
    aggregation_kind_counts = Counter(
        kind for dataset in datasets for kind in dataset.meta.aggregation_kinds
    )
    mechanism_kind_counts = Counter(
        kind for dataset in datasets for kind in dataset.meta.mechanism_kinds
    )

    rows = np.array([dataset.meta.num_rows for dataset in datasets], dtype=np.float64)
    features = np.array([dataset.meta.num_features for dataset in datasets], dtype=np.float64)
    categorical_feature_counts = np.array(
        [len(dataset.meta.categorical_features) for dataset in datasets], dtype=np.float64
    )
    temporal_flags = np.array([dataset.meta.temporal for dataset in datasets], dtype=np.float64)
    ood_flags = np.array([dataset.meta.ood for dataset in datasets], dtype=np.float64)
    dynamic_counts = np.array(
        [len(dataset.meta.dynamic_nodes) for dataset in datasets], dtype=np.float64
    )
    node_counts = np.array(
        [max(dataset.meta.target_node + 1, 1) for dataset in datasets], dtype=np.float64
    )

    edge_densities = []
    mean_in_degrees = []
    target_ancestor_fractions = []
    mean_abs_corrs = []
    max_abs_corrs = []
    feature_autocorrs = []
    feature_spectral_entropies = []
    target_autocorrs = []
    train_test_shifts = []
    class_entropies = []
    class_imbalances = []
    num_classes = []

    for dataset in datasets:
        edge_density, mean_in_degree, target_ancestor_fraction = _graph_metrics(dataset)
        edge_densities.append(edge_density)
        mean_in_degrees.append(mean_in_degree)
        target_ancestor_fractions.append(target_ancestor_fraction)
        mean_abs_corr, max_abs_corr = _feature_corr_metrics(dataset.x)
        mean_abs_corrs.append(mean_abs_corr)
        max_abs_corrs.append(max_abs_corr)
        feature_autocorrs.append(_mean_feature_autocorr(dataset.x))
        feature_spectral_entropies.append(_mean_feature_spectral_entropy(dataset.x))
        target_autocorrs.append(_autocorr(dataset.y.astype(np.float64)))
        train_test_shifts.append(_train_test_feature_shift(dataset.x, dataset.meta.train_size))
        if dataset.meta.task_type == "classification":
            entropy, imbalance = _class_metrics(
                dataset.y.astype(np.int64), dataset.meta.num_classes
            )
            class_entropies.append(entropy)
            class_imbalances.append(imbalance)
            if dataset.meta.num_classes is not None:
                num_classes.append(float(dataset.meta.num_classes))

    report = PriorStatisticsReport(
        sample_count=len(datasets),
        prior_counts=dict(prior_counts),
        task_counts=dict(task_counts),
        graph_layout_counts=dict(graph_layout_counts),
        source_kind_counts=dict(source_kind_counts),
        aggregation_kind_counts=dict(aggregation_kind_counts),
        mechanism_kind_counts=dict(mechanism_kind_counts),
        mean_rows=float(np.mean(rows)),
        mean_features=float(np.mean(features)),
        categorical_feature_fraction=float(
            np.sum(categorical_feature_counts) / max(np.sum(features), 1.0)
        ),
        temporal_fraction=float(np.mean(temporal_flags)),
        ood_fraction=float(np.mean(ood_flags)),
        dynamic_dataset_fraction=float(np.mean(dynamic_counts > 0)),
        dynamic_node_fraction=float(np.sum(dynamic_counts) / max(np.sum(node_counts), 1.0)),
        mean_edge_density=_safe_mean(edge_densities),
        mean_in_degree=_safe_mean(mean_in_degrees),
        mean_target_ancestor_fraction=_safe_mean(target_ancestor_fractions),
        mean_abs_feature_corr=_safe_mean(mean_abs_corrs),
        mean_max_feature_corr=_safe_mean(max_abs_corrs),
        mean_feature_autocorr_lag1=_safe_mean(feature_autocorrs),
        mean_feature_spectral_entropy=_safe_mean(feature_spectral_entropies),
        mean_target_autocorr_lag1=_safe_mean(target_autocorrs),
        mean_train_test_feature_shift=_safe_mean(train_test_shifts),
        mean_class_entropy=_safe_mean(class_entropies),
        mean_class_imbalance=_safe_mean(class_imbalances),
        mean_num_classes=_safe_mean(num_classes),
        warnings=_build_warnings(datasets),
    )
    return report


def evaluate_quality_gate(
    report: PriorStatisticsReport,
    thresholds: PriorQualityThresholds | None = None,
) -> PriorQualityGate:
    thresholds = thresholds or PriorQualityThresholds()
    failures: list[str] = []

    if len(report.graph_layout_counts) < thresholds.min_graph_layouts:
        failures.append(
            f"graph layout coverage too low: {len(report.graph_layout_counts)} < {thresholds.min_graph_layouts}"
        )
    if len(report.source_kind_counts) < thresholds.min_source_kinds:
        failures.append(
            f"source kind coverage too low: {len(report.source_kind_counts)} < {thresholds.min_source_kinds}"
        )
    if len(report.aggregation_kind_counts) < thresholds.min_aggregation_kinds:
        failures.append(
            f"aggregation coverage too low: {len(report.aggregation_kind_counts)} < {thresholds.min_aggregation_kinds}"
        )
    if len(report.mechanism_kind_counts) < thresholds.min_mechanism_kinds:
        failures.append(
            f"mechanism coverage too low: {len(report.mechanism_kind_counts)} < {thresholds.min_mechanism_kinds}"
        )
    if report.dynamic_dataset_fraction < thresholds.min_dynamic_dataset_fraction:
        failures.append(
            f"dynamic dataset fraction too low: {report.dynamic_dataset_fraction:.3f} < "
            f"{thresholds.min_dynamic_dataset_fraction:.3f}"
        )
    if report.dynamic_node_fraction < thresholds.min_dynamic_node_fraction:
        failures.append(
            f"dynamic node fraction too low: {report.dynamic_node_fraction:.3f} < "
            f"{thresholds.min_dynamic_node_fraction:.3f}"
        )
    if report.mean_abs_feature_corr < thresholds.min_mean_abs_feature_corr:
        failures.append(
            f"feature correlation too weak: {report.mean_abs_feature_corr:.3f} < "
            f"{thresholds.min_mean_abs_feature_corr:.3f}"
        )
    if report.mean_abs_feature_corr > thresholds.max_mean_abs_feature_corr:
        failures.append(
            f"feature correlation too strong: {report.mean_abs_feature_corr:.3f} > "
            f"{thresholds.max_mean_abs_feature_corr:.3f}"
        )
    if (
        report.temporal_fraction > 0.0
        and abs(report.mean_feature_autocorr_lag1)
        < thresholds.min_mean_feature_autocorr_lag1_when_temporal
    ):
        failures.append(
            f"temporal autocorrelation too weak: {report.mean_feature_autocorr_lag1:.3f}"
        )
    if report.mean_feature_spectral_entropy < thresholds.min_mean_feature_spectral_entropy:
        failures.append(
            f"feature spectral entropy too low: {report.mean_feature_spectral_entropy:.3f} < "
            f"{thresholds.min_mean_feature_spectral_entropy:.3f}"
        )
    if (
        report.ood_fraction > 0.0
        and report.mean_train_test_feature_shift
        < thresholds.min_mean_train_test_feature_shift_when_ood
    ):
        failures.append(
            f"train/test feature shift too weak: {report.mean_train_test_feature_shift:.3f}"
        )
    if (
        report.task_counts.get("classification", 0) > 0
        and report.mean_class_entropy < thresholds.min_mean_class_entropy
    ):
        failures.append(
            f"class entropy too low: {report.mean_class_entropy:.3f} < {thresholds.min_mean_class_entropy:.3f}"
        )
    if (
        report.task_counts.get("classification", 0) > 0
        and report.mean_class_imbalance > thresholds.max_mean_class_imbalance
    ):
        failures.append(
            f"class imbalance too high: {report.mean_class_imbalance:.3f} > {thresholds.max_mean_class_imbalance:.3f}"
        )
    failures.extend(report.warnings)
    return PriorQualityGate(passed=len(failures) == 0, failures=tuple(failures))


def _graph_metrics(dataset: SyntheticDataset) -> tuple[float, float, float]:
    num_nodes = max(dataset.meta.target_node + 1, 1)
    edge_count = len(dataset.meta.dag_edges)
    max_edges = max(num_nodes * (num_nodes - 1) / 2.0, 1.0)
    in_degrees = np.zeros(num_nodes, dtype=np.float64)
    reverse_edges: dict[int, list[int]] = {}
    for parent, child in dataset.meta.dag_edges:
        if 0 <= child < num_nodes:
            in_degrees[child] += 1.0
        reverse_edges.setdefault(child, []).append(parent)
    ancestors = _target_ancestors(dataset.meta.target_node, reverse_edges)
    return (
        float(edge_count / max_edges),
        float(np.mean(in_degrees)),
        float(len(ancestors) / max(num_nodes - 1, 1)),
    )


def _target_ancestors(target_node: int, reverse_edges: dict[int, list[int]]) -> set[int]:
    ancestors: set[int] = set()
    stack = list(reverse_edges.get(target_node, []))
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        stack.extend(reverse_edges.get(node, []))
    return ancestors


def _feature_corr_metrics(x: NDArray[np.float32]) -> tuple[float, float]:
    if x.shape[1] < 2:
        return 0.0, 0.0
    corr = np.corrcoef(_fill_nan_for_statistics(x.astype(np.float64)), rowvar=False)
    if corr.ndim != 2:
        return 0.0, 0.0
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
    if len(upper) == 0:
        return 0.0, 0.0
    return float(np.mean(upper)), float(np.max(upper))


def _mean_feature_autocorr(x: NDArray[np.float32]) -> float:
    if x.shape[0] < 3:
        return 0.0
    filled = _fill_nan_for_statistics(x.astype(np.float64))
    return _safe_mean(
        [_autocorr(filled[:, col].astype(np.float64)) for col in range(filled.shape[1])]
    )


def _mean_feature_spectral_entropy(x: NDArray[np.float32]) -> float:
    if x.shape[0] < 4:
        return 0.0
    filled = _fill_nan_for_statistics(x.astype(np.float64))
    return _safe_mean(
        [_spectral_entropy(filled[:, col].astype(np.float64)) for col in range(filled.shape[1])]
    )


def _spectral_entropy(values: NDArray[np.float64]) -> float:
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if len(spectrum) <= 1:
        return 0.0
    spectrum = spectrum[1:]
    total = float(np.sum(spectrum))
    if total <= 1e-12 or not np.isfinite(total):
        return 0.0
    probs = spectrum / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)) / np.log(len(spectrum)))


def _autocorr(values: NDArray[np.float64]) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _train_test_feature_shift(x: NDArray[np.float32], train_size: int) -> float:
    if train_size <= 1 or train_size >= x.shape[0] - 1:
        return 0.0
    train = x[:train_size].astype(np.float64)
    test = x[train_size:].astype(np.float64)
    train_mean = np.nanmean(train, axis=0)
    test_mean = np.nanmean(test, axis=0)
    train_std = np.nanstd(train, axis=0) + 1e-6
    shift = np.abs(test_mean - train_mean) / train_std
    return float(np.mean(shift[np.isfinite(shift)])) if bool(np.any(np.isfinite(shift))) else 0.0


def _class_metrics(labels: NDArray[np.int64], num_classes: int | None) -> tuple[float, float]:
    if num_classes is None or num_classes < 2:
        return 0.0, 1.0
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    probs = counts / max(float(np.sum(counts)), 1.0)
    nonzero = probs[probs > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)) / np.log(num_classes))
    imbalance = float(np.max(probs))
    return entropy, imbalance


def _safe_mean(values: Sequence[float]) -> float:
    if len(values) == 0:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return 0.0
    return float(np.mean(array[np.isfinite(array)])) if np.any(np.isfinite(array)) else 0.0


def _fill_nan_for_statistics(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values = values.astype(np.float64, copy=True)
    for col in range(values.shape[1]):
        column = values[:, col]
        finite = np.isfinite(column)
        if not bool(np.any(finite)):
            values[:, col] = 0.0
            continue
        fill = float(np.median(column[finite]))
        column[~finite] = fill
        values[:, col] = column
    return values


def _build_warnings(datasets: Sequence[SyntheticDataset]) -> tuple[str, ...]:
    warnings: list[str] = []
    if any(np.isinf(dataset.x).any() for dataset in datasets):
        warnings.append("infinite feature values detected")
    if any(not np.isfinite(dataset.y.astype(np.float64)).all() for dataset in datasets):
        warnings.append("non-finite target values detected")
    if any(dataset.meta.num_features <= 0 for dataset in datasets):
        warnings.append("empty feature table detected")
    return tuple(warnings)
