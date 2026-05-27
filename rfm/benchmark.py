from __future__ import annotations

import importlib
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rfm.config import PriorConfig
from rfm.prior import make_prior_generator
from rfm.statistics import (
    PriorQualityGate,
    PriorStatisticsReport,
    evaluate_quality_gate,
    summarize_datasets,
)
from rfm.types import DatasetMeta, FeatureType, PriorType, SyntheticDataset, TaskType

ProbeKind = Literal["linear", "rff"]
BenchmarkSourceType = Literal["synthetic", "reference", "real", "file"]

NUMERIC_STAT_KEYS: tuple[str, ...] = (
    "mean_rows",
    "mean_features",
    "categorical_feature_fraction",
    "temporal_fraction",
    "ood_fraction",
    "dynamic_dataset_fraction",
    "dynamic_node_fraction",
    "mean_edge_density",
    "mean_in_degree",
    "mean_target_ancestor_fraction",
    "mean_abs_feature_corr",
    "mean_max_feature_corr",
    "mean_feature_autocorr_lag1",
    "mean_feature_spectral_entropy",
    "mean_target_autocorr_lag1",
    "mean_train_test_feature_shift",
    "mean_class_entropy",
    "mean_class_imbalance",
    "mean_num_classes",
)


class BenchmarkSourceError(RuntimeError):
    """Raised when an external benchmark source cannot be loaded as requested."""


@dataclass(frozen=True)
class ProbeReport:
    probe: str
    dataset_count: int
    evaluated_count: int
    task_counts: dict[str, int]
    classification_count: int
    classification_accuracy_mean: float | None
    classification_balanced_accuracy_mean: float | None
    regression_count: int
    regression_rmse_mean: float | None
    regression_mae_mean: float | None
    regression_r2_mean: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceBenchmarkReport:
    name: str
    source_type: BenchmarkSourceType
    dataset_count: int
    generation_seconds: float
    statistics: PriorStatisticsReport
    quality_gate: PriorQualityGate
    probes: dict[str, ProbeReport]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["statistics"] = self.statistics.to_dict()
        payload["quality_gate"] = self.quality_gate.to_dict()
        payload["probes"] = {name: report.to_dict() for name, report in self.probes.items()}
        return payload


def make_rfm_datasets(
    prior: Literal["scm", "tree", "gp", "bag"],
    num_datasets: int,
    seed: int,
    min_rows: int,
    max_rows: int,
    min_features: int,
    max_features: int,
    max_classes: int,
    device: str = "cpu",
) -> list[SyntheticDataset]:
    config = PriorConfig(
        prior=prior,
        min_rows=min(min_rows, max_rows),
        max_rows=max_rows,
        min_features=min(min_features, max_features),
        max_features=max_features,
        max_classes=max_classes,
        device=device,
        seed=seed,
    )
    return make_prior_generator(config).sample_batch(num_datasets)


def make_tabicl_datasets(
    num_datasets: int,
    seed: int,
    tabicl_path: Path,
    min_rows: int,
    max_rows: int,
    min_features: int,
    max_features: int,
    max_classes: int,
    batch_size: int = 8,
    prior_type: str = "mix_scm",
    device: str = "cpu",
) -> list[SyntheticDataset]:
    _seed_global_generators(seed)
    _prepend_import_path(tabicl_path)
    try:
        module = importlib.import_module("tabicl.prior._dataset")
    except (ImportError, AttributeError) as exc:
        raise BenchmarkSourceError(
            f"could not import TabICL prior from {tabicl_path}: {exc}"
        ) from exc

    prior_dataset_cls = getattr(module, "PriorDataset")
    source = prior_dataset_cls(
        batch_size=min(batch_size, num_datasets),
        batch_size_per_gp=min(4, batch_size, num_datasets),
        min_features=min_features,
        max_features=max_features,
        max_classes=max_classes,
        min_seq_len=min_rows,
        max_seq_len=max_rows + 1,
        log_seq_len=False,
        seq_len_per_gp=False,
        min_train_size=0.5,
        max_train_size=0.85,
        prior_type=prior_type,
        n_jobs=1,
        device=device,
    )

    datasets: list[SyntheticDataset] = []
    while len(datasets) < num_datasets:
        batch_count = min(batch_size, num_datasets - len(datasets))
        x_tensor, y_tensor, d_tensor, seq_lens_tensor, train_sizes_tensor = source.get_batch(
            batch_count
        )
        x_batch = _tensor_to_numpy(x_tensor)
        y_batch = _tensor_to_numpy(y_tensor)
        feature_counts = _tensor_to_numpy(d_tensor).astype(np.int64)
        seq_lens = _tensor_to_numpy(seq_lens_tensor).astype(np.int64)
        train_sizes = _tensor_to_numpy(train_sizes_tensor).astype(np.int64)
        for idx in range(batch_count):
            rows = int(seq_lens[idx])
            features = int(feature_counts[idx])
            x = x_batch[idx, :rows, :features]
            datasets.append(
                make_dataset_from_arrays(
                    x=x,
                    y=y_batch[idx, :rows],
                    source_name="tabicl",
                    prior_type="tabicl",
                    train_size=int(train_sizes[idx]),
                    task_type="classification",
                    categorical_features=_infer_low_cardinality_columns(x),
                    shuffle=False,
                )
            )
    return datasets


def make_tabpfn_v1_datasets(
    num_datasets: int,
    seed: int,
    tabpfn_prior_path: Path,
    rows: int,
    features: int,
    max_classes: int,
    batch_size: int = 8,
    prior_type: str = "prior_bag",
    device: str = "cpu",
) -> list[SyntheticDataset]:
    _seed_global_generators(seed)
    _prepend_import_path(tabpfn_prior_path)
    try:
        module = importlib.import_module("tabpfn_prior")
        build_tabpfn_prior = getattr(module, "build_tabpfn_prior")
    except (ImportError, AttributeError) as exc:
        raise BenchmarkSourceError(
            f"could not import TabPFN v1 prior from {tabpfn_prior_path}: {exc}"
        ) from exc

    steps = int(math.ceil(num_datasets / max(batch_size, 1)))
    loader = build_tabpfn_prior(
        prior_type=prior_type,
        num_steps=steps,
        batch_size=min(batch_size, num_datasets),
        num_datapoints_max=rows,
        num_features=features,
        max_num_classes=max_classes,
        device=device,
    )

    datasets: list[SyntheticDataset] = []
    for batch in loader:
        x_batch = _tensor_to_numpy(batch["x"])
        y_batch = _tensor_to_numpy(batch["y"])
        train_size = int(batch["single_eval_pos"])
        for idx in range(x_batch.shape[0]):
            datasets.append(
                make_dataset_from_arrays(
                    x=x_batch[idx],
                    y=y_batch[idx],
                    source_name="tabpfn_v1",
                    prior_type="tabpfn_v1",
                    train_size=train_size,
                    task_type="classification",
                    categorical_features=(),
                    shuffle=False,
                )
            )
            if len(datasets) >= num_datasets:
                return datasets
    return datasets


def make_openml_dataset(
    data_id: int,
    seed: int,
    max_rows: int | None = None,
    train_fraction: float = 0.7,
) -> SyntheticDataset:
    try:
        datasets_module = importlib.import_module("sklearn.datasets")
        fetch_openml = getattr(datasets_module, "fetch_openml")
    except (ImportError, AttributeError) as exc:
        raise BenchmarkSourceError(f"could not import sklearn OpenML loader: {exc}") from exc

    try:
        bunch = fetch_openml(data_id=data_id, as_frame=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BenchmarkSourceError(f"could not fetch OpenML data_id={data_id}: {exc}") from exc

    x = bunch.data
    y = bunch.target
    if max_rows is not None and len(y) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(y), size=max_rows, replace=False))
        x = x.iloc[indices] if hasattr(x, "iloc") else np.asarray(x)[indices]
        y = y.iloc[indices] if hasattr(y, "iloc") else np.asarray(y)[indices]

    return make_dataset_from_arrays(
        x=x,
        y=y,
        source_name=f"openml_{data_id}",
        prior_type="real",
        train_size=max(1, min(int(round(len(y) * train_fraction)), len(y) - 1)),
        shuffle=True,
        seed=seed,
    )


def load_npz_datasets(
    path: Path, seed: int, max_datasets: int | None = None
) -> list[SyntheticDataset]:
    paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    datasets: list[SyntheticDataset] = []
    for npz_path in paths:
        with np.load(npz_path, allow_pickle=False) as data:
            if "x" not in data or "y" not in data:
                raise BenchmarkSourceError(f"{npz_path} must contain x and y arrays")
            x = data["x"]
            y = data["y"]
            train_sizes = data["train_size"] if "train_size" in data else None
            task_type = (
                str(data["task_type"].item())
                if "task_type" in data and data["task_type"].ndim == 0
                else None
            )
            categorical_features = _npz_categorical_features(data)
            if x.ndim == 2:
                datasets.append(
                    make_dataset_from_arrays(
                        x=x,
                        y=y,
                        source_name=npz_path.stem,
                        prior_type="file",
                        train_size=_single_train_size(train_sizes, x.shape[0]),
                        task_type=_optional_task_type(task_type),
                        categorical_features=categorical_features,
                        shuffle=True,
                        seed=seed + len(datasets),
                    )
                )
            elif x.ndim == 3:
                for idx in range(x.shape[0]):
                    datasets.append(
                        make_dataset_from_arrays(
                            x=x[idx],
                            y=y[idx],
                            source_name=f"{npz_path.stem}_{idx}",
                            prior_type="file",
                            train_size=_single_train_size(
                                train_sizes[idx] if train_sizes is not None else None, x.shape[1]
                            ),
                            task_type=_optional_task_type(task_type),
                            categorical_features=categorical_features,
                            shuffle=True,
                            seed=seed + len(datasets),
                        )
                    )
                    if max_datasets is not None and len(datasets) >= max_datasets:
                        return datasets
            else:
                raise BenchmarkSourceError(f"{npz_path} x must be 2D or 3D")
        if max_datasets is not None and len(datasets) >= max_datasets:
            return datasets
    return datasets


def load_csv_dataset(
    path: Path,
    target: str,
    seed: int,
    task_type: TaskType | None = None,
    max_rows: int | None = None,
    train_fraction: float = 0.7,
) -> SyntheticDataset:
    try:
        pandas = importlib.import_module("pandas")
    except ImportError as exc:
        raise BenchmarkSourceError(f"could not import pandas for CSV loading: {exc}") from exc

    frame = pandas.read_csv(path)
    if target not in frame.columns:
        raise BenchmarkSourceError(f"target column {target!r} not found in {path}")
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).sort_index()
    y = frame[target]
    x = frame.drop(columns=[target])
    return make_dataset_from_arrays(
        x=x,
        y=y,
        source_name=path.stem,
        prior_type="real",
        train_size=max(1, min(int(round(len(frame) * train_fraction)), len(frame) - 1)),
        task_type=task_type,
        shuffle=True,
        seed=seed,
    )


def make_dataset_from_arrays(
    x: Any,
    y: Any,
    source_name: str,
    prior_type: PriorType,
    train_size: int | None = None,
    task_type: TaskType | None = None,
    categorical_features: Sequence[int] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> SyntheticDataset:
    x_array, inferred_categorical = _coerce_features(x, categorical_features)
    y_array, inferred_task_type, num_classes = _coerce_target(y, task_type)
    if x_array.shape[0] != y_array.shape[0]:
        raise BenchmarkSourceError(
            f"{source_name} has mismatched x/y rows: {x_array.shape[0]} != {y_array.shape[0]}"
        )
    if x_array.shape[0] < 4:
        raise BenchmarkSourceError(f"{source_name} must contain at least 4 rows")

    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(x_array.shape[0])
        x_array = x_array[order]
        y_array = y_array[order]

    train_size = _valid_train_size(train_size, x_array.shape[0])
    feature_types: tuple[FeatureType, ...] = tuple(
        "categorical" if idx in inferred_categorical else "numerical"
        for idx in range(x_array.shape[1])
    )
    categorical_tuple = tuple(sorted(inferred_categorical))

    return SyntheticDataset(
        x=x_array.astype(np.float32),
        y=y_array.astype(np.int64)
        if inferred_task_type == "classification"
        else y_array.astype(np.float32),
        feature_types=feature_types,
        meta=DatasetMeta(
            prior_type=prior_type,
            num_rows=x_array.shape[0],
            num_features=x_array.shape[1],
            task_type=inferred_task_type,
            train_size=train_size,
            num_classes=num_classes,
            temporal=False,
            ood=False,
            feature_nodes=tuple(range(x_array.shape[1])),
            target_node=x_array.shape[1],
            dag_edges=(),
            categorical_features=categorical_tuple,
            graph_layout=None,
        ),
    )


def benchmark_source(
    name: str,
    source_type: BenchmarkSourceType,
    datasets: Sequence[SyntheticDataset],
    probe_kinds: Sequence[ProbeKind],
    seed: int,
    generation_seconds: float,
    ridge_alpha: float = 1.0,
    rff_features: int = 128,
) -> SourceBenchmarkReport:
    report = summarize_datasets(datasets)
    gate = evaluate_quality_gate(report)
    probes = {
        probe: evaluate_probe(
            datasets=datasets,
            probe=probe,
            seed=seed,
            ridge_alpha=ridge_alpha,
            rff_features=rff_features,
        )
        for probe in probe_kinds
    }
    warnings = _source_warnings(source_type, gate)
    return SourceBenchmarkReport(
        name=name,
        source_type=source_type,
        dataset_count=len(datasets),
        generation_seconds=generation_seconds,
        statistics=report,
        quality_gate=gate,
        probes=probes,
        warnings=warnings,
    )


def timed_dataset_call(factory: Any) -> tuple[list[SyntheticDataset], float]:
    start = time.perf_counter()
    datasets = factory()
    return datasets, time.perf_counter() - start


def evaluate_probe(
    datasets: Sequence[SyntheticDataset],
    probe: ProbeKind,
    seed: int = 0,
    ridge_alpha: float = 1.0,
    rff_features: int = 128,
) -> ProbeReport:
    task_counts = Counter(dataset.meta.task_type for dataset in datasets)
    accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    rmses: list[float] = []
    maes: list[float] = []
    r2s: list[float] = []
    warnings: list[str] = []
    evaluated = 0

    for idx, dataset in enumerate(datasets):
        rng = np.random.default_rng(seed + idx * 104729)
        try:
            metrics = _evaluate_probe_dataset(
                dataset=dataset,
                probe=probe,
                rng=rng,
                ridge_alpha=ridge_alpha,
                rff_features=rff_features,
            )
        except ValueError as exc:
            warnings.append(f"dataset {idx}: {exc}")
            continue

        evaluated += 1
        if dataset.meta.task_type == "classification":
            accuracies.append(metrics["accuracy"])
            balanced_accuracies.append(metrics["balanced_accuracy"])
        else:
            rmses.append(metrics["rmse"])
            maes.append(metrics["mae"])
            r2s.append(metrics["r2"])

    return ProbeReport(
        probe=probe,
        dataset_count=len(datasets),
        evaluated_count=evaluated,
        task_counts=dict(task_counts),
        classification_count=len(accuracies),
        classification_accuracy_mean=_optional_mean(accuracies),
        classification_balanced_accuracy_mean=_optional_mean(balanced_accuracies),
        regression_count=len(rmses),
        regression_rmse_mean=_optional_mean(rmses),
        regression_mae_mean=_optional_mean(maes),
        regression_r2_mean=_optional_mean(r2s),
        warnings=tuple(warnings[:20]),
    )


def compare_statistics(
    reports: Mapping[str, SourceBenchmarkReport],
    reference_name: str,
) -> dict[str, object]:
    if reference_name not in reports:
        raise BenchmarkSourceError(f"reference source {reference_name!r} not found")

    reference = reports[reference_name].statistics
    comparison: dict[str, object] = {}
    for name, source_report in reports.items():
        report = source_report.statistics
        metric_diffs: dict[str, dict[str, float]] = {}
        relative_diffs: list[float] = []
        for key in NUMERIC_STAT_KEYS:
            left = float(getattr(report, key))
            right = float(getattr(reference, key))
            absolute = abs(left - right)
            relative = absolute / max(abs(left), abs(right), 1e-6)
            metric_diffs[key] = {"absolute": absolute, "relative": relative}
            if math.isfinite(relative):
                relative_diffs.append(min(relative, 100.0))
        comparison[name] = {
            "mean_relative_stat_distance": float(np.mean(relative_diffs))
            if relative_diffs
            else 0.0,
            "metrics": metric_diffs,
        }
    return comparison


def _evaluate_probe_dataset(
    dataset: SyntheticDataset,
    probe: ProbeKind,
    rng: np.random.Generator,
    ridge_alpha: float,
    rff_features: int,
) -> dict[str, float]:
    train_size = dataset.meta.train_size
    if train_size < 2 or train_size >= dataset.x.shape[0] - 1:
        raise ValueError("invalid train/test split for probe")

    features = _prepare_probe_features(dataset)
    if features.shape[1] == 0:
        raise ValueError("no usable features")
    if probe == "rff":
        features = _append_random_fourier_features(features, train_size, rng, rff_features)

    x_train = features[:train_size]
    x_test = features[train_size:]
    if dataset.meta.task_type == "classification":
        y = dataset.y.astype(np.int64)
        y_train = y[:train_size]
        y_test = y[train_size:]
        classes = np.unique(y_train)
        if len(classes) < 2:
            raise ValueError("classification train split has fewer than two classes")
        pred = _ridge_classify(x_train, y_train, x_test, classes, ridge_alpha)
        return {
            "accuracy": _accuracy(y_test, pred),
            "balanced_accuracy": _balanced_accuracy(y_test, pred),
        }

    y_float = dataset.y.astype(np.float64)
    pred = _ridge_regress(x_train, y_float[:train_size], x_test, ridge_alpha)
    return _regression_metrics(y_float[train_size:], pred)


def _prepare_probe_features(
    dataset: SyntheticDataset, max_one_hot_categories: int = 32
) -> NDArray[np.float64]:
    x = dataset.x.astype(np.float64)
    train_size = dataset.meta.train_size
    categorical = set(dataset.meta.categorical_features)
    pieces: list[NDArray[np.float64]] = []
    for col in range(x.shape[1]):
        column = x[:, col]
        train_column = column[:train_size]
        if col in categorical:
            finite_train = train_column[np.isfinite(train_column)]
            if len(finite_train) == 0:
                continue
            categories = np.unique(finite_train.astype(np.int64))
            if 1 < len(categories) <= max_one_hot_categories:
                encoded = np.zeros((x.shape[0], len(categories)), dtype=np.float64)
                finite_column = np.isfinite(column)
                int_column = np.zeros_like(column, dtype=np.int64)
                int_column[finite_column] = column[finite_column].astype(np.int64)
                for cat_idx, category in enumerate(categories):
                    encoded[:, cat_idx] = (finite_column & (int_column == category)).astype(
                        np.float64
                    )
                pieces.append(encoded)
            else:
                pieces.append(_standardize_for_probe(column, train_size)[:, None])
        else:
            pieces.append(_standardize_for_probe(column, train_size)[:, None])
    if not pieces:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    return np.concatenate(pieces, axis=1)


def _standardize_for_probe(values: NDArray[np.float64], train_size: int) -> NDArray[np.float64]:
    values = values.astype(np.float64, copy=True)
    train = values[:train_size]
    finite = np.isfinite(train)
    fill = float(np.median(train[finite])) if bool(np.any(finite)) else 0.0
    values[~np.isfinite(values)] = fill
    train = values[:train_size]
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def _append_random_fourier_features(
    features: NDArray[np.float64],
    train_size: int,
    rng: np.random.Generator,
    rff_features: int,
) -> NDArray[np.float64]:
    if rff_features <= 0 or features.shape[1] == 0:
        return features
    train = features[:train_size]
    scale = np.std(train, axis=0) + 1e-6
    normalized = features / scale
    weights = rng.normal(0.0, 1.0, size=(normalized.shape[1], rff_features))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=rff_features)
    fourier = np.sqrt(2.0 / rff_features) * np.cos(normalized @ weights + phases)
    return np.concatenate([features, fourier], axis=1)


def _ridge_classify(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    classes: NDArray[np.int64],
    alpha: float,
) -> NDArray[np.int64]:
    targets = (y_train[:, None] == classes[None, :]).astype(np.float64)
    weights = _ridge_weights(_with_bias(x_train), targets, alpha)
    scores = _with_bias(x_test) @ weights
    return classes[np.argmax(scores, axis=1)].astype(np.int64)


def _ridge_regress(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    weights = _ridge_weights(_with_bias(x_train), y_train[:, None], alpha)
    return (_with_bias(x_test) @ weights).ravel()


def _ridge_weights(
    x_train: NDArray[np.float64],
    targets: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    gram = x_train.T @ x_train
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    rhs = x_train.T @ targets
    try:
        return np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + regularizer, rhs, rcond=None)[0]


def _with_bias(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _accuracy(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _balanced_accuracy(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    recalls: list[float] = []
    for label in np.unique(y_true):
        mask = y_true == label
        if bool(np.any(mask)):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def _regression_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    errors = y_pred - y_true
    mse = float(np.mean(errors * errors))
    mae = float(np.mean(np.abs(errors)))
    variance = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    residual = float(np.sum(errors * errors))
    r2 = 1.0 - residual / variance if variance > 1e-12 else 0.0
    return {"rmse": float(np.sqrt(mse)), "mae": mae, "r2": float(r2)}


def _coerce_features(
    x: Any,
    categorical_features: Sequence[int] | None,
) -> tuple[NDArray[np.float32], set[int]]:
    if _looks_like_dataframe(x):
        return _coerce_dataframe_features(x, categorical_features)
    array = np.asarray(x)
    if array.ndim != 2:
        raise BenchmarkSourceError(f"x must be 2D, got shape {array.shape}")
    return _coerce_2d_array_features(array, categorical_features)


def _coerce_dataframe_features(
    frame: Any,
    categorical_features: Sequence[int] | None,
) -> tuple[NDArray[np.float32], set[int]]:
    forced = set(categorical_features or ())
    columns: list[NDArray[np.float64]] = []
    inferred_categorical: set[int] = set()
    for idx, column_name in enumerate(frame.columns):
        series = frame[column_name]
        if idx in forced or not _is_numeric_dtype(series):
            codes = _encode_categories(series)
            columns.append(codes)
            inferred_categorical.add(idx)
        else:
            columns.append(
                _clean_numeric_column(np.asarray(series, dtype=np.float64), standardize=True)
            )
    if not columns:
        raise BenchmarkSourceError("feature table has no columns")
    return np.column_stack(columns).astype(np.float32), inferred_categorical


def _infer_low_cardinality_columns(
    x: NDArray[Any],
    max_unique: int = 32,
) -> tuple[int, ...]:
    array = np.asarray(x)
    if array.ndim != 2:
        raise BenchmarkSourceError(f"tabular feature array must be 2D, got shape {array.shape}")
    categorical: list[int] = []
    for idx in range(array.shape[1]):
        values = np.asarray(array[:, idx], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        unique_count = len(np.unique(finite))
        threshold = min(max_unique, max(2, int(0.1 * len(finite))))
        if 1 < unique_count <= threshold:
            categorical.append(idx)
    return tuple(categorical)


def _coerce_2d_array_features(
    array: NDArray[Any],
    categorical_features: Sequence[int] | None,
) -> tuple[NDArray[np.float32], set[int]]:
    forced = set(categorical_features or ())
    columns: list[NDArray[np.float64]] = []
    inferred_categorical: set[int] = set(forced)
    for idx in range(array.shape[1]):
        column = array[:, idx]
        if idx in forced or not np.issubdtype(column.dtype, np.number):
            columns.append(_encode_categories(column))
            inferred_categorical.add(idx)
        else:
            values = np.asarray(column, dtype=np.float64)
            if _looks_categorical_numeric(values):
                inferred_categorical.add(idx)
                columns.append(_clean_numeric_column(values, standardize=False))
            else:
                columns.append(_clean_numeric_column(values, standardize=True))
    return np.column_stack(columns).astype(np.float32), inferred_categorical


def _coerce_target(
    y: Any, task_type: TaskType | None
) -> tuple[NDArray[np.float64], TaskType, int | None]:
    array = np.asarray(y)
    if array.ndim != 1:
        array = array.reshape(-1)
    if task_type == "classification" or (
        task_type is None and not np.issubdtype(array.dtype, np.number)
    ):
        encoded = _encode_categories(array).astype(np.int64)
        num_classes = int(len(np.unique(encoded)))
        if num_classes < 2:
            raise BenchmarkSourceError("classification target must have at least two classes")
        return encoded.astype(np.float64), "classification", num_classes

    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        raise BenchmarkSourceError("target contains no finite values")
    fill = float(np.median(values[finite]))
    values[~finite] = fill
    if task_type is None and _looks_categorical_numeric(values):
        encoded = _encode_categories(values).astype(np.int64)
        num_classes = int(len(np.unique(encoded)))
        if num_classes >= 2:
            return encoded.astype(np.float64), "classification", num_classes
    return _clean_numeric_column(values, standardize=True), "regression", None


def _encode_categories(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values)
    labels: list[str] = []
    for value in array:
        if _is_missing_value(value):
            labels.append("__missing__")
        else:
            labels.append(str(value))
    uniques = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.asarray([uniques[label] for label in labels], dtype=np.float64)


def _clean_numeric_column(values: NDArray[np.float64], standardize: bool) -> NDArray[np.float64]:
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    fill = float(np.median(values[finite])) if bool(np.any(finite)) else 0.0
    values[~finite] = fill
    if not standardize:
        return values
    std = float(np.std(values))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _looks_categorical_numeric(values: NDArray[np.float64]) -> bool:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return False
    unique = np.unique(finite)
    return bool(
        len(unique) <= min(20, max(2, int(0.1 * len(finite))))
        and np.allclose(unique, np.round(unique))
    )


def _looks_like_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "__getitem__")


def _is_numeric_dtype(series: Any) -> bool:
    try:
        pandas = importlib.import_module("pandas")
        return bool(pandas.api.types.is_numeric_dtype(series))
    except ImportError:
        return np.issubdtype(np.asarray(series).dtype, np.number)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _valid_train_size(train_size: int | None, num_rows: int) -> int:
    if train_size is None:
        train_size = int(round(0.7 * num_rows))
    return max(1, min(int(train_size), num_rows - 1))


def _single_train_size(value: Any, num_rows: int) -> int:
    if value is None:
        return _valid_train_size(None, num_rows)
    array = np.asarray(value)
    return _valid_train_size(int(array.item()) if array.ndim == 0 else int(array[0]), num_rows)


def _optional_task_type(value: str | None) -> TaskType | None:
    if value in ("classification", "regression"):
        return value
    return None


def _npz_categorical_features(data: Mapping[str, NDArray[Any]]) -> tuple[int, ...]:
    if "categorical_features" in data:
        return tuple(
            int(idx) for idx in np.asarray(data["categorical_features"]).reshape(-1).tolist()
        )
    if "feature_is_categorical" in data:
        mask = np.asarray(data["feature_is_categorical"]).astype(bool).reshape(-1)
        return tuple(int(idx) for idx in np.flatnonzero(mask).tolist())
    return ()


def _optional_mean(values: Sequence[float]) -> float | None:
    if len(values) == 0:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return None
    return float(np.mean(finite))


def _tensor_to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _seed_global_generators(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _prepend_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _source_warnings(source_type: BenchmarkSourceType, gate: PriorQualityGate) -> tuple[str, ...]:
    if source_type in ("reference", "real", "file") and not gate.passed:
        return (
            "structure coverage gate is designed for RFM priors; failures on non-RFM sources are descriptive",
        )
    return ()
