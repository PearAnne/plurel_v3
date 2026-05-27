from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from rfm.config import PriorConfig, SourceKind
from rfm.scm import SCMGenerator, SCMSpec
from rfm.types import (
    DatasetMeta,
    FeatureType,
    PriorType,
    SyntheticBatch,
    SyntheticDataset,
    TaskType,
)


@dataclass(frozen=True)
class TreeNode:
    feature: int | None
    threshold: float
    value: float | None
    left: TreeNode | None = None
    right: TreeNode | None = None


class PostProcessor:
    def __init__(self, config: PriorConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def postprocess_features(
        self,
        x: NDArray[np.float32],
        feature_nodes: tuple[int, ...],
        feature_components: tuple[int, ...],
        train_size: int,
        ood: bool,
    ) -> tuple[
        NDArray[np.float32],
        tuple[FeatureType, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        x = x.astype(np.float64, copy=True)
        feature_types: list[FeatureType] = ["numerical"] * x.shape[1]

        for col in range(x.shape[1]):
            if self.rng.random() < self.config.categorical_feature_prob:
                categories = self._sample_category_count(x.shape[0])
                categories = min(categories, max(2, x.shape[0] // 2))
                x[:, col] = _quantile_bins(x[:, col], categories).astype(np.float64)
                feature_types[col] = "categorical"
            else:
                x[:, col] = self._maybe_warp_continuous(x[:, col])
                x[:, col] = _standardize(x[:, col], train_size=train_size if ood else None).astype(
                    np.float64
                )

        order = self.rng.permutation(x.shape[1])
        x = x[:, order]
        feature_types = [feature_types[int(i)] for i in order]
        ordered_nodes = [feature_nodes[int(i)] for i in order]
        ordered_components = [feature_components[int(i)] for i in order]

        x = self._add_missingness(x=x, feature_types=feature_types, train_size=train_size)
        x = self._add_outliers(x=x, feature_types=feature_types)

        keep_mask = np.array(
            [_is_informative_feature(x[:, col], feature_types[col]) for col in range(x.shape[1])]
        )
        if not bool(np.any(keep_mask)):
            raise ValueError("all generated features were constant")

        x = x[:, keep_mask]
        feature_types = [
            feature_type for feature_type, keep in zip(feature_types, keep_mask) if keep
        ]
        ordered_nodes = [node for node, keep in zip(ordered_nodes, keep_mask) if keep]
        ordered_components = [
            component for component, keep in zip(ordered_components, keep_mask) if keep
        ]
        categorical_features = [
            idx for idx, feature_type in enumerate(feature_types) if feature_type == "categorical"
        ]

        return (
            x.astype(np.float32),
            tuple(feature_types),
            tuple(categorical_features),
            tuple(ordered_nodes),
            tuple(ordered_components),
        )

    def _sample_category_count(self, num_rows: int) -> int:
        if self.rng.random() < self.config.high_cardinality_categorical_prob:
            low = min(self.config.max_categories + 1, self.config.max_high_cardinality_categories)
            high = self.config.max_high_cardinality_categories
            categories = int(self.rng.integers(low, high + 1))
        else:
            categories = int(self.rng.integers(2, self.config.max_categories + 1))
        return min(categories, max(2, num_rows // 2))

    def _maybe_warp_continuous(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.rng.random() >= self.config.marginal_warp_prob:
            return values.astype(np.float64, copy=True)
        kind = ("kumaraswamy", "power", "log", "exp", "rank_gauss")[int(self.rng.integers(0, 5))]
        values = values.astype(np.float64, copy=True)
        if kind == "kumaraswamy":
            uniform = _rank_to_uniform(values)
            a = float(np.exp(self.rng.uniform(np.log(0.1), np.log(10.0))))
            b = float(np.exp(self.rng.uniform(np.log(0.1), np.log(10.0))))
            return 1.0 - (1.0 - uniform**a) ** b
        if kind == "power":
            exponent = float(np.exp(self.rng.uniform(np.log(0.2), np.log(3.0))))
            return np.sign(values) * np.abs(values) ** exponent
        if kind == "log":
            return np.sign(values) * np.log1p(np.abs(values))
        if kind == "exp":
            return np.exp(np.clip(values, -5.0, 5.0))
        if kind == "rank_gauss":
            normal = NormalDist()
            return np.asarray(
                [normal.inv_cdf(float(value)) for value in _rank_to_uniform(values, eps=1e-4)]
            )
        raise ValueError(f"unknown marginal warp {kind}")

    def _add_missingness(
        self,
        x: NDArray[np.float64],
        feature_types: Sequence[FeatureType],
        train_size: int,
    ) -> NDArray[np.float64]:
        x = x.astype(np.float64, copy=True)
        if self.rng.random() < self.config.mcar_missing_prob:
            rate = min(float(self.rng.beta(0.5, 8.0)), self.config.max_missing_rate)
            x[self.rng.random(x.shape) < rate] = np.nan
        if x.shape[1] > 1 and self.rng.random() < self.config.mar_missing_prob:
            driver_col = int(self.rng.integers(0, x.shape[1]))
            driver = _fill_missing_with_train_median(x[:, driver_col], train_size)
            driver = _standardize(driver, train_size=train_size)
            probabilities = _sigmoid(driver) * min(
                float(self.rng.beta(0.8, 8.0)), self.config.max_missing_rate
            )
            target_col = int(self.rng.integers(0, x.shape[1]))
            column = x[:, target_col]
            column[self.rng.random(x.shape[0]) < probabilities] = np.nan
            x[:, target_col] = column
        if self.rng.random() < self.config.mnar_missing_prob:
            for col, feature_type in enumerate(feature_types):
                if feature_type != "numerical" or self.rng.random() >= 0.25:
                    continue
                values = _fill_missing_with_train_median(x[:, col], train_size)
                values = _standardize(values, train_size=train_size)
                rate = min(float(self.rng.beta(0.7, 10.0)), self.config.max_missing_rate)
                probabilities = _sigmoid(np.abs(values) - 1.0) * rate
                column = x[:, col]
                column[self.rng.random(x.shape[0]) < probabilities] = np.nan
                x[:, col] = column
        return x

    def _add_outliers(
        self,
        x: NDArray[np.float64],
        feature_types: Sequence[FeatureType],
    ) -> NDArray[np.float64]:
        x = x.astype(np.float64, copy=True)
        for col, feature_type in enumerate(feature_types):
            if feature_type != "numerical" or self.rng.random() >= self.config.outlier_column_prob:
                continue
            finite = np.isfinite(x[:, col])
            if not bool(np.any(finite)):
                continue
            rate = min(float(self.rng.beta(0.3, 20.0)), self.config.max_outlier_rate)
            mask = (self.rng.random(x.shape[0]) < rate) & finite
            if not bool(np.any(mask)):
                continue
            scale = float(np.exp(self.rng.uniform(np.log(6.0), np.log(30.0))))
            signs = self.rng.choice(np.array([-1.0, 1.0]), size=int(np.sum(mask)))
            column = x[:, col]
            column[mask] = column[mask] + signs * scale
            x[:, col] = column
        return x

    def make_target(
        self,
        raw_target: NDArray[np.float64],
        task_type: TaskType,
        num_rows: int,
        train_size: int,
        ordered: bool,
    ) -> tuple[NDArray[np.integer] | NDArray[np.floating], int | None]:
        if task_type == "regression":
            return _standardize(raw_target).astype(np.float32), None

        num_classes = self._sample_num_classes(num_rows=num_rows, train_size=train_size)
        labels = self._assign_classes(raw_target, num_classes)
        if ordered:
            labels = _ensure_ordered_split_classes(labels, raw_target, num_classes, train_size)
        return labels.astype(np.int64), num_classes

    def finalize_order(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.integer] | NDArray[np.floating],
        task_type: TaskType,
        train_size: int,
        ordered: bool,
    ) -> tuple[NDArray[np.float32], NDArray[np.integer] | NDArray[np.floating]]:
        if ordered:
            if task_type == "classification" and not _has_valid_classification_split(
                y.astype(np.int64), train_size
            ):
                raise ValueError(
                    "classification split does not contain matching train/test classes"
                )
            return x, y

        if task_type == "regression":
            order = self.rng.permutation(x.shape[0])
            return x[order], y[order]

        labels = y.astype(np.int64)
        for _ in range(self.config.valid_split_attempts):
            order = self.rng.permutation(x.shape[0])
            candidate_labels = labels[order]
            if _has_valid_classification_split(candidate_labels, train_size):
                return x[order], candidate_labels
        raise ValueError("could not find a valid classification split")

    def _sample_num_classes(self, num_rows: int, train_size: int) -> int:
        max_by_split = max(2, min(train_size, num_rows - train_size) // 2)
        max_classes = min(self.config.max_classes, max(2, max_by_split))
        if self.rng.random() < self.config.many_class_prob and max_classes > 10:
            return int(self.rng.integers(11, max_classes + 1))
        return int(self.rng.integers(2, min(10, max_classes) + 1))

    def _assign_classes(self, values: NDArray[np.float64], num_classes: int) -> NDArray[np.int64]:
        assignment_kind = self.config.class_assignment_kinds[
            int(self.rng.integers(0, len(self.config.class_assignment_kinds)))
        ]
        if self.rng.random() < self.config.balanced_class_prob or assignment_kind == "rank":
            labels = _quantile_bins(values, num_classes)
        elif assignment_kind == "nested":
            labels = self._nested_classes(values, num_classes)
        elif assignment_kind == "dirichlet":
            labels = self._dirichlet_classes(values, num_classes)
        elif assignment_kind == "multilabel_score":
            labels = self._multilabel_score_classes(values, num_classes)
        else:
            labels = self._value_classes(values, num_classes)
        labels = _ensure_dense_labels(labels, values, num_classes)
        unique_count = len(np.unique(labels))
        if unique_count != num_classes:
            raise ValueError(
                f"class assignment {assignment_kind!r} produced {unique_count} "
                f"classes instead of {num_classes}"
            )
        return self._maybe_permute_labels(labels, num_classes)

    def _value_classes(self, values: NDArray[np.float64], num_classes: int) -> NDArray[np.int64]:
        return _proportion_bins(
            values, self._sample_class_proportions(num_classes, min_mass=1.0 / len(values))
        )

    def _nested_classes(self, values: NDArray[np.float64], num_classes: int) -> NDArray[np.int64]:
        labels = np.zeros(values.shape[0], dtype=np.int64)
        remaining = np.arange(values.shape[0])
        for class_id in range(num_classes - 1):
            if len(remaining) == 0:
                break
            remaining_values = values[remaining]
            threshold = np.quantile(remaining_values, float(self.rng.uniform(0.25, 0.75)))
            if self.rng.random() < 0.5:
                selected = remaining_values <= threshold
            else:
                selected = remaining_values >= threshold
            selected_indices = remaining[selected]
            labels[selected_indices] = class_id
            remaining = remaining[~selected]
        labels[remaining] = num_classes - 1
        return labels

    def _dirichlet_classes(
        self, values: NDArray[np.float64], num_classes: int
    ) -> NDArray[np.int64]:
        return _proportion_bins(
            values, self._sample_class_proportions(num_classes, min_mass=1.0 / len(values))
        )

    def _sample_class_proportions(
        self, num_classes: int, min_mass: float = 0.0
    ) -> NDArray[np.float64]:
        concentration = float(np.exp(self.rng.uniform(np.log(0.05), np.log(1.5))))
        probs = self.rng.dirichlet(np.full(num_classes, concentration))
        if 0.0 < min_mass * num_classes < 0.9:
            probs = np.maximum(probs, min_mass)
        return probs / probs.sum()

    def _multilabel_score_classes(
        self, values: NDArray[np.float64], num_classes: int
    ) -> NDArray[np.int64]:
        scores = np.zeros((len(values), num_classes), dtype=np.float64)
        standardized = _standardize(values)
        for class_id in range(num_classes):
            frequency = float(np.exp(self.rng.uniform(np.log(0.2), np.log(20.0))))
            phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
            slope = float(self.rng.normal())
            bias = float(self.rng.normal(0.0, 0.5))
            scores[:, class_id] = (
                slope * standardized
                + np.sin(frequency * standardized + phase)
                + 0.3 * np.cos(0.5 * frequency * standardized - phase)
                + bias
            )
        labels = np.argmax(scores, axis=1).astype(np.int64)
        return labels

    def _maybe_permute_labels(
        self, labels: NDArray[np.int64], num_classes: int
    ) -> NDArray[np.int64]:
        if self.rng.random() > self.config.ordered_label_prob:
            if self.rng.random() < self.config.permute_label_prob:
                labels = self.rng.permutation(num_classes)[labels]
            if self.rng.random() < 0.5:
                labels = num_classes - 1 - labels
        return labels.astype(np.int64)


class BasePriorGenerator:
    prior_type: PriorType

    def __init__(
        self,
        config: PriorConfig | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("seed and rng cannot both be provided")
        self.config = config or PriorConfig()
        self.rng = rng or np.random.default_rng(self.config.seed if seed is None else seed)
        self.postprocessor = PostProcessor(self.config, self.rng)

    def sample_dataset(self) -> SyntheticDataset:
        raise NotImplementedError

    def sample_batch(self, batch_size: int) -> list[SyntheticDataset]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return [self.sample_dataset() for _ in range(batch_size)]

    def _retry_dataset(self, factory: Callable[[], SyntheticDataset]) -> SyntheticDataset:
        last_error: ValueError | None = None
        for _ in range(self.config.valid_split_attempts):
            try:
                return factory()
            except ValueError as exc:
                last_error = exc
        if last_error is None:
            raise ValueError("dataset generation failed")
        raise last_error

    def _sample_rows(self) -> int:
        if self.rng.random() < self.config.replay_small_prob:
            max_small = min(self.config.max_rows, max(self.config.min_rows, 1000))
            return int(self.rng.integers(self.config.min_rows, max_small + 1))
        if self.config.log_rows:
            log_min = np.log(self.config.min_rows)
            log_max = np.log(self.config.max_rows + 1)
            return min(self.config.max_rows, int(np.exp(self.rng.uniform(log_min, log_max))))
        return int(self.rng.integers(self.config.min_rows, self.config.max_rows + 1))

    def _sample_train_size(self, num_rows: int) -> int:
        train_fraction = float(
            self.rng.uniform(self.config.min_train_fraction, self.config.max_train_fraction)
        )
        train_size = int(round(num_rows * train_fraction))
        return min(max(train_size, 1), num_rows - 1)

    def _sample_task(self) -> TaskType:
        if self.config.task == "classification":
            return "classification"
        if self.config.task == "regression":
            return "regression"
        return (
            "classification"
            if self.rng.random() < self.config.classification_prob
            else "regression"
        )

    def _sample_num_features(self) -> int:
        return int(self.rng.integers(self.config.min_features, self.config.max_features + 1))

    def _sample_node_dim(self) -> int:
        probabilities = np.asarray(self.config.node_dim_probs, dtype=np.float64)
        probabilities = probabilities / float(np.sum(probabilities))
        return int(
            self.rng.choice(np.asarray(self.config.node_dims, dtype=np.int64), p=probabilities)
        )

    def _sample_flags(self) -> tuple[bool, bool]:
        temporal = bool(self.rng.random() < self.config.temporal_prob)
        ood = bool(self.rng.random() < self.config.ood_prob)
        return temporal, ood

    def _sample_feature_matrix(
        self,
        num_rows: int,
        num_features: int,
        train_size: int,
        temporal: bool,
        ood: bool,
    ) -> NDArray[np.float32]:
        x = np.zeros((num_rows, num_features), dtype=np.float64)
        for col in range(num_features):
            source_kind = self._sample_source_kind(temporal and col == 0)
            raw = self._sample_source_column(num_rows, source_kind)
            if ood:
                shift = float(self.rng.normal(0.5, 0.2))
                scale = float(self.rng.uniform(0.7, 1.8))
                raw[train_size:] = raw[train_size:] * scale + shift
            raw = raw + self.rng.normal(0.0, 1e-4, size=num_rows)
            x[:, col] = _standardize(raw, train_size=train_size if ood else None)
        return x.astype(np.float32)

    def _ensure_accepted_task(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.integer] | NDArray[np.floating],
        task_type: TaskType,
        train_size: int,
        categorical_features: tuple[int, ...],
    ) -> None:
        if not self.config.difficulty_filter:
            return
        features = _prepare_accept_task_features(
            x.astype(np.float64), train_size, categorical_features
        )
        if features.shape[1] == 0:
            raise ValueError("difficulty filter rejected task with no usable features")
        rff_features = _append_accept_rff_features(features, train_size, self.rng)
        if task_type == "classification":
            labels = y.astype(np.int64)
            y_train = labels[:train_size]
            y_test = labels[train_size:]
            classes = np.unique(y_train)
            if len(classes) < 2:
                raise ValueError(
                    "difficulty filter rejected classification task with fewer than two train classes"
                )
            majority_accuracy = _majority_accuracy(y_test)
            probe_accuracy = max(
                _ridge_classification_accuracy(
                    features[:train_size],
                    y_train,
                    features[train_size:],
                    y_test,
                    classes,
                ),
                _ridge_classification_accuracy(
                    rff_features[:train_size],
                    y_train,
                    rff_features[train_size:],
                    y_test,
                    classes,
                ),
            )
            if probe_accuracy > self.config.max_classification_probe_accuracy:
                raise ValueError("difficulty filter rejected near-perfect classification task")
            if probe_accuracy < majority_accuracy + self.config.min_classification_probe_margin:
                raise ValueError("difficulty filter rejected low-signal classification task")
            return

        targets = y.astype(np.float64)
        r2 = max(
            _ridge_regression_r2(
                features[:train_size],
                targets[:train_size],
                features[train_size:],
                targets[train_size:],
            ),
            _ridge_regression_r2(
                rff_features[:train_size],
                targets[:train_size],
                rff_features[train_size:],
                targets[train_size:],
            ),
        )
        if r2 > self.config.max_regression_probe_r2:
            raise ValueError("difficulty filter rejected near-perfect regression task")
        if r2 < self.config.min_regression_probe_r2:
            raise ValueError("difficulty filter rejected low-signal regression task")

    def _sample_source_kind(self, prefer_temporal: bool) -> SourceKind:
        temporal_kinds = tuple(
            kind
            for kind in self.config.source_kinds
            if kind in ("trend", "cycle", "multi_sine", "ar1", "seasonal_ar", "event")
        )
        if prefer_temporal and temporal_kinds:
            return temporal_kinds[int(self.rng.integers(0, len(temporal_kinds)))]
        return self.config.source_kinds[int(self.rng.integers(0, len(self.config.source_kinds)))]

    def _sample_source_column(self, num_rows: int, source_kind: SourceKind) -> NDArray[np.float64]:
        if source_kind == "mixed":
            source_kind = ("normal", "uniform", "beta", "zipf")[int(self.rng.integers(0, 4))]
        if source_kind == "normal":
            return self.rng.normal(0.0, self.rng.uniform(0.5, 2.0), size=num_rows)
        if source_kind == "uniform":
            low, high = sorted(self.rng.uniform(-3.0, 3.0, size=2))
            return self.rng.uniform(low, high, size=num_rows)
        if source_kind == "beta":
            alpha = float(self.rng.uniform(0.5, 5.0))
            beta = float(self.rng.uniform(0.5, 5.0))
            return self.rng.beta(alpha, beta, size=num_rows)
        if source_kind == "zipf":
            return np.minimum(self.rng.zipf(float(self.rng.uniform(1.5, 4.0)), size=num_rows), 20.0)
        if source_kind == "trend":
            slope = float(self.rng.uniform(-2.0, 2.0))
            return slope * np.linspace(-1.0, 1.0, num_rows) + self.rng.normal(
                0.0, 0.1, size=num_rows
            )
        if source_kind == "cycle":
            time = np.linspace(0.0, 1.0, num_rows)
            frequency = float(self.rng.uniform(1.0, 25.0))
            phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
            return np.sin(2.0 * np.pi * frequency * time + phase)
        if source_kind == "multi_sine":
            time = np.linspace(0.0, 1.0, num_rows)
            raw = np.zeros(num_rows, dtype=np.float64)
            for _ in range(int(self.rng.integers(2, 7))):
                amplitude = float(self.rng.uniform(0.2, 1.5))
                frequency = float(np.exp(self.rng.uniform(np.log(0.25), np.log(150.0))))
                phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
                raw = raw + amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
            return raw
        if source_kind == "ar1":
            rho = float(self.rng.uniform(0.0, 0.95))
            raw = np.zeros(num_rows, dtype=np.float64)
            noise = self.rng.normal(0.0, 1.0, size=num_rows)
            for idx in range(1, num_rows):
                raw[idx] = rho * raw[idx - 1] + noise[idx]
            return raw
        if source_kind == "seasonal_ar":
            rho = float(self.rng.uniform(0.2, 0.95))
            period = float(self.rng.uniform(4.0, max(5.0, num_rows / 3.0)))
            raw = np.zeros(num_rows, dtype=np.float64)
            noise = self.rng.normal(0.0, 0.5, size=num_rows)
            for idx in range(1, num_rows):
                seasonal = np.sin(2.0 * np.pi * idx / period)
                raw[idx] = rho * raw[idx - 1] + seasonal + noise[idx]
            return raw
        if source_kind == "event":
            raw = self.rng.normal(0.0, 0.3, size=num_rows)
            idx = np.arange(num_rows, dtype=np.float64)
            for _ in range(int(self.rng.integers(1, 5))):
                center = float(self.rng.integers(0, num_rows))
                width = float(self.rng.uniform(1.0, max(2.0, num_rows / 10.0)))
                amplitude = float(self.rng.normal(0.0, 2.0))
                raw = raw + amplitude * np.exp(-((idx - center) ** 2) / (2.0 * width * width))
            return raw
        raise ValueError(f"unknown source kind {source_kind}")


class PriorGenerator(BasePriorGenerator):
    prior_type: PriorType = "scm"

    def __init__(
        self,
        config: PriorConfig | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(config=config, seed=seed, rng=rng)
        self.scm = SCMGenerator(rng=self.rng, config=self.config)

    def sample_dataset(self) -> SyntheticDataset:
        return self._retry_dataset(self._sample_dataset_once)

    def _sample_dataset_once(self) -> SyntheticDataset:
        num_rows = self._sample_rows()
        num_features = self._sample_num_features()
        node_dim = self._sample_node_dim()
        num_hidden = int(
            self.rng.integers(self.config.min_hidden_nodes, self.config.max_hidden_nodes + 1)
        )
        num_nodes = num_features + num_hidden + 1
        train_size = self._sample_train_size(num_rows)
        task_type = self._sample_task()
        temporal, ood = self._sample_flags()

        spec = self.scm.sample_spec(
            num_nodes, node_dim=node_dim, prefer_temporal=temporal, enable_dynamic=True
        )
        values = self.scm.sample_values(
            spec, num_rows=num_rows, train_size=train_size, temporal=temporal, ood=ood
        )

        initial_feature_nodes, initial_feature_components, target_node, target_component = (
            self._sample_feature_target_slots(
                spec,
                num_features,
            )
        )
        x = np.column_stack(
            [
                values[:, node, component]
                for node, component in zip(initial_feature_nodes, initial_feature_components)
            ]
        ).astype(np.float32)
        x, feature_types, categorical_features, feature_nodes, feature_components = (
            self.postprocessor.postprocess_features(
                x=x,
                feature_nodes=initial_feature_nodes,
                feature_components=initial_feature_components,
                train_size=train_size,
                ood=ood,
            )
        )
        ordered = temporal or ood or len(spec.dynamic_nodes) > 0
        y, num_classes = self.postprocessor.make_target(
            raw_target=values[:, target_node, target_component],
            task_type=task_type,
            num_rows=num_rows,
            train_size=train_size,
            ordered=ordered,
        )
        x, y = self.postprocessor.finalize_order(x, y, task_type, train_size, ordered=ordered)
        self._ensure_accepted_task(x, y, task_type, train_size, categorical_features)

        return SyntheticDataset(
            x=x,
            y=y,
            feature_types=feature_types,
            meta=DatasetMeta(
                prior_type=self.prior_type,
                num_rows=num_rows,
                num_features=x.shape[1],
                task_type=task_type,
                train_size=train_size,
                num_classes=num_classes,
                temporal=temporal,
                ood=ood,
                feature_nodes=feature_nodes,
                target_node=target_node,
                dag_edges=spec.edges,
                categorical_features=categorical_features,
                graph_layout=spec.graph_layout,
                source_kinds=spec.root_source_kinds,
                aggregation_kinds=spec.aggregation_kinds,
                mechanism_kinds=spec.mechanism_kinds,
                edge_mechanism_kinds=spec.edge_mechanism_kinds,
                dynamic_nodes=spec.dynamic_nodes,
                node_dim=node_dim,
                feature_components=feature_components,
                target_component=target_component,
            ),
        )

    def _sample_feature_target_slots(
        self,
        spec: SCMSpec,
        num_features: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
        parents = spec.parents
        num_nodes = len(parents)
        target_candidates = [
            node for node, node_parents in enumerate(parents) if len(node_parents) > 0
        ]
        if not target_candidates:
            raise ValueError("SCM graph has no non-root target candidates")

        terminal_node = num_nodes - 1
        non_terminal_candidates = [node for node in target_candidates if node != terminal_node]
        if non_terminal_candidates and self.rng.random() < 0.75:
            target_node = int(self.rng.choice(non_terminal_candidates))
        else:
            target_node = int(self.rng.choice(target_candidates))
        target_component = int(self.rng.integers(0, spec.node_dim))

        feature_candidates = [
            (node, component)
            for node in range(num_nodes)
            for component in range(spec.node_dim)
            if (node, component) != (target_node, target_component)
        ]
        if num_features > len(feature_candidates):
            raise ValueError("not enough SCM node components to sample requested features")

        ancestors = _target_ancestors_from_parents(spec.parents, target_node)
        ancestor_candidates = [slot for slot in feature_candidates if slot[0] in ancestors]
        selected: list[tuple[int, int]] = []
        if ancestor_candidates:
            required_count = min(len(ancestor_candidates), max(1, (2 * num_features) // 3))
            ancestor_indices = self.rng.choice(
                len(ancestor_candidates), size=required_count, replace=False
            )
            selected.extend(
                ancestor_candidates[int(idx)] for idx in np.atleast_1d(ancestor_indices)
            )

        remaining = [slot for slot in feature_candidates if slot not in set(selected)]
        remaining_count = num_features - len(selected)
        if remaining_count > 0:
            remaining_indices = self.rng.choice(len(remaining), size=remaining_count, replace=False)
            selected.extend(remaining[int(idx)] for idx in np.atleast_1d(remaining_indices))

        self.rng.shuffle(selected)
        feature_nodes = tuple(int(node) for node, _ in selected)
        feature_components = tuple(int(component) for _, component in selected)
        return feature_nodes, feature_components, target_node, target_component


class TreePriorGenerator(BasePriorGenerator):
    prior_type: PriorType = "tree"

    def sample_dataset(self) -> SyntheticDataset:
        return self._retry_dataset(self._sample_dataset_once)

    def _sample_dataset_once(self) -> SyntheticDataset:
        num_rows = self._sample_rows()
        num_features = self._sample_num_features()
        train_size = self._sample_train_size(num_rows)
        task_type = self._sample_task()
        temporal, ood = self._sample_flags()

        x_raw = self._sample_feature_matrix(num_rows, num_features, train_size, temporal, ood)
        target = self._sample_tree_ensemble(x_raw)
        x, feature_types, categorical_features, feature_nodes, feature_components = (
            self.postprocessor.postprocess_features(
                x=x_raw,
                feature_nodes=tuple(range(num_features)),
                feature_components=tuple(0 for _ in range(num_features)),
                train_size=train_size,
                ood=ood,
            )
        )
        y, num_classes = self.postprocessor.make_target(
            raw_target=target,
            task_type=task_type,
            num_rows=num_rows,
            train_size=train_size,
            ordered=temporal or ood,
        )
        x, y = self.postprocessor.finalize_order(
            x, y, task_type, train_size, ordered=temporal or ood
        )
        self._ensure_accepted_task(x, y, task_type, train_size, categorical_features)

        return SyntheticDataset(
            x=x,
            y=y,
            feature_types=feature_types,
            meta=DatasetMeta(
                prior_type=self.prior_type,
                num_rows=num_rows,
                num_features=x.shape[1],
                task_type=task_type,
                train_size=train_size,
                num_classes=num_classes,
                temporal=temporal,
                ood=ood,
                feature_nodes=feature_nodes,
                target_node=num_features,
                dag_edges=tuple((feature, num_features) for feature in feature_nodes),
                categorical_features=categorical_features,
                node_dim=1,
                feature_components=feature_components,
            ),
        )

    def _sample_tree_ensemble(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        num_estimators = int(
            self.rng.integers(self.config.tree_min_estimators, self.config.tree_max_estimators + 1)
        )
        target = np.zeros(x.shape[0], dtype=np.float64)
        for _ in range(num_estimators):
            depth = int(
                self.rng.integers(self.config.tree_min_depth, self.config.tree_max_depth + 1)
            )
            tree = self._sample_tree(depth=depth, num_features=x.shape[1])
            target += self._eval_tree(tree, x)
        noise = self.rng.normal(0.0, self.config.noise_max, size=x.shape[0])
        return _standardize(target / np.sqrt(num_estimators) + noise)

    def _sample_tree(self, depth: int, num_features: int) -> TreeNode:
        if depth == 0:
            return TreeNode(feature=None, threshold=0.0, value=float(self.rng.normal()))
        feature = int(self.rng.integers(0, num_features))
        threshold = float(self.rng.normal(0.0, 0.8))
        return TreeNode(
            feature=feature,
            threshold=threshold,
            value=None,
            left=self._sample_tree(depth - 1, num_features),
            right=self._sample_tree(depth - 1, num_features),
        )

    def _eval_tree(self, tree: TreeNode, x: NDArray[np.float32]) -> NDArray[np.float64]:
        if tree.value is not None:
            return np.full(x.shape[0], tree.value, dtype=np.float64)
        if tree.feature is None or tree.left is None or tree.right is None:
            raise ValueError("invalid tree node")
        left_values = self._eval_tree(tree.left, x)
        right_values = self._eval_tree(tree.right, x)
        return np.where(x[:, tree.feature] <= tree.threshold, left_values, right_values)


class GPPriorGenerator(BasePriorGenerator):
    prior_type: PriorType = "gp"

    def sample_dataset(self) -> SyntheticDataset:
        return self._retry_dataset(self._sample_dataset_once)

    def _sample_dataset_once(self) -> SyntheticDataset:
        num_rows = self._sample_rows()
        num_features = self._sample_num_features()
        train_size = self._sample_train_size(num_rows)
        task_type = self._sample_task()
        temporal, ood = self._sample_flags()

        x_raw = self._sample_feature_matrix(num_rows, num_features, train_size, temporal, ood)
        target = self._sample_gp_function(x_raw)
        x, feature_types, categorical_features, feature_nodes, feature_components = (
            self.postprocessor.postprocess_features(
                x=x_raw,
                feature_nodes=tuple(range(num_features)),
                feature_components=tuple(0 for _ in range(num_features)),
                train_size=train_size,
                ood=ood,
            )
        )
        y, num_classes = self.postprocessor.make_target(
            raw_target=target,
            task_type=task_type,
            num_rows=num_rows,
            train_size=train_size,
            ordered=temporal or ood,
        )
        x, y = self.postprocessor.finalize_order(
            x, y, task_type, train_size, ordered=temporal or ood
        )
        self._ensure_accepted_task(x, y, task_type, train_size, categorical_features)

        return SyntheticDataset(
            x=x,
            y=y,
            feature_types=feature_types,
            meta=DatasetMeta(
                prior_type=self.prior_type,
                num_rows=num_rows,
                num_features=x.shape[1],
                task_type=task_type,
                train_size=train_size,
                num_classes=num_classes,
                temporal=temporal,
                ood=ood,
                feature_nodes=feature_nodes,
                target_node=num_features,
                dag_edges=tuple((feature, num_features) for feature in feature_nodes),
                categorical_features=categorical_features,
                node_dim=1,
                feature_components=feature_components,
            ),
        )

    def _sample_gp_function(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        num_basis = self.config.gp_num_basis
        lengthscale = float(
            np.exp(
                self.rng.uniform(
                    np.log(self.config.gp_lengthscale_min), np.log(self.config.gp_lengthscale_max)
                )
            )
        )
        weights = self.rng.normal(0.0, 1.0 / lengthscale, size=(x.shape[1], num_basis))
        phases = self.rng.uniform(0.0, 2.0 * np.pi, size=num_basis)
        coefficients = self.rng.normal(0.0, 1.0, size=num_basis)
        features = np.sqrt(2.0 / num_basis) * np.cos(x.astype(np.float64) @ weights + phases)
        noise_scale = float(
            np.exp(self.rng.uniform(np.log(self.config.noise_min), np.log(self.config.noise_max)))
        )
        target = features @ coefficients + self.rng.normal(0.0, noise_scale, size=x.shape[0])
        return _standardize(target)


class PriorBagGenerator:
    def __init__(self, config: PriorConfig | None = None, seed: int | None = None) -> None:
        self.config = config or PriorConfig()
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.generators: dict[PriorType, BasePriorGenerator] = {
            "scm": PriorGenerator(self.config, rng=self.rng),
            "tree": TreePriorGenerator(self.config, rng=self.rng),
            "gp": GPPriorGenerator(self.config, rng=self.rng),
        }

    def sample_dataset(self) -> SyntheticDataset:
        prior_type = self._sample_prior_type()
        return self.generators[prior_type].sample_dataset()

    def sample_batch(self, batch_size: int) -> list[SyntheticDataset]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return [self.sample_dataset() for _ in range(batch_size)]

    def _sample_prior_type(self) -> PriorType:
        if self.config.prior in ("scm", "tree", "gp"):
            return self.config.prior

        prior_types: tuple[PriorType, ...] = ("scm", "tree", "gp")
        weights = np.array(
            [self.config.scm_weight, self.config.tree_weight, self.config.gp_weight],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
        return prior_types[int(self.rng.choice(len(prior_types), p=probabilities))]


def make_prior_generator(
    config: PriorConfig | None = None, seed: int | None = None
) -> BasePriorGenerator | PriorBagGenerator:
    config = config or PriorConfig()
    if config.prior == "scm":
        return PriorGenerator(config, seed=seed)
    if config.prior == "tree":
        return TreePriorGenerator(config, seed=seed)
    if config.prior == "gp":
        return GPPriorGenerator(config, seed=seed)
    return PriorBagGenerator(config, seed=seed)


def collate_datasets(datasets: Sequence[SyntheticDataset]) -> SyntheticBatch:
    if len(datasets) == 0:
        raise ValueError("datasets must be non-empty")

    batch_size = len(datasets)
    max_rows = max(dataset.x.shape[0] for dataset in datasets)
    max_features = max(dataset.x.shape[1] for dataset in datasets)

    x = np.zeros((batch_size, max_rows, max_features), dtype=np.float32)
    y = np.zeros((batch_size, max_rows), dtype=np.float32)
    row_mask = np.zeros((batch_size, max_rows), dtype=np.bool_)
    feature_mask = np.zeros((batch_size, max_features), dtype=np.bool_)
    feature_is_categorical = np.zeros((batch_size, max_features), dtype=np.bool_)
    train_sizes = np.zeros(batch_size, dtype=np.int64)
    num_classes = np.full(batch_size, -1, dtype=np.int64)
    task_types: list[TaskType] = []

    for i, dataset in enumerate(datasets):
        rows, features = dataset.x.shape
        x[i, :rows, :features] = dataset.x
        y[i, :rows] = dataset.y.astype(np.float32)
        row_mask[i, :rows] = True
        feature_mask[i, :features] = True
        feature_is_categorical[i, list(dataset.meta.categorical_features)] = True
        train_sizes[i] = dataset.meta.train_size
        if dataset.meta.num_classes is not None:
            num_classes[i] = dataset.meta.num_classes
        task_types.append(dataset.meta.task_type)

    return SyntheticBatch(
        x=x,
        y=y,
        row_mask=row_mask,
        feature_mask=feature_mask,
        feature_is_categorical=feature_is_categorical,
        train_sizes=train_sizes,
        task_types=tuple(task_types),
        num_classes=num_classes,
    )


def _standardize(values: NDArray[np.float64], train_size: int | None = None) -> NDArray[np.float64]:
    fit_values = values if train_size is None else values[:train_size]
    std = float(np.std(fit_values))
    if std <= 1e-12 or not np.isfinite(std):
        raise ValueError("cannot standardize a constant column")
    return (values - float(np.mean(fit_values))) / std


def _remove_outliers(values: NDArray[np.float64], threshold: float = 4.0) -> NDArray[np.float64]:
    mean = float(np.mean(values))
    std = max(float(np.std(values)), 1e-6)
    lower = mean - threshold * std
    upper = mean + threshold * std
    clipped = np.clip(values, lower, upper)
    return clipped.astype(np.float64)


def _quantile_bins(values: NDArray[np.float64], categories: int) -> NDArray[np.int64]:
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    bins = np.floor(ranks * categories / len(values)).astype(np.int64)
    return np.minimum(bins, categories - 1)


def _rank_to_uniform(values: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float64)
    uniform = (ranks + 0.5) / float(len(values))
    return np.clip(uniform, eps, 1.0 - eps)


def _proportion_bins(
    values: NDArray[np.float64], proportions: NDArray[np.float64]
) -> NDArray[np.int64]:
    proportions = np.asarray(proportions, dtype=np.float64)
    proportions = np.maximum(proportions, 0.0)
    if float(np.sum(proportions)) <= 0.0:
        raise ValueError("class proportions must have positive mass")
    proportions = proportions / float(np.sum(proportions))
    order = np.argsort(values, kind="mergesort")
    raw_counts = np.maximum(1, np.floor(proportions * len(values)).astype(np.int64))
    while int(np.sum(raw_counts)) > len(values):
        idx = int(np.argmax(raw_counts))
        if raw_counts[idx] == 1:
            break
        raw_counts[idx] -= 1
    while int(np.sum(raw_counts)) < len(values):
        raw_counts[int(np.argmax(proportions))] += 1
    labels = np.empty(len(values), dtype=np.int64)
    start = 0
    for class_id, count in enumerate(raw_counts):
        stop = min(start + int(count), len(values))
        labels[order[start:stop]] = class_id
        start = stop
    return labels


def _ensure_dense_labels(
    labels: NDArray[np.int64],
    values: NDArray[np.float64],
    num_classes: int,
) -> NDArray[np.int64]:
    labels = labels.astype(np.int64, copy=True)
    present = set(np.unique(labels).astype(int).tolist())
    missing = [class_id for class_id in range(num_classes) if class_id not in present]
    if not missing:
        return labels
    order = np.argsort(values, kind="mergesort")
    donor_cursor = 0
    for class_id in missing:
        while donor_cursor < len(order):
            idx = int(order[donor_cursor])
            donor_cursor += 1
            if int(np.sum(labels == labels[idx])) > 1:
                labels[idx] = class_id
                break
    return labels


def _ensure_ordered_split_classes(
    labels: NDArray[np.int64],
    values: NDArray[np.float64],
    num_classes: int,
    train_size: int,
) -> NDArray[np.int64]:
    labels = labels.astype(np.int64, copy=True)
    split_slices = (slice(0, train_size), slice(train_size, len(labels)))
    for split_slice in split_slices:
        split_indices = np.arange(len(labels))[split_slice]
        present = set(np.unique(labels[split_slice]).astype(int).tolist())
        missing = [class_id for class_id in range(num_classes) if class_id not in present]
        if not missing:
            continue
        ordered_indices = split_indices[np.argsort(values[split_slice], kind="mergesort")]
        for class_id in missing:
            donor = _find_ordered_split_donor(labels, ordered_indices)
            labels[donor] = class_id
    return labels


def _find_ordered_split_donor(labels: NDArray[np.int64], ordered_indices: NDArray[np.int64]) -> int:
    for idx in ordered_indices:
        if int(np.sum(labels[ordered_indices] == labels[int(idx)])) > 1:
            return int(idx)
    raise ValueError("could not make ordered classification split dense")


def _is_informative_feature(values: NDArray[np.float64], feature_type: FeatureType) -> bool:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return False
    if feature_type == "categorical":
        return len(np.unique(finite.astype(np.int64))) > 1
    return bool(float(np.std(finite)) > 1e-8)


def _has_valid_classification_split(labels: NDArray[np.integer], train_size: int) -> bool:
    train_classes = set(np.unique(labels[:train_size]).astype(int).tolist())
    test_classes = set(np.unique(labels[train_size:]).astype(int).tolist())
    return train_classes == test_classes and len(train_classes) >= 2


def _target_ancestors_from_parents(
    parents: tuple[tuple[int, ...], ...], target_node: int
) -> set[int]:
    ancestors: set[int] = set()
    stack = list(parents[target_node])
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        stack.extend(parents[node])
    return ancestors


def _fill_missing_with_train_median(
    values: NDArray[np.float64], train_size: int
) -> NDArray[np.float64]:
    values = values.astype(np.float64, copy=True)
    train = values[:train_size]
    finite = np.isfinite(train)
    if not bool(np.any(finite)):
        raise ValueError("cannot impute column with no finite train values")
    fill = float(np.median(train[finite]))
    values[~np.isfinite(values)] = fill
    return values


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _prepare_accept_task_features(
    x: NDArray[np.float64],
    train_size: int,
    categorical_features: tuple[int, ...],
) -> NDArray[np.float64]:
    categorical = set(categorical_features)
    pieces: list[NDArray[np.float64]] = []
    for col in range(x.shape[1]):
        column = x[:, col].astype(np.float64, copy=True)
        if col in categorical:
            train = column[:train_size]
            finite_train = train[np.isfinite(train)]
            if len(finite_train) == 0:
                continue
            categories = np.unique(finite_train.astype(np.int64))
            if len(categories) <= 1:
                continue
            if len(categories) <= 64:
                encoded = np.zeros((x.shape[0], len(categories)), dtype=np.float64)
                finite_column = np.isfinite(column)
                int_column = np.zeros_like(column, dtype=np.int64)
                int_column[finite_column] = column[finite_column].astype(np.int64)
                for idx, category in enumerate(categories):
                    encoded[:, idx] = (finite_column & (int_column == category)).astype(np.float64)
                pieces.append(encoded)
                continue
        filled = _fill_missing_with_train_median(column, train_size)
        pieces.append(_standardize(filled, train_size=train_size)[:, None])
    if not pieces:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    return np.concatenate(pieces, axis=1)


def _append_accept_rff_features(
    features: NDArray[np.float64],
    train_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    rff_count = min(64, max(16, features.shape[1] * 4))
    train = features[:train_size]
    scale = np.std(train, axis=0) + 1e-6
    normalized = features / scale
    weights = rng.normal(0.0, 1.0, size=(normalized.shape[1], rff_count))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=rff_count)
    fourier = np.sqrt(2.0 / rff_count) * np.cos(normalized @ weights + phases)
    return np.concatenate([features, fourier], axis=1)


def _majority_accuracy(labels: NDArray[np.int64]) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    return float(np.max(counts) / len(labels))


def _ridge_classification_accuracy(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    classes: NDArray[np.int64],
) -> float:
    targets = (y_train[:, None] == classes[None, :]).astype(np.float64)
    weights = _ridge_weights(_with_bias(x_train), targets, alpha=1.0)
    predictions = classes[np.argmax(_with_bias(x_test) @ weights, axis=1)]
    return float(np.mean(predictions == y_test)) if len(y_test) > 0 else 0.0


def _ridge_regression_r2(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> float:
    weights = _ridge_weights(_with_bias(x_train), y_train[:, None], alpha=1.0)
    predictions = (_with_bias(x_test) @ weights).ravel()
    variance = float(np.sum((y_test - float(np.mean(y_test))) ** 2))
    if variance <= 1e-12:
        raise ValueError("difficulty filter rejected constant regression test target")
    residual = float(np.sum((predictions - y_test) ** 2))
    return 1.0 - residual / variance


def _ridge_weights(
    x_train: NDArray[np.float64], targets: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    gram = x_train.T @ x_train
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    try:
        return np.linalg.solve(gram + regularizer, x_train.T @ targets)
    except np.linalg.LinAlgError as exc:
        raise ValueError("difficulty filter rejected singular probe design") from exc


def _with_bias(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
