from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.config import RDBPriorConfig
from rfm.rdb.types import FeatureColumnType, RelationMessageSet, TableSpec
from rfm.scm import ExogenousContext, SCMGenerator


@dataclass(frozen=True)
class ContentGenerationResult:
    values: dict[str, NDArray[np.float32]]
    missing_mask: NDArray[np.bool_]
    feature_parent_sources: dict[str, tuple[str, ...]]
    uses_parent_features: dict[str, bool]
    num_exogenous: int


class RelationalSCMGenerator:
    """Stage 3: TabPFN-3 SCM engine with relational exogenous conditioning."""

    def __init__(self, rng: np.random.Generator, config: RDBPriorConfig) -> None:
        self.rng = rng
        self.config = config
        no_missing_prior = replace(
            config.relational_scm.prior,
            mcar_missing_prob=0.0,
            mar_missing_prob=0.0,
            mnar_missing_prob=0.0,
        )
        self.scm = SCMGenerator(rng=rng, config=no_missing_prior)
        self.prior_config = no_missing_prior

    def sample_table_columns(
        self,
        table: TableSpec,
        row_context: NDArray[np.float64],
        parent_context: NDArray[np.float64],
        topology_context: NDArray[np.float64],
        relation_messages: RelationMessageSet | None = None,
    ) -> ContentGenerationResult:
        feature_columns = table.feature_columns
        if len(feature_columns) == 0:
            return ContentGenerationResult(
                values={},
                missing_mask=np.zeros((table.row_count, 0), dtype=np.bool_),
                feature_parent_sources={},
                uses_parent_features={},
                num_exogenous=0,
            )

        num_rows = table.row_count
        train_size = min(max(int(round(0.7 * num_rows)), 1), num_rows - 1)
        edge_messages = relation_messages.edge_messages if relation_messages is not None else None
        if not self.config.use_parent_feature_messages:
            edge_messages = None

        exogenous = ExogenousContext(
            row_context=row_context.astype(np.float64, copy=False),
            parent_context=parent_context.astype(np.float64, copy=False),
            topology_context=topology_context.astype(np.float64, copy=False),
            edge_messages=edge_messages,
        )
        stacked = exogenous.stacked()
        num_exogenous = stacked.shape[1]
        num_features = len(feature_columns)
        num_hidden = int(
            self.rng.integers(
                self.config.relational_scm.min_hidden_nodes,
                self.config.relational_scm.max_hidden_nodes + 1,
            )
        )
        num_nodes = num_features + num_hidden

        spec = self.scm.sample_spec_with_exogenous(
            num_nodes=num_nodes,
            num_exogenous=num_exogenous,
            node_dim=1,
            prefer_temporal=table.role in ("activity/event", "snapshot/state"),
            enable_dynamic=False,
        )
        node_values = self.scm.sample_values(
            spec=spec,
            num_rows=num_rows,
            train_size=train_size,
            temporal=False,
            ood=False,
            exogenous=exogenous,
        )

        feature_start = spec.num_exogenous + num_hidden
        feature_nodes = tuple(range(feature_start, feature_start + num_features))
        raw_features = np.column_stack(
            [node_values[:, node, 0] for node in feature_nodes[:num_features]]
        ).astype(np.float64)

        processed = self._postprocess_raw_features(raw_features)
        if processed.shape[1] != len(feature_columns):
            raise ValueError("relational content generation changed the schema feature count")

        parent_tables = _parent_tables_from_messages(relation_messages)
        feature_parent_sources = {
            column.name: parent_tables for column in feature_columns if len(parent_tables) > 0
        }
        uses_parent_features = {
            column.name: bool(len(parent_tables) > 0 and self.config.use_parent_feature_messages)
            for column in feature_columns
        }

        values: dict[str, NDArray[np.float32]] = {}
        missing_mask = self._sample_feature_missing_mask(num_rows, len(feature_columns))
        for col_idx, column in enumerate(feature_columns):
            column_values = self._cast_feature(processed[:, col_idx], column.value_type)
            column_values = column_values.astype(np.float32, copy=True)
            column_values[missing_mask[:, col_idx]] = np.nan
            values[column.name] = column_values

        return ContentGenerationResult(
            values=values,
            missing_mask=missing_mask,
            feature_parent_sources=feature_parent_sources,
            uses_parent_features=uses_parent_features,
            num_exogenous=num_exogenous,
        )

    def _postprocess_raw_features(self, raw_features: NDArray[np.float64]) -> NDArray[np.float64]:
        processed = raw_features.astype(np.float64, copy=True)
        for col in range(processed.shape[1]):
            processed[:, col] = self._maybe_warp_continuous(processed[:, col])
            processed[:, col] = _standardize(processed[:, col])
        return processed

    def _maybe_warp_continuous(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.rng.random() >= self.prior_config.marginal_warp_prob:
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

    def _cast_feature(
        self, raw: NDArray[np.float64], feature_type: FeatureColumnType | str
    ) -> NDArray[np.float32]:
        if feature_type == "continuous":
            return raw.astype(np.float32, copy=True)
        if feature_type == "categorical":
            return _quantile_bins(raw, int(self.rng.integers(3, 16))).astype(np.float32)
        if feature_type == "ordinal":
            return _quantile_bins(raw, int(self.rng.integers(3, 8))).astype(np.float32)
        if feature_type == "binary":
            threshold = float(self.rng.normal(0.0, 0.5))
            return (raw > threshold).astype(np.float32)
        if feature_type == "count":
            cleaned = np.nan_to_num(raw, nan=0.0, posinf=3.0, neginf=-5.0)
            rate = np.exp(np.clip(cleaned, -5.0, 3.0))
            rate = np.minimum(rate, 20.0)
            return self.rng.poisson(rate).astype(np.float32)
        if feature_type == "quantized":
            return _quantile_bins(raw, int(self.rng.integers(8, 33))).astype(np.float32)
        if feature_type == "high_cardinality_categorical":
            categories = min(max(16, raw.shape[0] // 2), 256)
            return _quantile_bins(raw, categories).astype(np.float32)
        raise ValueError(f"unknown feature type {feature_type}")

    def _sample_feature_missing_mask(self, num_rows: int, num_features: int) -> NDArray[np.bool_]:
        prob = self.config.relational_scm.feature_missing_probability
        max_rate = self.config.relational_scm.max_feature_missing_rate
        mask = np.zeros((num_rows, num_features), dtype=np.bool_)
        if self.rng.random() >= prob:
            return mask
        rate = min(float(self.rng.beta(0.5, 10.0)), max_rate)
        count = int(round(rate * num_rows))
        if count > 0:
            rows = self.rng.choice(num_rows, size=count, replace=False)
            for col in range(num_features):
                if self.rng.random() < 0.5:
                    mask[rows, col] = True
        return mask


def _parent_tables_from_messages(relation_messages: RelationMessageSet | None) -> tuple[str, ...]:
    if relation_messages is None:
        return ()
    tables = sorted(
        {message.parent_table for message in relation_messages.messages if not message.is_null}
    )
    return tuple(tables)


def _quantile_bins(values: NDArray[np.float64], categories: int) -> NDArray[np.int64]:
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    bins = np.floor(ranks * categories / len(values)).astype(np.int64)
    return np.minimum(bins, categories - 1)


def _rank_to_uniform(values: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float64)
    uniform = (ranks + 0.5) / float(len(values))
    return np.clip(uniform, eps, 1.0 - eps)


def _standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    std = float(np.std(values))
    if std <= 1e-12 or not np.isfinite(std):
        raise ValueError("cannot standardize a constant relational feature")
    return (values - float(np.mean(values))) / std
