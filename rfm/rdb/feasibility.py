from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.schema import edge_intent_spec
from rfm.rdb.types import ForeignKeyCardinality, ForeignKeySpec, SchemaGraph, TableSpec


class SchemaFeasibilityError(RuntimeError):
    """Raised when preflight finds an edge that cannot be made feasible by an explicit rule."""


@dataclass(frozen=True)
class EdgeFeasibilityAdjustment:
    fk_key: str
    action: str
    reason_code: str
    before: dict[str, object]
    after: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SchemaFeasibilityReport:
    checked_edges: int
    adjustments: tuple[EdgeFeasibilityAdjustment, ...]

    @property
    def edge_downgrade_count(self) -> int:
        return sum(
            1 for adjustment in self.adjustments if adjustment.action.startswith("downgrade_")
        )

    @property
    def edge_adjustment_count(self) -> int:
        return len(self.adjustments)

    @property
    def reason_histogram(self) -> dict[str, int]:
        return dict(Counter(adjustment.reason_code for adjustment in self.adjustments))

    @property
    def action_histogram(self) -> dict[str, int]:
        return dict(Counter(adjustment.action for adjustment in self.adjustments))

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_edges": self.checked_edges,
            "edge_adjustment_count": self.edge_adjustment_count,
            "edge_downgrade_count": self.edge_downgrade_count,
            "reason_histogram": self.reason_histogram,
            "action_histogram": self.action_histogram,
            "adjustments": [adjustment.to_dict() for adjustment in self.adjustments],
        }


_CARDINALITY_DOWNGRADE_DAG: dict[ForeignKeyCardinality, tuple[ForeignKeyCardinality, ...]] = {
    "one_to_one": ("many_to_one",),
    "capacity_limited": ("many_to_one",),
    "optional": ("many_to_one",),
    "multi_parent_member": (),
    "many_to_one": (),
}


def preflight_schema_feasibility(
    schema: SchemaGraph,
    timestamps: Mapping[str, NDArray[np.float64]],
) -> tuple[SchemaGraph, SchemaFeasibilityReport]:
    edges: list[ForeignKeySpec] = []
    adjustments: list[EdgeFeasibilityAdjustment] = []
    for fk in schema.edges:
        checked_fk, edge_adjustments = _preflight_edge(fk, schema.tables, timestamps)
        edges.append(checked_fk)
        adjustments.extend(edge_adjustments)
    report = SchemaFeasibilityReport(
        checked_edges=len(schema.edges), adjustments=tuple(adjustments)
    )
    return replace(schema, edges=tuple(edges)), report


def _preflight_edge(
    fk: ForeignKeySpec,
    table_specs: Mapping[str, TableSpec],
    timestamps: Mapping[str, NDArray[np.float64]],
) -> tuple[ForeignKeySpec, tuple[EdgeFeasibilityAdjustment, ...]]:
    adjustments: list[EdgeFeasibilityAdjustment] = []
    current = fk

    current, temporal_adjustment = _preflight_temporal(current, table_specs)
    if temporal_adjustment is not None:
        adjustments.append(temporal_adjustment)

    current, cardinality_adjustment = _preflight_cardinality(current, table_specs, timestamps)
    if cardinality_adjustment is not None:
        adjustments.append(cardinality_adjustment)

    intent_spec = edge_intent_spec(current.semantic)
    if current.cardinality not in intent_spec.allowed_cardinalities:
        downgraded, adjustment = _downgrade_cardinality(
            fk=current,
            allowed=intent_spec.allowed_cardinalities,
            reason_code="intent_cardinality_not_allowed",
        )
        current = downgraded
        adjustments.append(adjustment)
    return current, tuple(adjustments)


def _preflight_temporal(
    fk: ForeignKeySpec,
    table_specs: Mapping[str, TableSpec],
) -> tuple[ForeignKeySpec, EdgeFeasibilityAdjustment | None]:
    if not fk.temporal:
        return fk, None
    child = table_specs[fk.child_table]
    parent = table_specs[fk.parent_table]
    if not child.has_timestamp or not parent.has_timestamp:
        updated = replace(fk, temporal=False)
        return updated, _adjustment(
            fk=fk,
            updated=updated,
            action="disable_temporal",
            reason_code="temporal_missing_timestamp",
        )
    if fk.semantic not in {"activity_refs_entity", "snapshot_refs_entity_or_activity"}:
        updated = replace(fk, temporal=False)
        return updated, _adjustment(
            fk=fk,
            updated=updated,
            action="disable_temporal",
            reason_code="temporal_intent_not_allowed",
        )
    return fk, None


def _preflight_cardinality(
    fk: ForeignKeySpec,
    table_specs: Mapping[str, TableSpec],
    timestamps: Mapping[str, NDArray[np.float64]],
) -> tuple[ForeignKeySpec, EdgeFeasibilityAdjustment | None]:
    if fk.existence != "mandatory":
        return fk, None

    child_count = table_specs[fk.child_table].row_count
    parent_count = table_specs[fk.parent_table].row_count
    if fk.cardinality == "one_to_one" and child_count > parent_count:
        return _downgrade_cardinality(
            fk=fk,
            allowed=edge_intent_spec(fk.semantic).allowed_cardinalities,
            reason_code="one_to_one_child_exceeds_parent",
        )
    if fk.capacity is not None and child_count > parent_count * fk.capacity:
        return _downgrade_cardinality(
            fk=fk,
            allowed=edge_intent_spec(fk.semantic).allowed_cardinalities,
            reason_code="capacity_child_exceeds_total_capacity",
        )
    if fk.temporal and not _temporal_has_candidate_for_every_child(fk, timestamps):
        return fk, _adjustment(
            fk=fk,
            updated=fk,
            action="route_requires_timestamp_repair",
            reason_code="temporal_candidate_gap",
        )
    return fk, None


def _downgrade_cardinality(
    fk: ForeignKeySpec,
    allowed: Sequence[ForeignKeyCardinality],
    reason_code: str,
) -> tuple[ForeignKeySpec, EdgeFeasibilityAdjustment]:
    for candidate in _CARDINALITY_DOWNGRADE_DAG[fk.cardinality]:
        if candidate not in allowed:
            continue
        if candidate != "many_to_one":
            continue
        updated = replace(
            fk,
            cardinality="many_to_one",
            capacity=None,
            mechanism=replace(fk.mechanism, capacity_mode="unbounded", capacity_k=None),
        )
        return updated, _adjustment(
            fk=fk,
            updated=updated,
            action="downgrade_cardinality",
            reason_code=reason_code,
        )
    raise SchemaFeasibilityError(
        f"edge {fk.key} with semantic={fk.semantic!r} and cardinality={fk.cardinality!r} "
        f"is infeasible; no explicit downgrade is allowed"
    )


def _temporal_has_candidate_for_every_child(
    fk: ForeignKeySpec,
    timestamps: Mapping[str, NDArray[np.float64]],
) -> bool:
    child_times = timestamps[fk.child_table]
    parent_times = timestamps[fk.parent_table]
    if len(parent_times) == 0:
        return False
    earliest_parent = float(np.min(parent_times))
    return bool(np.all(child_times >= earliest_parent - 1e-12))


def _adjustment(
    fk: ForeignKeySpec,
    updated: ForeignKeySpec,
    action: str,
    reason_code: str,
) -> EdgeFeasibilityAdjustment:
    return EdgeFeasibilityAdjustment(
        fk_key=fk.key,
        action=action,
        reason_code=reason_code,
        before=_edge_state(fk),
        after=_edge_state(updated),
    )


def _edge_state(fk: ForeignKeySpec) -> dict[str, object]:
    return {
        "semantic": fk.semantic,
        "cardinality": fk.cardinality,
        "nullable": fk.nullable,
        "capacity": fk.capacity,
        "temporal": fk.temporal,
        "existence": fk.existence,
        "capacity_mode": fk.mechanism.capacity_mode,
        "capacity_k": fk.mechanism.capacity_k,
    }
