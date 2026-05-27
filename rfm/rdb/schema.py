from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from rfm.rdb.config import RDBPriorConfig
from rfm.rdb.types import (
    CapacityMode,
    ColumnSpec,
    EdgeIntent,
    EdgeIntentSpec,
    FeatureColumnType,
    ForeignKeyCardinality,
    ForeignKeySpec,
    MechanismProfile,
    SchemaArchetype,
    SchemaGraph,
    TableRole,
    TableSpec,
)

_ARCHETYPE_ROLE_SEQUENCES: dict[SchemaArchetype, tuple[TableRole, ...]] = {
    "star": (
        "activity/event",
        "entity",
        "dimension/lookup",
        "dimension/lookup",
        "entity",
        "dimension/lookup",
    ),
    "snowflake": (
        "activity/event",
        "entity",
        "dimension/lookup",
        "dimension/lookup",
        "entity",
        "activity/event",
    ),
    "entity-event": (
        "entity",
        "entity",
        "activity/event",
        "activity/event",
        "dimension/lookup",
        "activity/event",
    ),
    "event-lookup": (
        "activity/event",
        "dimension/lookup",
        "dimension/lookup",
        "entity",
        "activity/event",
        "dimension/lookup",
    ),
    "many-to-many": (
        "entity",
        "entity",
        "bridge",
        "activity/event",
        "dimension/lookup",
        "activity/event",
    ),
    "temporal-history": (
        "entity",
        "activity/event",
        "snapshot/state",
        "dimension/lookup",
        "snapshot/state",
        "activity/event",
    ),
}


_INTENT_SPECS: dict[EdgeIntent, EdgeIntentSpec] = {
    "entity_belongs_to_lookup": EdgeIntentSpec(
        intent="entity_belongs_to_lookup",
        allowed_cardinalities=("many_to_one", "one_to_one", "optional"),
        default_temporal=False,
        coordination="independent",
    ),
    "activity_refs_entity": EdgeIntentSpec(
        intent="activity_refs_entity",
        allowed_cardinalities=(
            "many_to_one",
            "one_to_one",
            "capacity_limited",
            "optional",
            "multi_parent_member",
        ),
        default_temporal=True,
        coordination="independent",
    ),
    "activity_refs_dimension": EdgeIntentSpec(
        intent="activity_refs_dimension",
        allowed_cardinalities=("many_to_one", "optional", "multi_parent_member"),
        default_temporal=False,
        coordination="independent",
    ),
    "bridge_pairs_entities": EdgeIntentSpec(
        intent="bridge_pairs_entities",
        allowed_cardinalities=("multi_parent_member",),
        default_temporal=False,
        coordination="bridge_pair",
    ),
    "snapshot_refs_entity_or_activity": EdgeIntentSpec(
        intent="snapshot_refs_entity_or_activity",
        allowed_cardinalities=("many_to_one", "one_to_one", "optional"),
        default_temporal=True,
        coordination="independent",
    ),
}


def edge_intent_spec(intent: EdgeIntent) -> EdgeIntentSpec:
    return _INTENT_SPECS[intent]


class SchemaGrammar:
    def __init__(self, rng: np.random.Generator, config: RDBPriorConfig) -> None:
        self.rng = rng
        self.config = config
        self.grammar = config.role_grammar

    def sample(self) -> SchemaGraph:
        table_count = int(self.rng.integers(self.config.min_tables, self.config.max_tables + 1))
        archetype = self._sample_archetype()
        roles = self._roles_for_archetype(archetype, table_count)
        base_specs = self._sample_table_specs(roles)
        structural_edges = self._sample_archetype_edges(archetype, base_specs)
        edges, edge_intents = self._finalize_edges(base_specs, structural_edges)
        table_specs = self._with_foreign_key_columns(base_specs, edges)
        order = self._topological_order(table_specs, edges)
        return SchemaGraph(
            tables=table_specs,
            edges=tuple(edges),
            edge_intents=edge_intents,
            topological_order=order,
            archetype=archetype,
        )

    def _sample_archetype(self) -> SchemaArchetype:
        forced = self.config.schema_archetype.forced_archetype
        if forced is not None:
            if not self._archetype_enabled(forced):
                raise ValueError(
                    f"schema archetype {forced!r} is incompatible with the active configuration"
                )
            return forced
        candidates = [
            (archetype, weight)
            for archetype, weight in self.config.schema_archetype.distribution
            if weight > 0.0 and self._archetype_enabled(archetype)
        ]
        if not candidates:
            raise ValueError(
                "schema_archetype.distribution has no enabled positive-weight archetype"
            )
        weights = np.asarray([weight for _, weight in candidates], dtype=np.float64)
        index = int(self.rng.choice(len(candidates), p=weights / weights.sum()))
        return candidates[index][0]

    def _archetype_enabled(self, archetype: SchemaArchetype) -> bool:
        required_roles = set(_ARCHETYPE_ROLE_SEQUENCES[archetype][:4])
        if not required_roles.issubset(set(self.config.table_roles)):
            return False
        if archetype == "many-to-many":
            return self.config.enable_many_to_many_motif
        if archetype == "temporal-history":
            return self.config.enable_snapshot_tables
        return True

    def _roles_for_archetype(self, archetype: SchemaArchetype, table_count: int) -> list[TableRole]:
        sequence = _ARCHETYPE_ROLE_SEQUENCES[archetype]
        roles = list(sequence[: min(table_count, len(sequence))])
        while len(roles) < table_count:
            roles.append(sequence[4 + ((len(roles) - 4) % (len(sequence) - 4))])
        return roles

    def _sample_table_specs(self, roles: list[TableRole]) -> dict[str, TableSpec]:
        role_counts: dict[str, int] = {}
        specs: dict[str, TableSpec] = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
            name = f"{_role_prefix(role)}_{role_counts[role] - 1}"
            row_count = self._sample_row_count(role)
            primary_key = f"{name}_id"
            has_timestamp = self._sample_table_has_timestamp(role)
            timestamp_column = "timestamp" if has_timestamp else None
            feature_count = int(
                self.rng.integers(
                    self.config.min_features_per_table, self.config.max_features_per_table + 1
                )
            )
            columns = [
                ColumnSpec(name=primary_key, kind="primary_key", value_type="integer"),
            ]
            if timestamp_column is not None:
                columns.append(
                    ColumnSpec(name=timestamp_column, kind="timestamp", value_type="timestamp")
                )
            columns.extend(
                ColumnSpec(
                    name=f"{name}_f{idx}",
                    kind="feature",
                    value_type=self._sample_feature_type(),
                    nullable=False,
                )
                for idx in range(feature_count)
            )
            specs[name] = TableSpec(
                name=name,
                role=role,
                row_count=row_count,
                columns=tuple(columns),
                primary_key=primary_key,
                timestamp_column=timestamp_column,
                has_timestamp=has_timestamp,
            )
        return specs

    def _sample_table_has_timestamp(self, role: TableRole) -> bool:
        probability = self.grammar.timestamp_probability_by_role.get(role, 0.0)
        return bool(self.rng.random() < probability)

    def _sample_row_count(self, role: TableRole) -> int:
        base = int(
            self.rng.integers(self.config.min_rows_per_table, self.config.max_rows_per_table + 1)
        )
        if role == "dimension/lookup":
            return max(2, int(round(base * 0.45)))
        if role in ("activity/event", "bridge"):
            return max(2, int(round(base * 1.15)))
        return max(2, base)

    def _sample_feature_type(self) -> FeatureColumnType:
        return self.config.feature_types[int(self.rng.integers(0, len(self.config.feature_types)))]

    def _sample_archetype_edges(
        self,
        archetype: SchemaArchetype,
        table_specs: Mapping[str, TableSpec],
    ) -> list[tuple[str, str, EdgeIntent, str | None]]:
        edges: list[tuple[str, str, EdgeIntent, str | None]] = []
        by_role = {
            role: [spec for spec in table_specs.values() if spec.role == role]
            for role in self.config.table_roles
        }
        entities = by_role.get("entity", [])
        activities = by_role.get("activity/event", [])
        dimensions = by_role.get("dimension/lookup", [])
        bridges = by_role.get("bridge", [])
        snapshots = by_role.get("snapshot/state", [])

        if archetype == "star":
            self._connect_activity(activities[0], entities + dimensions, edges)
            for entity_idx, entity in enumerate(entities):
                if dimensions:
                    self._append_edge(
                        edges,
                        entity,
                        dimensions[entity_idx % len(dimensions)],
                        "entity_belongs_to_lookup",
                    )
        elif archetype == "snowflake":
            for dimension_idx, dimension in enumerate(dimensions):
                self._append_edge(
                    edges,
                    entities[dimension_idx % len(entities)],
                    dimension,
                    "entity_belongs_to_lookup",
                )
            for activity in activities:
                self._connect_activity(activity, entities[:1], edges)
        elif archetype == "entity-event":
            for activity in activities:
                self._connect_activity(activity, entities, edges)
            if dimensions:
                for entity in entities:
                    self._append_edge(edges, entity, dimensions[0], "entity_belongs_to_lookup")
        elif archetype == "event-lookup":
            for activity in activities:
                self._connect_activity(activity, dimensions + entities[:1], edges)
            if entities:
                self._append_edge(edges, entities[0], dimensions[0], "entity_belongs_to_lookup")
        elif archetype == "many-to-many":
            bridge = bridges[0]
            group = f"{bridge.name}_bridge"
            for entity in entities[:2]:
                self._append_edge(edges, bridge, entity, "bridge_pairs_entities", group)
            for activity in activities:
                self._connect_activity(activity, entities[:2], edges)
            if dimensions:
                for entity in entities:
                    self._append_edge(edges, entity, dimensions[0], "entity_belongs_to_lookup")
        elif archetype == "temporal-history":
            for activity in activities:
                self._connect_activity(activity, entities[:1], edges)
            if not activities:
                raise ValueError("temporal-history archetype requires at least one activity table")
            for snapshot_idx, snapshot in enumerate(snapshots):
                self._append_edge(
                    edges,
                    snapshot,
                    activities[snapshot_idx % len(activities)],
                    "snapshot_refs_entity_or_activity",
                )
            if dimensions:
                for entity in entities:
                    self._append_edge(edges, entity, dimensions[0], "entity_belongs_to_lookup")
        else:
            raise ValueError(f"unsupported schema archetype {archetype!r}")
        return edges

    def _connect_activity(
        self,
        child: TableSpec,
        parents: list[TableSpec],
        edges: list[tuple[str, str, EdgeIntent, str | None]],
    ) -> None:
        selected = parents[: self.config.max_foreign_keys_per_table]
        group = None
        if len(selected) > 1 and self.rng.random() < self.config.multi_parent_probability:
            group = f"{child.name}_parents"
        for parent in selected:
            intent: EdgeIntent = (
                "activity_refs_entity" if parent.role == "entity" else "activity_refs_dimension"
            )
            self._append_edge(edges, child, parent, intent, group)

    def _append_edge(
        self,
        edges: list[tuple[str, str, EdgeIntent, str | None]],
        child: TableSpec,
        parent: TableSpec,
        intent: EdgeIntent,
        group: str | None = None,
    ) -> None:
        edge = (child.name, parent.name, intent, group)
        if edge not in edges:
            edges.append(edge)

    def _finalize_edges(
        self,
        table_specs: Mapping[str, TableSpec],
        structural: list[tuple[str, str, EdgeIntent, str | None]],
    ) -> tuple[list[ForeignKeySpec], dict[str, EdgeIntentSpec]]:
        edges: list[ForeignKeySpec] = []
        edge_intents: dict[str, EdgeIntentSpec] = {}
        placeholder = _placeholder_mechanism(self.config.latent_dim)

        for child_name, parent_name, intent, group in structural:
            child = table_specs[child_name]
            parent = table_specs[parent_name]
            intent_spec = edge_intent_spec(intent)
            cardinality, capacity, capacity_mode = self._sample_cardinality(
                child, parent, intent_spec, group is not None
            )
            temporal = self._sample_temporal_fk(
                child=child, parent=parent, intent=intent, intent_spec=intent_spec
            )
            nullable = (
                cardinality == "optional"
                or self.rng.random() < self.config.optional_foreign_key_probability
            )
            if intent == "bridge_pairs_entities":
                nullable = False
            existence = "optional" if nullable else "mandatory"

            fk = ForeignKeySpec(
                child_table=child_name,
                child_column=f"{parent_name}_id",
                parent_table=parent_name,
                parent_column=parent.primary_key,
                cardinality=cardinality,
                nullable=nullable,
                capacity=capacity,
                temporal=temporal,
                mechanism=replace(
                    placeholder,
                    existence=existence,
                    capacity_mode=capacity_mode,
                    capacity_k=capacity,
                ),
                intent=intent,
                multi_parent_group=group,
                semantic=intent,
                existence=existence,
            )
            edges.append(fk)
            edge_intents[fk.key] = intent_spec
        return edges, edge_intents

    def _sample_temporal_fk(
        self,
        child: TableSpec,
        parent: TableSpec,
        intent: EdgeIntent,
        intent_spec: EdgeIntentSpec,
    ) -> bool:
        if not child.has_timestamp or not parent.has_timestamp:
            return False
        if intent not in {"activity_refs_entity", "snapshot_refs_entity_or_activity"}:
            return False
        return bool(
            intent_spec.default_temporal
            or self.rng.random() < self.config.temporal_foreign_key_probability
        )

    def _sample_cardinality(
        self,
        child: TableSpec,
        parent: TableSpec,
        intent_spec: EdgeIntentSpec,
        force_multi: bool,
    ) -> tuple[ForeignKeyCardinality, int | None, CapacityMode]:
        allowed = intent_spec.allowed_cardinalities
        if force_multi or intent_spec.coordination in ("joint_tuple", "bridge_pair"):
            return "multi_parent_member", None, "unbounded"

        if self.config.capacity_limited_probability >= 1.0 and "capacity_limited" in allowed:
            expected = child.row_count / max(float(parent.row_count), 1.0)
            capacity = max(1, int(np.ceil(expected * float(self.rng.uniform(1.2, 2.4)))))
            return "capacity_limited", capacity, "k_limited"
        if (
            self.config.one_to_one_probability >= 1.0
            and "one_to_one" in allowed
            and child.row_count <= parent.row_count
        ):
            return "one_to_one", 1, "one_to_one"
        if self.config.optional_foreign_key_probability >= 1.0 and "optional" in allowed:
            return "optional", None, "unbounded"

        candidates: list[tuple[ForeignKeyCardinality, int | None, CapacityMode, float]] = []
        if "capacity_limited" in allowed:
            expected = child.row_count / max(float(parent.row_count), 1.0)
            capacity = max(1, int(np.ceil(expected * float(self.rng.uniform(1.2, 2.4)))))
            candidates.append(
                (
                    "capacity_limited",
                    capacity,
                    "k_limited",
                    self.config.capacity_limited_probability,
                )
            )
        if "one_to_one" in allowed and child.row_count <= parent.row_count:
            candidates.append(("one_to_one", 1, "one_to_one", self.config.one_to_one_probability))
        if "optional" in allowed:
            candidates.append(
                ("optional", None, "unbounded", self.config.optional_foreign_key_probability)
            )
        if "many_to_one" in allowed:
            candidates.append(("many_to_one", None, "unbounded", 1.0))

        if len(candidates) == 0:
            cardinality = allowed[0]
            if cardinality == "capacity_limited":
                capacity = max(1, int(np.ceil(child.row_count / max(float(parent.row_count), 1.0))))
                return cardinality, capacity, "k_limited"
            if cardinality == "one_to_one":
                return cardinality, 1, "one_to_one"
            return cardinality, None, "unbounded"

        weights = np.array([weight for *_, weight in candidates], dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            choice = candidates[0]
        else:
            choice = candidates[int(self.rng.choice(len(candidates), p=weights / weights.sum()))]
        return choice[0], choice[1], choice[2]

    def _with_foreign_key_columns(
        self,
        table_specs: Mapping[str, TableSpec],
        foreign_keys: list[ForeignKeySpec],
    ) -> dict[str, TableSpec]:
        by_child: dict[str, list[ForeignKeySpec]] = {}
        for fk in foreign_keys:
            by_child.setdefault(fk.child_table, []).append(fk)

        specs: dict[str, TableSpec] = {}
        for table_name, spec in table_specs.items():
            fixed = [
                column for column in spec.columns if column.kind in ("primary_key", "timestamp")
            ]
            features = [column for column in spec.columns if column.kind == "feature"]
            fk_columns = [
                ColumnSpec(
                    name=fk.child_column,
                    kind="foreign_key",
                    value_type="integer",
                    nullable=fk.nullable,
                    source=f"{fk.parent_table}.{fk.parent_column}",
                )
                for fk in by_child.get(table_name, [])
            ]
            specs[table_name] = replace(spec, columns=tuple(fixed + fk_columns + features))
        return specs

    def _topological_order(
        self,
        table_specs: Mapping[str, TableSpec],
        foreign_keys: list[ForeignKeySpec],
    ) -> tuple[str, ...]:
        children: dict[str, set[str]] = {name: set() for name in table_specs}
        indegree: dict[str, int] = {name: 0 for name in table_specs}
        for fk in foreign_keys:
            if fk.parent_table not in children:
                continue
            if fk.child_table not in indegree:
                continue
            if fk.child_table not in children[fk.parent_table]:
                children[fk.parent_table].add(fk.child_table)
                indegree[fk.child_table] += 1

        queue = sorted(name for name, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in sorted(children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(table_specs):
            raise ValueError("schema graph contains a cycle")
        return tuple(order)


def _role_prefix(role: TableRole) -> str:
    if role == "activity/event":
        return "activity"
    if role == "dimension/lookup":
        return "dimension"
    if role == "snapshot/state":
        return "snapshot"
    return role


def _placeholder_mechanism(latent_dim: int) -> MechanismProfile:
    return MechanismProfile(
        existence="mandatory",
        attachment="uniform",
        coordination="independent",
        field_weights=tuple(0.0 for _ in range(latent_dim)),
        temperature=1.0,
        capacity_mode="unbounded",
        existence_latent_weight=tuple(0.0 for _ in range(latent_dim)),
    )
