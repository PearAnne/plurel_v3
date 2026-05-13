from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from plurel.topology_measure import ForeignKeyEdge, TopologyDatabase, TopologyTable

_TIME_COLUMN_NAMES = ("timestamp", "date", "datetime", "offerdate", "publish_time")
_MILLISECOND_TIME_NAMES = ("milliseconds", "millisecond", "lap_ms", "time_ms")
_LAP_DURATION_STR_RE = re.compile(r"^\d{1,3}:\d{2}\.\d{3}$")
_FKEY_PREFIX = "FK_"


def discover_ctu_database_names(ctu_root: Path) -> list[str]:
    ctu_root = ctu_root.expanduser()
    names = [
        path.name
        for path in sorted(ctu_root.iterdir())
        if path.is_dir() and _resolve_ctu_db_dir(path) is not None
    ]
    if not names:
        raise ValueError(f"No CTU databases found under {ctu_root}")
    return names


def load_ctu_database(ctu_root: Path, db_name: str) -> TopologyDatabase:
    db_dir = _resolve_ctu_db_dir(ctu_root.expanduser() / db_name)
    if db_dir is None:
        raise FileNotFoundError(f"CTU database '{db_name}' has no parquet tables under {ctu_root}")

    parquet_paths = sorted(db_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"CTU database '{db_name}' has no parquet tables in {db_dir}")

    tables: dict[str, TopologyTable] = {}
    for parquet_path in parquet_paths:
        table_name = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue
        tables[table_name] = TopologyTable(df=df, time_col=_infer_time_column(df))

    table_names = set(tables)
    foreign_keys: list[ForeignKeyEdge] = []
    for table_name, table in tables.items():
        for column in table.df.columns:
            parsed = parse_ctu_fkey_column(column, table_names)
            if parsed is None:
                continue
            parent_table, parent_pkey_col = parsed
            if parent_table not in tables:
                continue
            parent_df = tables[parent_table].df
            if parent_pkey_col not in parent_df.columns:
                if "__PK__" in parent_df.columns:
                    parent_pkey_col = "__PK__"
                else:
                    continue
            foreign_keys.append(
                ForeignKeyEdge(
                    child_table=table_name,
                    fkey_col=column,
                    parent_table=parent_table,
                    parent_pkey_col=parent_pkey_col,
                )
            )

    return TopologyDatabase(
        db_name=f"ctu-{db_name}",
        tables=tables,
        foreign_keys=sorted(foreign_keys, key=lambda edge: (edge.child_table, edge.fkey_col)),
    )


def parse_ctu_fkey_column(column: str, table_names: set[str]) -> tuple[str, str] | None:
    if not column.startswith(_FKEY_PREFIX):
        return None
    suffix = column[len(_FKEY_PREFIX) :]
    for parent_table in sorted(table_names, key=len, reverse=True):
        prefix = f"{parent_table}_"
        if suffix.startswith(prefix):
            return parent_table, suffix[len(prefix) :]
    return None


def _resolve_ctu_db_dir(db_path: Path) -> Path | None:
    db_subdir = db_path / "db"
    if db_subdir.is_dir() and any(db_subdir.glob("*.parquet")):
        return db_subdir
    if any(db_path.glob("*.parquet")):
        return db_path
    return None


def _looks_like_lap_duration_strings(series: pd.Series) -> bool:
    """Ergast ``lapTimes.time`` stores lap duration as ``M:SS.mmm``, not a clock time."""
    if series.empty:
        return False
    if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return bool(sample.str.fullmatch(_LAP_DURATION_STR_RE, na=False).mean() >= 0.65)


def _fraction_parseable_as_datetime(series: pd.Series, column_name: str) -> float:
    if series.empty:
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(series):
        return float(series.notna().mean())
    sample = series.dropna().head(8000)
    if sample.empty:
        return 0.0
    lower = column_name.lower()
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if lower in _MILLISECOND_TIME_NAMES or any(
            hint in lower for hint in ("epoch", "unix_ts", "unixtime")
        ):
            return 1.0
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return float(parsed.notna().mean())


def _infer_time_column(df: pd.DataFrame) -> str | None:
    """Pick a timeline column for temporal metrics; conservative to avoid bogus casts."""
    best_dt: tuple[float, str] | None = None
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_datetime64_any_dtype(series):
            continue
        frac = float(series.notna().mean())
        if frac < 0.5 or series.notna().sum() < 2:
            continue
        if best_dt is None or frac > best_dt[0]:
            best_dt = (frac, column)
    if best_dt is not None:
        return best_dt[1]

    for column in df.columns:
        lower = column.lower()
        if lower in _MILLISECOND_TIME_NAMES and pd.api.types.is_numeric_dtype(df[column]):
            if df[column].notna().sum() >= 2:
                return column

    candidates: list[str] = []
    for column in df.columns:
        if column in _TIME_COLUMN_NAMES:
            candidates.append(column)
    for column in df.columns:
        lowered = column.lower()
        if lowered in _TIME_COLUMN_NAMES or lowered.endswith("_date"):
            candidates.append(column)
        elif lowered.endswith("_time"):
            candidates.append(column)
        elif lowered == "time":
            candidates.append(column)

    seen: set[str] = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]

    for column in ordered:
        series = df[column]
        lowered = column.lower()
        if lowered == "time" or lowered.endswith("_time"):
            if _looks_like_lap_duration_strings(series):
                continue
        if _fraction_parseable_as_datetime(series, column) >= 0.5:
            return column

    for column in df.columns:
        if column in ordered:
            continue
        series = df[column]
        if _fraction_parseable_as_datetime(series, column) >= 0.9:
            return column

    return None


DBINFER_NONOVERLAPPING_DATASETS: tuple[str, ...] = (
    "seznam",
    "diginetica",
    "retailrocket",
    "avs",
    "outbrain-small",
)

DBINFER_FOREIGN_KEYS: dict[str, list[tuple[str, str, str, str]]] = {
    "seznam": [
        ("Dobito", "client_id", "Client", "client_id"),
        ("Probehnuto", "client_id", "Client", "client_id"),
        ("ProbehnutoMimoPenezenku", "client_id", "Client", "client_id"),
    ],
    "diginetica": [
        ("Click", "queryId", "Query", "queryId"),
        ("Click", "itemId", "Product", "itemId"),
        ("ProductNameToken", "itemId", "Product", "itemId"),
        ("Purchase", "itemId", "Product", "itemId"),
        ("QueryResult", "queryId", "Query", "queryId"),
        ("QueryResult", "itemId", "Product", "itemId"),
        ("QuerySearchstringToken", "queryId", "Query", "queryId"),
        ("View", "itemId", "Product", "itemId"),
    ],
    "retailrocket": [
        ("Category", "parentid", "Category", "categoryid"),
        ("ItemCategory", "category", "Category", "categoryid"),
    ],
    "avs": [
        ("History", "offer", "Offer", "offer"),
    ],
    "outbrain-small": [
        ("Click", "ad_id", "PromotedContent", "ad_id"),
        ("DocumentsCategory", "document_id", "DocumentsMeta", "document_id"),
        ("DocumentsEntity", "document_id", "DocumentsMeta", "document_id"),
        ("DocumentsTopic", "document_id", "DocumentsMeta", "document_id"),
        ("Event", "document_id", "DocumentsMeta", "document_id"),
        ("Pageview", "document_id", "DocumentsMeta", "document_id"),
        ("PromotedContent", "document_id", "DocumentsMeta", "document_id"),
    ],
}

DBINFER_TIME_COLUMNS: dict[str, dict[str, str]] = {
    "seznam": {
        "Dobito": "date",
        "Probehnuto": "date",
        "ProbehnutoMimoPenezenku": "date",
    },
    "diginetica": {
        "Click": "timestamp",
        "Purchase": "timestamp",
        "Query": "timestamp",
        "QueryResult": "timestamp",
        "View": "timestamp",
    },
    "retailrocket": {
        "ItemAvailability": "timestamp",
        "ItemCategory": "timestamp",
        "ItemProperty": "timestamp",
        "View": "timestamp",
    },
    "avs": {
        "History": "offerdate",
        "Transaction": "date",
    },
    "outbrain-small": {
        "Click": "timestamp",
        "Event": "timestamp",
        "Pageview": "timestamp",
    },
}


def load_dbinfer_database(data_root: Path, dataset_name: str) -> TopologyDatabase:
    if dataset_name not in DBINFER_NONOVERLAPPING_DATASETS:
        raise KeyError(
            f"Unknown 4DBInfer dataset '{dataset_name}'. "
            f"Expected one of: {', '.join(DBINFER_NONOVERLAPPING_DATASETS)}"
        )

    db_dir = data_root.expanduser() / f"dbinfer-{dataset_name}" / "db"
    if not db_dir.is_dir():
        raise FileNotFoundError(f"4DBInfer dataset directory not found: {db_dir}")

    tables: dict[str, TopologyTable] = {}
    for parquet_path in sorted(db_dir.glob("*.parquet")):
        table_name = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue
        time_col = DBINFER_TIME_COLUMNS.get(dataset_name, {}).get(table_name)
        if time_col is None:
            time_col = _infer_time_column(df)
        tables[table_name] = TopologyTable(df=df, time_col=time_col)

    foreign_keys: list[ForeignKeyEdge] = []
    for child_table, fkey_col, parent_table, parent_pkey_col in DBINFER_FOREIGN_KEYS[dataset_name]:
        if child_table not in tables or parent_table not in tables:
            continue
        if fkey_col not in tables[child_table].df.columns:
            continue
        if parent_pkey_col not in tables[parent_table].df.columns:
            continue
        foreign_keys.append(
            ForeignKeyEdge(
                child_table=child_table,
                fkey_col=fkey_col,
                parent_table=parent_table,
                parent_pkey_col=parent_pkey_col,
            )
        )

    return TopologyDatabase(
        db_name=f"dbinfer-{dataset_name}",
        tables=tables,
        foreign_keys=foreign_keys,
    )


def normalize_dbinfer_dataset_name(raw_name: str) -> str:
    name = raw_name.strip()
    if name.startswith("dbinfer-"):
        name = name[len("dbinfer-") :]
    if not re.fullmatch(r"[A-Za-z0-9-]+", name):
        raise ValueError(f"Invalid 4DBInfer dataset name: {raw_name!r}")
    return name
