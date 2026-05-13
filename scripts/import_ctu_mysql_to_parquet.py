from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MySqlConfig:
    host: str
    port: int
    user: str
    password: str


@dataclass(frozen=True)
class ForeignKeySpec:
    table_name: str
    constraint_name: str
    child_columns: tuple[str, ...]
    parent_table: str
    parent_columns: tuple[str, ...]


def run_mysql_query(config: MySqlConfig, query: str) -> pd.DataFrame:
    cmd = mysql_cmd(config) + ["--batch", "--raw", "--quick", "-e", query]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"mysql exited with {result.returncode}")
    if not result.stdout.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(result.stdout), sep="\t", dtype=str, keep_default_na=False)


def read_mysql_table(
    config: MySqlConfig, database: str, table_name: str, order_by: list[str]
) -> pd.DataFrame:
    order_clause = ", ".join(quote_identifier(column) for column in order_by)
    query = f"SELECT * FROM {quote_identifier(database)}.{quote_identifier(table_name)}"
    if order_clause:
        query = f"{query} ORDER BY {order_clause}"

    cmd = mysql_cmd(config) + [
        "--batch",
        "--raw",
        "--quick",
        "--database",
        database,
        "-e",
        query,
    ]
    LOGGER.info("reading %s.%s", database, table_name)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("mysql stdout was not captured")
    try:
        df = pd.read_csv(
            proc.stdout,
            sep="\t",
            na_values=["NULL"],
            keep_default_na=True,
            low_memory=False,
        )
    finally:
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(stderr.strip() or f"mysql exited with {returncode}")
    return df


def mysql_cmd(config: MySqlConfig) -> list[str]:
    return [
        "mysql",
        "-h",
        config.host,
        "-P",
        str(config.port),
        "-u",
        config.user,
        f"-p{config.password}",
        "--connect-timeout=30",
        "--default-character-set=utf8mb4",
    ]


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def load_table_names(config: MySqlConfig, database: str) -> list[str]:
    query = f"""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = {sql_literal(database)}
  AND table_type = 'BASE TABLE'
ORDER BY table_name
"""
    df = run_mysql_query(config, query)
    return df["table_name"].tolist()


def load_primary_keys(config: MySqlConfig, database: str) -> dict[str, list[str]]:
    query = f"""
SELECT table_name, column_name, ordinal_position
FROM information_schema.key_column_usage
WHERE constraint_schema = {sql_literal(database)}
  AND constraint_name = 'PRIMARY'
ORDER BY table_name, ordinal_position
"""
    df = run_mysql_query(config, query)
    primary_keys: dict[str, list[str]] = {}
    for table_name, group in df.groupby("table_name", sort=False):
        ordered = group.sort_values("ordinal_position", key=lambda s: s.astype(int))
        primary_keys[str(table_name)] = ordered["column_name"].tolist()
    return primary_keys


def load_foreign_keys(config: MySqlConfig, database: str) -> list[ForeignKeySpec]:
    query = f"""
SELECT
  table_name,
  constraint_name,
  column_name,
  referenced_table_name,
  referenced_column_name,
  ordinal_position
FROM information_schema.key_column_usage
WHERE constraint_schema = {sql_literal(database)}
  AND referenced_table_name IS NOT NULL
ORDER BY table_name, constraint_name, ordinal_position
"""
    df = run_mysql_query(config, query)
    specs: list[ForeignKeySpec] = []
    if df.empty:
        return specs

    for (table_name, constraint_name), group in df.groupby(
        ["table_name", "constraint_name"], sort=False
    ):
        ordered = group.sort_values("ordinal_position", key=lambda s: s.astype(int))
        parent_tables = ordered["referenced_table_name"].unique()
        if len(parent_tables) != 1:
            raise ValueError(f"FK {database}.{table_name}.{constraint_name} has multiple parents")
        specs.append(
            ForeignKeySpec(
                table_name=str(table_name),
                constraint_name=str(constraint_name),
                child_columns=tuple(ordered["column_name"].tolist()),
                parent_table=str(parent_tables[0]),
                parent_columns=tuple(ordered["referenced_column_name"].tolist()),
            )
        )
    return specs


def sql_literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_parent_maps(
    config: MySqlConfig,
    database: str,
    table_names: list[str],
    primary_keys: dict[str, list[str]],
) -> dict[str, dict[Any, int]]:
    parent_maps: dict[str, dict[Any, int]] = {}
    for table_name in table_names:
        pk_columns = primary_keys.get(table_name)
        if not pk_columns:
            raise ValueError(f"{database}.{table_name} has no primary key")

        select_cols = ", ".join(quote_identifier(column) for column in pk_columns)
        order_cols = ", ".join(quote_identifier(column) for column in pk_columns)
        query = (
            f"SELECT {select_cols} FROM {quote_identifier(database)}.{quote_identifier(table_name)} "
            f"ORDER BY {order_cols}"
        )
        df = run_mysql_query(config, query)
        parent_maps[table_name] = make_key_to_row_index(df, pk_columns)
        LOGGER.info("indexed %s.%s primary key rows=%d", database, table_name, len(df))
    return parent_maps


def make_key_to_row_index(df: pd.DataFrame, columns: list[str]) -> dict[Any, int]:
    if len(columns) == 1:
        return {value: int(index) for index, value in enumerate(df[columns[0]].tolist())}
    return {
        tuple(values): int(index)
        for index, values in enumerate(df.loc[:, columns].itertuples(index=False, name=None))
    }


def add_primary_key_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "__PK__", np.arange(len(out), dtype=np.int64))
    return out


def add_foreign_key_columns(
    df: pd.DataFrame,
    fk_specs: list[ForeignKeySpec],
    parent_maps: dict[str, dict[Any, int]],
) -> pd.DataFrame:
    out = df.copy()
    for fk in fk_specs:
        mapping = parent_maps[fk.parent_table]
        fk_col_name = f"FK_{fk.parent_table}_{'_'.join(fk.child_columns)}"
        if len(fk.child_columns) == 1:
            mapped = out[fk.child_columns[0]].astype(str).map(mapping)
        else:
            keys = out.loc[:, list(fk.child_columns)].astype(str).itertuples(index=False, name=None)
            mapped = pd.Series((mapping.get(tuple(key)) for key in keys), index=out.index)
        if mapped.isna().any():
            out[fk_col_name] = mapped.astype("Int64")
        else:
            out[fk_col_name] = mapped.astype(np.int64)
        missing = int(out[fk_col_name].isna().sum())
        if missing:
            LOGGER.warning("%s missing parent rows for %d child rows", fk_col_name, missing)
    return out


def import_database(
    config: MySqlConfig,
    database: str,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    table_names = load_table_names(config, database)
    primary_keys = load_primary_keys(config, database)
    foreign_keys = load_foreign_keys(config, database)
    fks_by_table: dict[str, list[ForeignKeySpec]] = {}
    for fk in foreign_keys:
        fks_by_table.setdefault(fk.table_name, []).append(fk)

    LOGGER.info(
        "%s: tables=%d primary_key_tables=%d foreign_keys=%d",
        database,
        len(table_names),
        len(primary_keys),
        len(foreign_keys),
    )
    for fk in foreign_keys:
        LOGGER.info(
            "%s FK %s.%s -> %s.%s",
            database,
            fk.table_name,
            ",".join(fk.child_columns),
            fk.parent_table,
            ",".join(fk.parent_columns),
        )
    if dry_run:
        return

    target_dir = output_root.expanduser() / database
    tmp_dir = output_root.expanduser() / f".{database}.tmp-{os.getpid()}"
    db_tmp_dir = tmp_dir / "db"
    if tmp_dir.exists():
        raise FileExistsError(f"Temporary directory already exists: {tmp_dir}")
    if target_dir.exists() and any(target_dir.rglob("*")) and not overwrite:
        raise FileExistsError(f"Target is not empty; pass --overwrite to replace: {target_dir}")

    db_tmp_dir.mkdir(parents=True)
    try:
        parent_maps = build_parent_maps(config, database, table_names, primary_keys)
        for table_name in table_names:
            df = read_mysql_table(config, database, table_name, primary_keys[table_name])
            df = add_primary_key_column(df)
            df = add_foreign_key_columns(df, fks_by_table.get(table_name, []), parent_maps)
            output_path = db_tmp_dir / f"{table_name}.parquet"
            df.to_parquet(output_path, index=False)
            LOGGER.info("wrote %s rows=%d columns=%d", output_path, len(df), len(df.columns))

        if target_dir.exists():
            if any(target_dir.rglob("*")):
                if not overwrite:
                    raise FileExistsError(f"Target became non-empty: {target_dir}")
                shutil.rmtree(target_dir)
            else:
                target_dir.rmdir()
        tmp_dir.rename(target_dir)
        LOGGER.info("installed %s", target_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import CTU MariaDB databases into the local CTU parquet mirror."
    )
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument(
        "--output_root", type=Path, default=Path("/local/lzd/plurel_runtime/relbench/ctu")
    )
    parser.add_argument("--host", default="relational.fel.cvut.cz")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="guest")
    parser.add_argument("--password", default="ctu-relational")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = MySqlConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    for database in args.databases:
        import_database(
            config=config,
            database=database,
            output_root=args.output_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
