from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from plurel import Config, RFMSyntheticDataset, SyntheticDataset
from plurel.utils import set_random_seed

TaskSpec = tuple[str, str, str, list[str]]
SyntheticTaskBundle = dict[str, list[TaskSpec]]
SyntheticBackend = Literal["plurel", "rfm"]


def is_valid_db(db: Any) -> bool:
    # invalid if any table contains > 5 fk cols
    valid = True
    for table_name, table in db.table_dict.items():
        if len(list(table.fkey_col_to_pkey_table.keys())) > 5:
            valid = False
    return valid


def get_tasks_info(db: Any, db_name: str, table_name: str) -> dict[str, list[TaskSpec]]:
    clf_tasks = []
    reg_tasks = []
    table = db.table_dict[table_name]
    for col_name, data_type in table.df.dtypes.items():
        assert len(table.fkey_col_to_pkey_table.keys()) <= 5
        if "feature" not in col_name:
            continue
        task_ = (db_name, table_name, col_name, [])
        if pd.api.types.is_bool_dtype(data_type):
            clf_tasks.append(task_)
        elif pd.api.types.is_float_dtype(data_type) or pd.api.types.is_integer_dtype(data_type):
            reg_tasks.append(task_)
    return {"clf": clf_tasks, "reg": reg_tasks}


def get_clf_reg_tasks_for_db_names(
    db_names: list[str],
    cache_root: Path,
    max_db_count: int | None = None,
    per_db_task_limit: int | None = None,
    backend: SyntheticBackend = "plurel",
) -> tuple[list[TaskSpec], list[TaskSpec]]:
    all_db_clf_tasks = []
    all_db_reg_tasks = []
    db_count = 0
    for db_name in db_names:
        seed = _seed_from_rel_synthetic_db_name(db_name)
        dataset = _make_synthetic_dataset(
            db_name=db_name,
            seed=seed,
            cache_root=cache_root,
            backend=backend,
        )
        db = dataset.get_db()
        if not is_valid_db(db):
            print(f"invalid db: {db_name}")
            continue
        db_clf_tasks = []
        db_reg_tasks = []
        for table_name in sorted(list(db.table_dict.keys())):
            tasks_info = get_tasks_info(db=db, db_name=db_name, table_name=table_name)
            db_clf_tasks.extend(tasks_info["clf"])
            db_reg_tasks.extend(tasks_info["reg"])

        set_random_seed(0)
        np.random.shuffle(db_clf_tasks)
        np.random.shuffle(db_reg_tasks)
        db_clf_tasks = db_clf_tasks[: per_db_task_limit // 2] if per_db_task_limit else db_clf_tasks
        db_reg_tasks = db_reg_tasks[: per_db_task_limit // 2] if per_db_task_limit else db_reg_tasks

        all_db_clf_tasks.extend(db_clf_tasks)
        all_db_reg_tasks.extend(db_reg_tasks)

        db_count += 1
        if max_db_count is not None and db_count == max_db_count:
            print(
                f"------------------------------------ got desired db_count: {db_count} ------------------------------------"
            )
            break
    if max_db_count is not None and db_count != max_db_count:
        raise ValueError(f"required {max_db_count} dbs, got only {db_count}")

    return all_db_clf_tasks, all_db_reg_tasks


def _seed_from_rel_synthetic_db_name(db_name: str) -> int:
    try:
        return int(db_name.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot infer seed from synthetic DB name: {db_name}") from exc


def _make_synthetic_dataset(
    db_name: str,
    seed: int,
    cache_root: Path,
    backend: SyntheticBackend,
) -> Any:
    cache_dir = cache_root.expanduser() / db_name
    if backend == "plurel":
        return SyntheticDataset(
            seed=seed,
            config=Config(cache_dir=cache_dir),
        )
    if backend == "rfm":
        return RFMSyntheticDataset(seed=seed, cache_dir=cache_dir)
    raise ValueError(f"unsupported synthetic backend: {backend}")


def _synthetic_db_name(seed: int, backend: SyntheticBackend) -> str:
    if backend == "plurel":
        return f"rel-synthetic-{seed}"
    if backend == "rfm":
        return f"rel-synthetic-rfm-{seed}"
    raise ValueError(f"unsupported synthetic backend: {backend}")


def get_clf_reg_tasks(
    seeds: list[int],
    max_db_count: int,
    per_db_task_limit: int | None = None,
    backend: SyntheticBackend = "plurel",
) -> tuple[list[TaskSpec], list[TaskSpec]]:
    db_names = [_synthetic_db_name(seed=seed, backend=backend) for seed in seeds]
    return get_clf_reg_tasks_for_db_names(
        db_names=db_names,
        cache_root=Path("~/.cache/relbench"),
        max_db_count=max_db_count,
        per_db_task_limit=per_db_task_limit,
        backend=backend,
    )


def generate_rel_synthetic_tasks_from_db_names(
    train_db_names: list[str],
    test_db_names: list[str],
    cache_root: Path,
    skip_reg_tasks: bool = False,
    skip_clf_tasks: bool = False,
    backend: SyntheticBackend = "plurel",
) -> SyntheticTaskBundle:
    set_random_seed(0)
    test_autocomplete_clf_tasks, test_autocomplete_reg_tasks = get_clf_reg_tasks_for_db_names(
        db_names=test_db_names,
        cache_root=cache_root,
        max_db_count=len(test_db_names),
        per_db_task_limit=10,
        backend=backend,
    )
    train_autocomplete_clf_tasks, train_autocomplete_reg_tasks = get_clf_reg_tasks_for_db_names(
        db_names=train_db_names,
        cache_root=cache_root,
        max_db_count=len(train_db_names),
        backend=backend,
    )

    if skip_clf_tasks:
        train_autocomplete_clf_tasks = []
        test_autocomplete_clf_tasks = []
    if skip_reg_tasks:
        train_autocomplete_reg_tasks = []
        test_autocomplete_reg_tasks = []

    print(f"len(train_autocomplete_clf_tasks): {len(train_autocomplete_clf_tasks)}")
    print(f"len(train_autocomplete_reg_tasks): {len(train_autocomplete_reg_tasks)}")
    print(f"len(test_autocomplete_clf_tasks): {len(test_autocomplete_clf_tasks)}")
    print(f"len(test_autocomplete_reg_tasks): {len(test_autocomplete_reg_tasks)}")

    return {
        "train_autocomplete_clf_tasks": train_autocomplete_clf_tasks,
        "train_autocomplete_reg_tasks": train_autocomplete_reg_tasks,
        "test_autocomplete_clf_tasks": test_autocomplete_clf_tasks,
        "test_autocomplete_reg_tasks": test_autocomplete_reg_tasks,
    }


def generate_rel_synthetic_tasks(
    offset: int,
    num_dbs: int,
    num_train_dbs: int,
    num_test_dbs: int,
    skip_reg_tasks: bool = False,
    skip_clf_tasks: bool = False,
    backend: SyntheticBackend = "plurel",
) -> SyntheticTaskBundle:
    set_random_seed(0)
    seeds = [idx + offset for idx in range(num_dbs)]

    train_autocomplete_clf_tasks = []
    train_autocomplete_reg_tasks = []
    test_autocomplete_clf_tasks = []
    test_autocomplete_reg_tasks = []

    # have a collection for test dbs (2* is just a buffer)
    test_seeds = seeds[: 2 * num_test_dbs]
    train_seeds = seeds[2 * num_test_dbs :]

    test_autocomplete_clf_tasks, test_autocomplete_reg_tasks = get_clf_reg_tasks(
        seeds=test_seeds,
        max_db_count=num_test_dbs,
        per_db_task_limit=10,
        backend=backend,
    )

    train_autocomplete_clf_tasks, train_autocomplete_reg_tasks = get_clf_reg_tasks(
        seeds=train_seeds,
        max_db_count=num_train_dbs,
        backend=backend,
    )

    if skip_clf_tasks:
        train_autocomplete_clf_tasks = []
        test_autocomplete_clf_tasks = []
    if skip_reg_tasks:
        train_autocomplete_reg_tasks = []
        test_autocomplete_reg_tasks = []

    print(f"len(train_autocomplete_clf_tasks): {len(train_autocomplete_clf_tasks)}")
    print(f"len(train_autocomplete_reg_tasks): {len(train_autocomplete_reg_tasks)}")
    print(f"len(test_autocomplete_clf_tasks): {len(test_autocomplete_clf_tasks)}")
    print(f"len(test_autocomplete_reg_tasks): {len(test_autocomplete_reg_tasks)}")

    return {
        "train_autocomplete_clf_tasks": train_autocomplete_clf_tasks,
        "train_autocomplete_reg_tasks": train_autocomplete_reg_tasks,
        "test_autocomplete_clf_tasks": test_autocomplete_clf_tasks,
        "test_autocomplete_reg_tasks": test_autocomplete_reg_tasks,
    }


# tuples are (database, table, target column, leakage columns)

forecast_clf_tasks = [
    ("rel-amazon", "user-churn", "churn", []),
    ("rel-hm", "user-churn", "churn", []),
    ("rel-stack", "user-badge", "WillGetBadge", []),
    ("rel-amazon", "item-churn", "churn", []),
    ("rel-stack", "user-engagement", "contribution", []),
    ("rel-avito", "user-visits", "num_click", []),
    ("rel-avito", "user-clicks", "num_click", []),
    ("rel-event", "user-ignore", "target", []),
    ("rel-trial", "study-outcome", "outcome", []),
    ("rel-f1", "driver-dnf", "did_not_finish", []),
    ("rel-event", "user-repeat", "target", []),
    ("rel-f1", "driver-top3", "qualifying", []),
]

forecast_reg_tasks = [
    ("rel-hm", "item-sales", "sales", []),
    ("rel-amazon", "user-ltv", "ltv", []),
    ("rel-amazon", "item-ltv", "ltv", []),
    ("rel-stack", "post-votes", "popularity", []),
    ("rel-trial", "site-success", "success_rate", []),
    ("rel-trial", "study-adverse", "num_of_adverse_events", []),
    ("rel-event", "user-attendance", "target", []),
    ("rel-f1", "driver-position", "position", []),
    ("rel-avito", "ad-ctr", "num_click", []),
]

autocomplete_clf_tasks = [
    ("rel-avito", "SearchInfo", "IsUserLoggedOn", []),
    ("rel-stack", "postLinks", "LinkTypeId", []),
    ("rel-amazon", "review", "verified", []),
    ("rel-trial", "studies", "has_dmc", []),
    (
        "rel-trial",
        "eligibilities",
        "adult",
        [
            "child",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    (
        "rel-trial",
        "eligibilities",
        "child",
        [
            "adult",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    ("rel-event", "event_interest", "not_interested", ["interested"]),
]

autocomplete_reg_tasks = [
    ("rel-amazon", "review", "rating", ["review_text", "summary"]),
    (
        "rel-f1",
        "results",
        "position",
        [
            "statusId",
            "positionOrder",
            "points",
            "laps",
            "milliseconds",
            "fastestLap",
            "rank",
        ],
    ),
    ("rel-f1", "qualifying", "position", []),
    ("rel-trial", "studies", "enrollment", []),
    ("rel-f1", "constructor_results", "points", []),
    ("rel-f1", "constructor_standings", "position", ["wins", "points"]),
    ("rel-hm", "transactions", "price", []),
    ("rel-event", "users", "birthyear", []),
]

all_tasks = (
    forecast_clf_tasks + forecast_reg_tasks + autocomplete_clf_tasks + autocomplete_reg_tasks
)

forecast_tasks = forecast_clf_tasks + forecast_reg_tasks

all_dbs = [
    "rel-amazon",
    "rel-hm",
    "rel-stack",
    "rel-avito",
    "rel-event",
    "rel-trial",
    "rel-f1",
]
