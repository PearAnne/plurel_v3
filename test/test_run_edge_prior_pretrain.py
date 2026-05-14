from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_edge_prior_pretrain as runner


def _write_manifest(path: Path) -> None:
    payload = {
        "stages": ["gen", "pre"],
        "cohorts": [
            {
                "cohort": "G0_hsbm",
                "seeds": [16000, 16001, 16002],
                "num_dbs": 3,
                "config": {},
                "config_hash": "a",
                "description": "baseline",
            },
            {
                "cohort": "G7_realistic_mix",
                "seeds": [16000, 16001, 16002],
                "num_dbs": 3,
                "config": {},
                "config_hash": "b",
                "description": "candidate",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_task_bundle(
    train_db_names: list[str],
    test_db_names: list[str],
    cache_root: Path,
) -> dict[str, list[tuple[str, str, str, list[str]]]]:
    return {
        "train_autocomplete_clf_tasks": [
            (train_db_names[0], "table", "feature_bool", []),
        ],
        "train_autocomplete_reg_tasks": [
            (train_db_names[0], "table", "feature_float", []),
        ],
        "test_autocomplete_clf_tasks": [
            (test_db_names[0], "table", "feature_bool", []),
        ],
        "test_autocomplete_reg_tasks": [],
    }


def test_build_run_specs_uses_manifest_cohort_db_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(
        runner,
        "generate_rel_synthetic_tasks_from_db_names",
        _fake_task_bundle,
    )

    specs = runner.build_run_specs(
        manifest_path=manifest_path,
        cohorts=["G7_realistic_mix"],
        num_train_dbs=2,
        num_test_dbs=1,
        max_steps=101,
        max_bfs_width=128,
        cache_root=tmp_path / "relbench",
        save_root=tmp_path / "runs",
        seed=0,
        batch_size=128,
        eval_batch_size=128,
        ctx_len=1024,
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.test_db_names == ["rel-synthetic-G7_realistic_mix-16000"]
    assert spec.train_db_names == [
        "rel-synthetic-G7_realistic_mix-16001",
        "rel-synthetic-G7_realistic_mix-16002",
    ]
    assert spec.rt_kwargs["max_steps"] == 101
    assert spec.rt_kwargs["max_bfs_width"] == 128
    assert spec.rt_kwargs["cohort"] == "G7_realistic_mix"
    assert str(spec.save_ckpt_dir).endswith(
        "G7_realistic_mix/train2_test1_steps101_bfs128_bs128_ctx1024_seed0"
    )


def test_build_run_specs_rejects_unknown_cohort(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    with pytest.raises(ValueError, match="Unknown cohorts"):
        runner.build_run_specs(
            manifest_path=manifest_path,
            cohorts=["G9_missing"],
            num_train_dbs=1,
            num_test_dbs=1,
            max_steps=101,
            max_bfs_width=128,
            cache_root=tmp_path,
            save_root=tmp_path,
            seed=0,
            batch_size=128,
            eval_batch_size=128,
            ctx_len=1024,
        )


def test_run_specs_dry_run_does_not_call_rt_main(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = runner.CohortRunSpec(
        cohort="G0_hsbm",
        train_db_names=["rel-synthetic-G0_hsbm-16001"],
        test_db_names=["rel-synthetic-G0_hsbm-16000"],
        save_ckpt_dir=tmp_path / "ckpt",
        rt_kwargs={
            "train_tasks": [],
            "eval_tasks": [],
            "max_steps": 101,
            "max_bfs_width": 128,
        },
    )

    runner.run_specs(specs=[spec], pre_root=tmp_path / "pre", dry_run=True)

    assert "cohort=G0_hsbm" in capsys.readouterr().out
