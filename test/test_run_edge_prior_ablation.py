from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ablation_manifest import build_edge_prior_cohorts, summarize_config
from scripts.run_edge_prior_ablation import (
    build_pretrain_commands,
    build_seeds_by_cohort,
    parse_stages,
    run,
    run_preprocess_stage,
)


def test_parse_stages_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stages"):
        parse_stages("gen,missing")


def test_build_seeds_by_cohort_pairs_schema_seeds() -> None:
    result = build_seeds_by_cohort(
        cohort_names=["G0_hsbm", "G7_realistic_mix"],
        num_dbs_per_cohort=3,
        base_seed=16000,
        paired_seeds=True,
    )

    assert result["G0_hsbm"] == [16000, 16001, 16002]
    assert result["G7_realistic_mix"] == [16000, 16001, 16002]


def test_g7_db_uses_db_level_realistic_mix_prior() -> None:
    cohorts = build_edge_prior_cohorts()
    g7_edge = summarize_config(cohorts["G7_realistic_mix"].config)
    g7_db = summarize_config(cohorts["G7_db"].config)

    assert g7_db["edge_prior_assignment_strategy"] == "db_level"
    assert g7_edge["edge_prior_assignment_strategy"] == "edge_level_uniform"
    assert g7_db["topology_prior_choices"] == g7_edge["topology_prior_choices"]
    assert g7_db["chung_lu_gamma_choices"] == g7_edge["chung_lu_gamma_choices"]
    assert g7_db["dcsbm_theta_alpha_choices"] == g7_edge["dcsbm_theta_alpha_choices"]
    assert g7_db["dcsbm_theta_beta_choices"] == g7_edge["dcsbm_theta_beta_choices"]
    assert g7_db["tpa_alpha_choices"] == g7_edge["tpa_alpha_choices"]
    assert g7_db["edge_prior_null_rate_choices"] == [0.0, 0.0]


def test_build_pretrain_commands_points_to_cohort_aware_runner(tmp_path: Path) -> None:
    commands = build_pretrain_commands(
        cohort_names=["G0_hsbm", "G7_realistic_mix"],
        output_dir=tmp_path,
    )

    assert len(commands) == 2
    assert "scripts/run_edge_prior_pretrain.py" in commands[0]
    assert f"--manifest {tmp_path / 'manifest.json'}" in commands[0]
    assert "--cohorts G0_hsbm" in commands[0]
    assert "--dry_run" in commands[0]


def test_run_rejects_unknown_selected_cohort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown cohorts"):
        run(
            num_dbs_per_cohort=1,
            base_seed=16000,
            num_proc=1,
            stages=[],
            output_dir=tmp_path,
            cache_root=tmp_path,
            pre_root=tmp_path,
            selected_cohorts=["G9_missing"],
        )


def test_run_can_skip_manifest_for_preprocess_only_rerun(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("existing manifest", encoding="utf-8")

    run(
        num_dbs_per_cohort=1,
        base_seed=16000,
        num_proc=1,
        stages=[],
        output_dir=tmp_path,
        cache_root=tmp_path,
        pre_root=tmp_path,
        selected_cohorts=["G0_hsbm"],
        write_manifest_file=False,
    )

    assert manifest_path.read_text(encoding="utf-8") == "existing manifest"


def test_run_preprocess_stage_skips_existing_embedding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    db_name = "rel-synthetic-G0_hsbm-16000"
    pre_root = tmp_path / "pre"
    embedding_dir = pre_root / db_name
    embedding_dir.mkdir(parents=True)
    (embedding_dir / "text_emb_all-MiniLM-L12-v2.bin").write_bytes(b"existing")
    monkeypatch.setattr("scripts.run_edge_prior_ablation.subprocess.run", fake_run)

    run_preprocess_stage(
        db_names=[db_name],
        stage="embed",
        cache_root=tmp_path / "relbench",
        pre_root=pre_root,
    )

    assert calls == []
