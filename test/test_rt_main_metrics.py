from __future__ import annotations

from rt.main import EvalMetric, is_better_metric


def test_metric_comparison_uses_metric_direction() -> None:
    auc_metric = EvalMetric(
        split="val",
        db_name="rel-f1",
        table_name="driver-dnf",
        metric_name="auc",
        metric_value=0.7,
        higher_is_better=True,
    )
    loss_metric = EvalMetric(
        split="val",
        db_name="relbench-loss",
        table_name="",
        metric_name="loss",
        metric_value=0.7,
        higher_is_better=False,
    )

    assert is_better_metric(metric=auc_metric, best_value=0.6)
    assert not is_better_metric(metric=auc_metric, best_value=0.8)
    assert is_better_metric(metric=loss_metric, best_value=0.8)
    assert not is_better_metric(metric=loss_metric, best_value=0.6)
