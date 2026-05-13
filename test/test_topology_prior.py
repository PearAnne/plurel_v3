from dataclasses import asdict

import numpy as np
import pytest

from plurel.topology_metrics import compute_edge_metrics
from plurel.topology_prior import (
    TOPOLOGY_PRIOR_REGISTRY,
    ChungLuSampler,
    DCSBMSampler,
    EdgePriorSpec,
    ErdosRenyiSampler,
    HSBMSampler,
    StructuralNullDecorator,
    TPASampler,
)


@pytest.mark.parametrize(
    ("sampler", "params"),
    [
        (HSBMSampler(), {"hierarchy_a": [2], "hierarchy_b": [2]}),
        (ErdosRenyiSampler(), {}),
        (ChungLuSampler(), {"gamma": 2.0}),
        (
            DCSBMSampler(),
            {
                "hierarchy_a": [2],
                "hierarchy_b": [2],
                "theta_alpha": 2.0,
                "theta_beta": 5.0,
            },
        ),
        (TPASampler(), {"alpha": 1.0, "beta": 0.0}),
    ],
)
def test_sampler_outputs_have_valid_shape_and_range(sampler, params):
    rng = np.random.default_rng(0)

    parent_idx, null_mask = sampler.sample(
        size_a=16,
        size_b=64,
        params=params,
        rng=rng,
        child_timestamps=np.arange(64),
    )

    assert parent_idx.shape == (64,)
    assert parent_idx.dtype == np.int64
    assert parent_idx.min() >= 0
    assert parent_idx.max() < 16
    assert null_mask is None or null_mask.shape == (64,)


def test_chung_lu_has_more_concentrated_fanout_than_uniform_assignments():
    rng = np.random.default_rng(0)
    parent_idx, _ = ChungLuSampler().sample(
        size_a=1000,
        size_b=10000,
        params={"gamma": 2.0},
        rng=rng,
    )
    uniform_idx = np.arange(1000).repeat(10)

    chung_lu_metrics = compute_edge_metrics(parent_idx=parent_idx, num_parents=1000)
    uniform_metrics = compute_edge_metrics(parent_idx=uniform_idx, num_parents=1000)

    assert chung_lu_metrics["fanout_gini"] > uniform_metrics["fanout_gini"]


def test_tpa_preferential_attachment_creates_heavy_fanout():
    rng = np.random.default_rng(0)
    parent_idx, _ = TPASampler().sample(
        size_a=64,
        size_b=1000,
        params={"alpha": 1.5, "beta": 0.0, "epsilon": 0.1},
        rng=rng,
        child_timestamps=np.arange(1000),
    )
    fanout = np.bincount(parent_idx, minlength=64)

    assert fanout.max() > 3 * fanout.mean()


def test_structural_null_decorator_masks_requested_fraction():
    rng = np.random.default_rng(0)
    parent_idx, null_mask = StructuralNullDecorator(ChungLuSampler()).sample(
        size_a=32,
        size_b=1000,
        params={"gamma": 2.0, "null_rate": 0.3},
        rng=rng,
    )

    assert parent_idx.shape == (1000,)
    assert null_mask is not None
    assert null_mask.mean() == pytest.approx(0.3, abs=0.05)


def test_registry_and_edge_prior_spec_round_trip():
    assert set(TOPOLOGY_PRIOR_REGISTRY) == {
        "hsbm",
        "erdos_renyi",
        "chung_lu",
        "dcsbm",
        "tpa",
    }

    spec = EdgePriorSpec(kind="chung_lu", params={"gamma": 2.0}, null_rate=0.1)

    assert EdgePriorSpec(**asdict(spec)) == spec
