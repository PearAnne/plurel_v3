from __future__ import annotations

import math
import warnings

import networkx as nx
import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised only when the optional dependency exists
    import powerlaw as _powerlaw
except ImportError:  # pragma: no cover - expected in the current sandbox
    _powerlaw = None

_POWERLAW_BOOTSTRAPS = 32

DEFAULT_MAX_POWERLAW_SAMPLE = 200_000
_PLAUSIBLE_GAMMA_RANGE = (1.5, 8.0)
_PLAUSIBLE_KS_MAX = 0.3


def compute_edge_metrics(
    parent_idx: np.ndarray,
    num_parents: int,
    null_mask: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
) -> dict[str, float | int | None]:
    """Compute topology metrics for one FK->PK edge.

    The input is the row-wise parent assignment for a single foreign-key edge.
    ``null_mask`` removes rows from all fanout-based statistics. When timestamps
    are provided, they are used to compute a simple temporal growth proxy over
    the child-row arrival order.

    The power-law fit is restricted to a uniformly subsampled tail of size
    ``max_powerlaw_sample`` when the non-zero fanout vector exceeds that
    threshold. MLE std error scales as ``(gamma - 1) / sqrt(N)`` so 200k samples
    keep gamma accurate to ~0.005 while bounding ``powerlaw.Fit`` xmin-scan
    runtime to minutes. The actual sample size used is reported in
    ``powerlaw_fit_n`` and a sanity ``powerlaw_plausible`` flag aggregates
    gamma and KS distance into a single boolean for downstream filtering.
    """

    parent_idx_arr = _coerce_int_array(parent_idx, name="parent_idx")
    if parent_idx_arr.ndim != 1:
        raise ValueError("parent_idx must be a 1-D array")
    if num_parents < 0:
        raise ValueError("num_parents must be non-negative")

    if null_mask is None:
        active_mask = np.ones(parent_idx_arr.shape[0], dtype=bool)
        null_rate = 0.0
    else:
        active_mask = _coerce_bool_array(null_mask, name="null_mask")
        if active_mask.shape != parent_idx_arr.shape:
            raise ValueError("null_mask must have the same shape as parent_idx")
        null_rate = float(active_mask.mean()) if active_mask.size else 0.0
        active_mask = ~active_mask

    active_parent_idx = parent_idx_arr[active_mask]
    if active_parent_idx.size:
        if active_parent_idx.min() < 0 or active_parent_idx.max() >= num_parents:
            raise ValueError("parent_idx values must lie in [0, num_parents)")
    active_timestamps = _active_timestamps(timestamps=timestamps, active_mask=active_mask)

    fanout = np.bincount(active_parent_idx, minlength=num_parents)
    fanout = fanout.astype(np.int64, copy=False)

    theta_beta_alpha, theta_beta_beta = _fit_beta_mom(fanout)

    metrics: dict[str, float | int | str | bool | None] = {
        "fanout_p05": _safe_percentile(fanout, 5),
        "fanout_p25": _safe_percentile(fanout, 25),
        "fanout_p50": _safe_percentile(fanout, 50),
        "fanout_p75": _safe_percentile(fanout, 75),
        "fanout_p95": _safe_percentile(fanout, 95),
        "fanout_p99": _safe_percentile(fanout, 99),
        "fanout_max": int(fanout.max()) if fanout.size else 0,
        "fanout_gini": _gini(fanout),
        "fanout_ks_to_poisson": _poisson_ks_distance(fanout),
        "fanout_ks_to_powerlaw": math.nan,
        "powerlaw_gamma": math.nan,
        "powerlaw_xmin": math.nan,
        "powerlaw_pvalue": math.nan,
        "powerlaw_fit_n": math.nan,
        "powerlaw_plausible": False,
        "theta_beta_alpha": theta_beta_alpha,
        "theta_beta_beta": theta_beta_beta,
        "pa_exponent_alpha": _pa_exponent_alpha(
            active_parent_idx=active_parent_idx,
            timestamps=active_timestamps,
            num_parents=num_parents,
        ),
        "cardinality_kind": _classify_cardinality(fanout),
        "isolated_parent_rate": float(np.mean(fanout == 0)) if fanout.size else 0.0,
        "null_rate": null_rate,
        "degree_assortativity": math.nan,
        "temporal_growth_alpha": math.nan,
    }

    powerlaw_fit = _fit_powerlaw_tail(fanout, max_powerlaw_sample=max_powerlaw_sample)
    if powerlaw_fit is not None:
        metrics["fanout_ks_to_powerlaw"] = powerlaw_fit["ks_distance"]
        metrics["powerlaw_gamma"] = powerlaw_fit["gamma"]
        metrics["powerlaw_xmin"] = powerlaw_fit["xmin"]
        metrics["powerlaw_pvalue"] = powerlaw_fit["pvalue"]
        metrics["powerlaw_fit_n"] = powerlaw_fit["fit_n"]
        metrics["powerlaw_plausible"] = _is_powerlaw_plausible(
            gamma=powerlaw_fit["gamma"], ks_distance=powerlaw_fit["ks_distance"]
        )

    metrics["degree_assortativity"] = _degree_assortativity(
        active_parent_idx=active_parent_idx, num_parents=num_parents
    )
    metrics["temporal_growth_alpha"] = _temporal_growth_alpha(
        active_parent_idx=active_parent_idx, timestamps=active_timestamps
    )
    return metrics


def _coerce_int_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        return array.astype(np.int64)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must not contain NaN or inf")
        rounded = np.rint(array)
        if not np.allclose(array, rounded):
            raise ValueError(f"{name} must contain integer-valued entries")
        return rounded.astype(np.int64)
    raise TypeError(f"{name} must be an integer array")


def _coerce_bool_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype == bool:
        return array
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(bool, copy=False)
    raise TypeError(f"{name} must be a boolean array")


def _active_timestamps(timestamps: np.ndarray | None, active_mask: np.ndarray) -> np.ndarray | None:
    if timestamps is None:
        return None
    array = np.asarray(timestamps)
    if array.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if array.shape != active_mask.shape:
        raise ValueError("timestamps must have the same shape as parent_idx")
    return array[active_mask]


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(float), percentile))


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.astype(float))
    total = float(sorted_values.sum())
    if total <= 0.0:
        return 0.0
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=float)
    return float(((2.0 * index - n - 1.0) * sorted_values).sum() / (n * total))


def _poisson_ks_distance(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    lam = float(counts.mean())
    max_count = int(counts.max())
    grid = np.arange(max_count + 1, dtype=int)
    empirical = np.searchsorted(np.sort(counts), grid, side="right") / counts.size
    model = _poisson_cdf(grid, lam)
    return float(np.max(np.abs(empirical - model)))


def _poisson_cdf(values: np.ndarray, lam: float) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if values.size == 0:
        return np.zeros(0, dtype=float)
    if lam <= 0.0:
        return np.ones(values.shape, dtype=float)
    max_value = int(values.max())
    pmf = np.empty(max_value + 1, dtype=float)
    pmf[0] = math.exp(-lam)
    for k in range(1, max_value + 1):
        pmf[k] = pmf[k - 1] * lam / k
    cdf = np.cumsum(pmf)
    return cdf[values]


def _fit_powerlaw_tail(
    counts: np.ndarray,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
) -> dict[str, float] | None:
    positive = counts[counts > 0].astype(float)
    if positive.size < 100:
        return None

    # Subsample the non-zero fanout vector before the xmin scan when it exceeds
    # the configured cap. The xmin scan cost in `powerlaw.Fit` is roughly
    # O(unique_values * positive.size); shrinking positive.size from millions to
    # 2e5 turns a multi-hour fit into ~minutes while keeping gamma std error
    # below 0.01 (asymptotic MLE variance: (gamma-1)^2 / N).
    if max_powerlaw_sample > 0 and positive.size > max_powerlaw_sample:
        rng = np.random.default_rng(0)
        sample_indices = rng.choice(positive.size, size=max_powerlaw_sample, replace=False)
        sample = positive[sample_indices]
    else:
        sample = positive
    fit_n = int(sample.size)

    unique_values = np.unique(sample)
    if unique_values.size < 2:
        return None

    if _powerlaw is not None:
        try:  # pragma: no cover - exercised only when the optional dependency exists
            fit = _powerlaw.Fit(sample, discrete=False, verbose=False)
            gamma = float(fit.power_law.alpha)
            xmin = float(fit.power_law.xmin)
            tail = sample[sample >= xmin]
            if tail.size < 2:
                return None
            ks_distance = _continuous_powerlaw_ks(tail, xmin=xmin, gamma=gamma)
            pvalue = _powerlaw_bootstrap_pvalue(tail, xmin=xmin, gamma=gamma)
            return {
                "gamma": gamma,
                "xmin": xmin,
                "ks_distance": ks_distance,
                "pvalue": pvalue,
                "fit_n": fit_n,
            }
        except Exception:
            pass

    best_fit: dict[str, float] | None = None
    for xmin in unique_values:
        tail = sample[sample >= xmin]
        if tail.size < 2:
            continue
        if np.allclose(tail, xmin):
            continue
        gamma = _continuous_powerlaw_gamma(tail, xmin=xmin)
        if not np.isfinite(gamma) or gamma <= 1.0:
            continue
        ks_distance = _continuous_powerlaw_ks(tail, xmin=xmin, gamma=gamma)
        if best_fit is None or ks_distance < best_fit["ks_distance"]:
            best_fit = {
                "gamma": float(gamma),
                "xmin": float(xmin),
                "ks_distance": float(ks_distance),
                "pvalue": math.nan,
                "fit_n": fit_n,
            }

    if best_fit is None:
        return None

    tail = sample[sample >= best_fit["xmin"]]
    best_fit["pvalue"] = _powerlaw_bootstrap_pvalue(
        tail,
        xmin=best_fit["xmin"],
        gamma=best_fit["gamma"],
    )
    return best_fit


def _is_powerlaw_plausible(gamma: float, ks_distance: float) -> bool:
    if not np.isfinite(gamma) or not np.isfinite(ks_distance):
        return False
    low, high = _PLAUSIBLE_GAMMA_RANGE
    return bool(low <= gamma <= high and ks_distance <= _PLAUSIBLE_KS_MAX)


def _continuous_powerlaw_gamma(samples: np.ndarray, xmin: float) -> float:
    tail = samples[samples >= xmin].astype(float)
    if tail.size < 2:
        return math.nan
    if xmin <= 0.0 or np.any(tail < xmin):
        return math.nan
    denom = np.log(tail / xmin).sum()
    if denom <= 0.0:
        return math.nan
    return float(1.0 + tail.size / denom)


def _continuous_powerlaw_cdf(values: np.ndarray, xmin: float, gamma: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if xmin <= 0.0 or gamma <= 1.0:
        return np.zeros(values.shape, dtype=float)
    cdf = np.zeros(values.shape, dtype=float)
    tail = values >= xmin
    cdf[tail] = 1.0 - np.power(values[tail] / xmin, 1.0 - gamma)
    cdf[~tail] = 0.0
    return np.clip(cdf, 0.0, 1.0)


def _continuous_powerlaw_ks(samples: np.ndarray, xmin: float, gamma: float) -> float:
    tail = np.sort(np.asarray(samples, dtype=float))
    if tail.size == 0 or not np.isfinite(gamma):
        return math.nan
    empirical = np.arange(1, tail.size + 1, dtype=float) / tail.size
    model = _continuous_powerlaw_cdf(tail, xmin=xmin, gamma=gamma)
    return float(np.max(np.abs(empirical - model)))


def _sample_continuous_powerlaw(
    size: int,
    xmin: float,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=float)
    if xmin <= 0.0 or gamma <= 1.0:
        return np.full(size, xmin, dtype=float)
    u = rng.random(size)
    return xmin * np.power(1.0 - u, -1.0 / (gamma - 1.0))


def _powerlaw_bootstrap_pvalue(
    samples: np.ndarray,
    xmin: float,
    gamma: float,
    *,
    n_bootstraps: int = _POWERLAW_BOOTSTRAPS,
    seed: int = 0,
) -> float:
    tail = np.asarray(samples, dtype=float)
    if tail.size < 2 or not np.isfinite(gamma) or gamma <= 1.0:
        return math.nan
    observed_ks = _continuous_powerlaw_ks(tail, xmin=xmin, gamma=gamma)
    if not np.isfinite(observed_ks):
        return math.nan

    rng = np.random.default_rng(seed)
    boot_ks = []
    for _ in range(n_bootstraps):
        boot = _sample_continuous_powerlaw(tail.size, xmin=xmin, gamma=gamma, rng=rng)
        boot_gamma = _continuous_powerlaw_gamma(boot, xmin=xmin)
        if not np.isfinite(boot_gamma) or boot_gamma <= 1.0:
            continue
        boot_ks.append(_continuous_powerlaw_ks(boot, xmin=xmin, gamma=boot_gamma))
    if not boot_ks:
        return math.nan
    return float(np.mean(np.asarray(boot_ks) >= observed_ks))


def _degree_assortativity(
    active_parent_idx: np.ndarray,
    num_parents: int,
) -> float:
    if active_parent_idx.size < 2 or num_parents == 0:
        return math.nan

    graph = nx.Graph()
    graph.add_nodes_from(("parent", parent_id) for parent_id in range(num_parents))
    graph.add_nodes_from(("child", child_id) for child_id in range(active_parent_idx.size))
    graph.add_edges_from(
        (("parent", int(parent_id)), ("child", child_id))
        for child_id, parent_id in enumerate(active_parent_idx)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coefficient = nx.degree_assortativity_coefficient(graph)
    if not np.isfinite(coefficient):
        return math.nan
    return float(coefficient)


def _temporal_growth_alpha(
    active_parent_idx: np.ndarray,
    timestamps: np.ndarray | None,
) -> float:
    if timestamps is None or active_parent_idx.size < 2:
        return math.nan
    ts = np.asarray(timestamps)
    if ts.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if ts.shape != active_parent_idx.shape:
        raise ValueError("timestamps must have the same shape as parent_idx")

    timestamp_values = _timestamps_to_float(ts)
    timestamp_values = timestamp_values[np.isfinite(timestamp_values)]
    if timestamp_values.size < 2:
        return math.nan

    unique_timestamps, counts = np.unique(timestamp_values, return_counts=True)
    if unique_timestamps.size < 2:
        return 0.0

    positive_deltas = np.diff(unique_timestamps)
    positive_deltas = positive_deltas[positive_deltas > 0.0]
    if positive_deltas.size == 0:
        return 0.0

    elapsed = unique_timestamps - unique_timestamps.min() + float(np.min(positive_deltas))
    cumulative_edges = np.cumsum(counts).astype(float, copy=False)
    slope, _intercept = np.polyfit(np.log(elapsed), np.log(cumulative_edges), deg=1)
    if not np.isfinite(slope):
        return math.nan
    return float(slope)


def _timestamps_to_float(timestamps: np.ndarray) -> np.ndarray:
    """Convert per-row timestamps to float64 ns-since-epoch; invalid rows become NaN.

    Preserves array length so callers can align with ``parent_idx`` / ``active_mask``.
    Handles ``datetime64``, numeric ordinals, and object/string columns (including
    ``<NA>`` / ``NaT``) via ``pandas.to_datetime`` without raising.
    """
    timestamps = np.asarray(timestamps)
    n = int(timestamps.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if np.issubdtype(timestamps.dtype, np.datetime64):
        out = np.full(n, np.nan, dtype=np.float64)
        valid = ~np.isnat(timestamps)
        out[valid] = timestamps[valid].astype("datetime64[ns]").astype(np.float64)
        return out

    if np.issubdtype(timestamps.dtype, np.floating) or np.issubdtype(timestamps.dtype, np.integer):
        return np.asarray(timestamps, dtype=np.float64)

    parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce", utc=True)
    out = np.full(n, np.nan, dtype=np.float64)
    mask = parsed.notna().to_numpy(dtype=bool)
    if mask.any():
        dt = parsed[mask].to_numpy(dtype="datetime64[ns]")
        out[mask] = dt.astype(np.float64)
    return out


def _fit_beta_mom(fanout: np.ndarray) -> tuple[float, float]:
    """Estimate Beta(alpha, beta) shape from per-parent fanout via method of moments.

    Treats per-parent fanout as proportional to an intrinsic attractiveness
    ``theta_i ~ Beta(alpha, beta)``. Normalizes fanout to ``[0, 1]`` by dividing
    by the max and applies the standard MoM identity
    ``alpha = m * (m * (1 - m) / v - 1)``, ``beta = alpha * (1 - m) / m``
    where ``m`` and ``v`` are the sample mean and variance. Returns
    ``(NaN, NaN)`` on degenerate input (all-zero, single unique value, or
    variance large enough that MoM has no positive solution).
    """
    if fanout.size == 0:
        return math.nan, math.nan
    f_max = float(fanout.max())
    if f_max <= 0:
        return math.nan, math.nan
    theta = fanout.astype(float) / f_max
    m = float(theta.mean())
    v = float(theta.var(ddof=0))
    if v <= 0.0 or m <= 0.0 or m >= 1.0:
        return math.nan, math.nan
    if v >= m * (1.0 - m):
        return math.nan, math.nan
    alpha = m * (m * (1.0 - m) / v - 1.0)
    beta = alpha * (1.0 - m) / m
    if alpha <= 0.0 or beta <= 0.0:
        return math.nan, math.nan
    return float(alpha), float(beta)


def _pa_exponent_alpha(
    active_parent_idx: np.ndarray,
    timestamps: np.ndarray | None,
    num_parents: int,
) -> float:
    """Estimate preferential-attachment exponent ``alpha`` in ``W(k) ∝ k^alpha``.

    Uses Newman's split-half kernel estimator (Newman 2003, "Mixing patterns in
    networks"): sort children by arrival time, split in halves, regress
    ``log(degree_gained_in_second_half)`` on ``log(degree_at_midpoint)``. The
    slope is ``alpha``. Returns NaN when timestamps are missing, the population
    is too small (< 200 events / < 10 parents), or there is no degree variance.

    Unlike ``temporal_growth_alpha`` (which measures cumulative-edge growth
    versus elapsed time, a coarse activity-table signal), this estimator
    targets the **conditional attachment kernel** that the TPA sampler uses
    directly. Empirical alpha ≈ 1 corresponds to linear preferential
    attachment (BA model), alpha > 1 to super-linear / hub-dominated, alpha
    near 0 to ER-like uniform attachment.
    """
    if timestamps is None or active_parent_idx.size < 200 or num_parents < 10:
        return math.nan
    timestamps_arr = _timestamps_to_float(np.asarray(timestamps))
    if timestamps_arr.size != active_parent_idx.size:
        return math.nan
    finite_mask = np.isfinite(timestamps_arr)
    if finite_mask.sum() < 200:
        return math.nan
    timestamps_arr = timestamps_arr[finite_mask]
    parent_idx = active_parent_idx[finite_mask]

    order = np.argsort(timestamps_arr, kind="stable")
    p_sorted = parent_idx[order]
    n = p_sorted.size
    mid = n // 2
    if mid < 50 or n - mid < 50:
        return math.nan

    deg_first = np.bincount(p_sorted[:mid], minlength=num_parents).astype(np.float64)
    deg_total = np.bincount(p_sorted, minlength=num_parents).astype(np.float64)
    deg_delta = deg_total - deg_first

    mask = (deg_first > 0) & (deg_delta > 0)
    if mask.sum() < 10:
        return math.nan

    log_k = np.log(deg_first[mask])
    log_delta = np.log(deg_delta[mask])
    if log_k.std(ddof=0) <= 0.0:
        return math.nan

    slope, _intercept = np.polyfit(log_k, log_delta, 1)
    if not np.isfinite(slope):
        return math.nan
    return float(slope)


def _classify_cardinality(fanout: np.ndarray) -> str:
    """Coarse cardinality classification for one FK edge.

    Returned values:
        "no_data"       - empty / all-zero fanout (e.g. fully-null FK column).
        "one_to_one"    - every reached parent receives exactly one child.
        "many_to_one"   - standard case, some parent receives >= 2 children.
    """
    if fanout.size == 0:
        return "no_data"
    fmax = int(fanout.max())
    if fmax == 0:
        return "no_data"
    if fmax == 1:
        return "one_to_one"
    return "many_to_one"
