"""Online supervised adaptation of the fill-probability NN.

The fill network only *predicts* fill probability; the HJB still does the
quoting.  The loop is iterative/fitted supervised learning with a replay
buffer:

1.  Run ``SIMS_PER_ROUND`` single-agent sims in parallel.  Each agent quotes
    via the HJB using the *current* fill checkpoint and logs, per 1s window
    and side, the exact (normalised) NN input row, the realized fill in
    ``{0, 1}`` and the predicted ``h`` at collection time.
2.  Train the network to predict the realized fills (``BCEWithLogitsLoss``,
    warm-started, normalisation frozen) on a replay buffer of the most recent
    rounds.
3.  Save a fresh, drop-in checkpoint; the next round of workers loads it.

Plotting predicted-vs-realized between rounds shows whether calibration
improves. There is no reward or policy-gradient objective in this module.

CPU-only: inference workers remain single-threaded to prevent Joblib
oversubscription. The parent-process trainer may use multiple Torch threads;
no CUDA is required.

This module is fully additive and reload-safe: it does not change any existing
quoting behaviour (the feature logging in :class:`CadenceNNErgodicMM` is gated
on ``feature_log is not None`` and stays off for normal runs).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .mm_competition import (
    CadenceNNErgodicMM,
    SimFeatureExtractor,
    init_worker_seed,
    make_sim_nn_h_fn,
    DEFAULT_N_QUEUE_LEVELS,
)
from .mm_backtest_parallel import _load_nn_bundle, assemble_fill_X


def resolve_vol_use_realized(bundle: Dict[str, Any], vol_feature_mode: str) -> bool:
    """Mirror the ``recent_vol`` mode resolution used by the fill closure."""
    mode = str(vol_feature_mode)
    if mode == "auto":
        mode = str(bundle.get("vol_mode") or "ewma_event")
    return mode == "realized_time"


# --- Feature-logging agent ---
class RLFillMM(CadenceNNErgodicMM):
    """``CadenceNNErgodicMM`` that logs (X, realized, pred) training tuples.

    The quoting path is unchanged; only :attr:`feature_log` is enabled and
    :meth:`_capture_features` reproduces the model input via the shared
    :func:`assemble_fill_X` helper so logged rows are byte-identical to what
    the fill network saw at quote time.
    """

    def __init__(self, *args,
                 rl_feat_mean=None, rl_feat_std=None, rl_n_feat=None,
                 rl_use_realized: bool = False,
                 rl_t0: Optional[float] = None, rl_span: Optional[float] = None,
                 rl_queue_transform: Optional[str] = None,
                 rl_queue_ahead_mode: str = "exact_fifo",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._rl_feat_mean = rl_feat_mean
        self._rl_feat_std = rl_feat_std
        self._rl_n_feat = int(rl_n_feat)
        self._rl_use_realized = bool(rl_use_realized)
        self._rl_t0 = rl_t0
        self._rl_span = rl_span
        self._rl_queue_transform = rl_queue_transform
        self._rl_queue_ahead_mode = str(rl_queue_ahead_mode)
        # Enable per-window feature logging in the base class.
        self.feature_log = []

    def _capture_features(self, feature_view, delta_b_pln, delta_a_pln):
        if self._rl_feat_mean is None or self._rl_n_feat is None:
            return None, None
        X_b = assemble_fill_X(
            1, self._rl_feat_mean, self._rl_feat_std, self._rl_n_feat,
            feature_view, np.array([delta_b_pln], dtype=np.float64),
            agent=self, use_realized=self._rl_use_realized,
            t0=self._rl_t0, span=self._rl_span,
            queue_transform=self._rl_queue_transform,
            queue_ahead_mode=self._rl_queue_ahead_mode,
        )[0]
        X_a = assemble_fill_X(
            2, self._rl_feat_mean, self._rl_feat_std, self._rl_n_feat,
            feature_view, np.array([delta_a_pln], dtype=np.float64),
            agent=self, use_realized=self._rl_use_realized,
            t0=self._rl_t0, span=self._rl_span,
            queue_transform=self._rl_queue_transform,
            queue_ahead_mode=self._rl_queue_ahead_mode,
        )[0]
        return X_b, X_a


# --- Single-agent joblib worker ---
def run_fill_rl_sim(
    run_id: int,
    *,
    ckpt_path: str,
    snapshot_kwargs: Dict[str, Any],
    erg_params: Dict[str, Any],
    size: int,
    gamma: float,
    solver_tick: float,
    poisson_tau: float,
    delta_lo: float,
    max_iter: int,
    tol: float,
    T: int,
    max_delta: float = 2.0,
    drift_eps: float = 0.0,
    requote_cadence: float = 1.0,
    base_seed: int = 12345,
    day_span_s: Optional[float] = None,
    vol_feature_mode: str = "auto",
    agents_affect_kernels: bool = False,
    agents_affect_mo_sizing: bool = False,
    rho_in: float = 0.0,
    solver_engine: str = "scan",
    candidate_grid: str = "legal",
    queue_ahead_mode: str = "exact_fifo",
    resil_kappa: float = 0.0,
    resil_tau: float = 10.0,
    resil_varphi: float = 0.0,
    resil_tau_f: float = 40.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one single-agent ``Simulate`` and collect fill training tuples.

    The lone agent quotes via the HJB using the fill network at ``ckpt_path``
    and logs, per 1s window and side, the normalised NN input row, the realized
    fill in ``{0, 1}`` and the predicted ``h``.  Returns compact numpy arrays
    (small IPC) so the parent can train on them:

    ``{"run_id", "seed", "t_end", "realized_pnl", "X" (N, in_dim),
    "y" (N,), "pred" (N,)}``.

    The environment matches the competition runs (single agent, agents do not
    excite the Hawkes kernels, no in-spread background LOs) so the learned net
    is valid as the static benchmark net in multi-agent runs.
    """
    import os
    import sys

    seed = init_worker_seed(base_seed, run_id)

    from .simulate import Simulate

    bundle = _load_nn_bundle(ckpt_path)
    extractor = SimFeatureExtractor(DEFAULT_N_QUEUE_LEVELS)
    tick_size = float(snapshot_kwargs.get("tick_size", 0.05))
    use_realized = resolve_vol_use_realized(bundle, vol_feature_mode)

    holder: list = [None]
    h_bid = make_sim_nn_h_fn(
        1, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
        vol_feature_mode=vol_feature_mode,
        queue_ahead_mode=queue_ahead_mode)
    h_ask = make_sim_nn_h_fn(
        2, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
        vol_feature_mode=vol_feature_mode,
        queue_ahead_mode=queue_ahead_mode)
    agent = RLFillMM(
        **erg_params, gamma=gamma, size=size, verbose=False,
        solver_tick=solver_tick, solver_engine=solver_engine,
        candidate_grid=candidate_grid,
        h_b=h_bid, h_a=h_ask,
        poisson_tau=poisson_tau, delta_lo=delta_lo, max_delta=max_delta,
        max_iter=max_iter, tol=tol, state_extractor=extractor,
        agent_id="rl_0", requote_cadence=requote_cadence, n_agents=1,
        rl_feat_mean=bundle["feat_mean"], rl_feat_std=bundle["feat_std"],
        rl_n_feat=bundle["n_feat"], rl_use_realized=use_realized,
        rl_t0=0.0, rl_span=day_span_s,
        rl_queue_transform=bundle.get("queue_transform", None),
        rl_queue_ahead_mode=queue_ahead_mode)
    holder[0] = agent

    _devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")
    _old_stdout = sys.stdout
    if not verbose:
        sys.stdout = _devnull
    try:
        sim = Simulate(
            T=int(T), lightweight=True, agents_when_lightweight=True,
            agents=[agent], shuffle_agents=False, drift_eps=drift_eps,
            agents_affect_kernels=agents_affect_kernels,
            agents_affect_mo_sizing=agents_affect_mo_sizing,
            rho_in=rho_in,
            resil_kappa=resil_kappa, resil_tau=resil_tau,
            resil_varphi=resil_varphi, resil_tau_f=resil_tau_f,
            tick_size=tick_size)
        sim.load_real_orderbook_snapshot(**snapshot_kwargs)
        sim.run()
        t_end = float(getattr(sim, "current_time", 0.0) or 0.0)
        agent.liquidate(sim, t_end)
    finally:
        if not verbose:
            sys.stdout = _old_stdout
        _devnull.close()

    feature_rows = agent.feature_log
    in_dim = int(bundle["n_feat"]) + 3
    if feature_rows:
        X = np.vstack([row[0] for row in feature_rows]).astype(np.float32)
        y = np.asarray([row[1] for row in feature_rows], dtype=np.float32)
        pred = np.asarray([row[2] for row in feature_rows], dtype=np.float32)
    else:
        X = np.empty((0, in_dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        pred = np.empty((0,), dtype=np.float32)

    realized_pnl = (
        float(agent.state_log[-1][4])
        if agent.state_log
        else float(agent.cash)
    )
    return {
        "run_id": int(run_id), "seed": int(seed), "t_end": t_end,
        "realized_pnl": realized_pnl,
        "X": X, "y": y, "pred": pred, "n_windows": int(X.shape[0]),
    }


# --- Multi-agent joblib worker ---
def run_fill_rl_multi_sim(
    run_id: int,
    n_agents: int,
    *,
    ckpt_path: str,
    snapshot_kwargs: Dict[str, Any],
    erg_params: Dict[str, Any],
    size: int,
    gamma: float,
    solver_tick: float,
    poisson_tau: float,
    delta_lo: float,
    max_iter: int,
    tol: float,
    T: int,
    max_delta: float = 2.0,
    drift_eps: float = 0.0,
    requote_cadence: float = 1.0,
    base_seed: int = 12345,
    day_span_s: Optional[float] = None,
    vol_feature_mode: str = "auto",
    agents_affect_kernels: bool = False,
    agents_affect_mo_sizing: bool = False,
    rho_in: float = 0.0,
    solver_engine: str = "scan",
    candidate_grid: str = "legal",
    queue_ahead_mode: str = "exact_fifo",
    resil_kappa: float = 0.0,
    resil_tau: float = 10.0,
    resil_varphi: float = 0.0,
    resil_tau_f: float = 40.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one ``n_agents``-agent ``Simulate`` and collect fill training
    tuples from **all** agents (pooled).

    Identical to :func:`run_fill_rl_sim` except ``n_agents`` RLFillMM agents
    compete in the same simulation.  The returned ``X``, ``y``, ``pred``
    arrays contain data from every agent, so the caller can train a single
    shared network on pooled multi-agent experience.
    """
    import os
    import sys

    seed = init_worker_seed(base_seed, run_id)

    from .simulate import Simulate

    bundle = _load_nn_bundle(ckpt_path)
    extractor = SimFeatureExtractor(DEFAULT_N_QUEUE_LEVELS)
    tick_size = float(snapshot_kwargs.get("tick_size", 0.05))
    use_realized = resolve_vol_use_realized(bundle, vol_feature_mode)

    agents: List[RLFillMM] = []
    for agent_idx in range(int(n_agents)):
        holder: list = [None]
        h_bid = make_sim_nn_h_fn(
            1, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode,
            queue_ahead_mode=queue_ahead_mode)
        h_ask = make_sim_nn_h_fn(
            2, bundle, holder, day_t0_s=0.0, day_span_s=day_span_s,
            vol_feature_mode=vol_feature_mode,
            queue_ahead_mode=queue_ahead_mode)
        agent = RLFillMM(
            **erg_params, gamma=gamma, size=size, verbose=False,
            solver_tick=solver_tick, solver_engine=solver_engine,
            candidate_grid=candidate_grid,
            h_b=h_bid, h_a=h_ask,
            poisson_tau=poisson_tau, delta_lo=delta_lo, max_delta=max_delta,
            max_iter=max_iter, tol=tol, state_extractor=extractor,
            agent_id=f"rl_{agent_idx}", requote_cadence=requote_cadence,
            n_agents=int(n_agents),
            rl_feat_mean=bundle["feat_mean"], rl_feat_std=bundle["feat_std"],
            rl_n_feat=bundle["n_feat"], rl_use_realized=use_realized,
            rl_t0=0.0, rl_span=day_span_s,
            rl_queue_transform=bundle.get("queue_transform", None),
            rl_queue_ahead_mode=queue_ahead_mode)
        holder[0] = agent
        agents.append(agent)

    _devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")
    _old_stdout = sys.stdout
    if not verbose:
        sys.stdout = _devnull
    try:
        sim = Simulate(
            T=int(T), lightweight=True, agents_when_lightweight=True,
            agents=agents, shuffle_agents=True, drift_eps=drift_eps,
            agents_affect_kernels=agents_affect_kernels,
            agents_affect_mo_sizing=agents_affect_mo_sizing,
            rho_in=rho_in,
            resil_kappa=resil_kappa, resil_tau=resil_tau,
            resil_varphi=resil_varphi, resil_tau_f=resil_tau_f,
            tick_size=tick_size)
        sim.load_real_orderbook_snapshot(**snapshot_kwargs)
        sim.run()
        t_end = float(getattr(sim, "current_time", 0.0) or 0.0)
        for agent in agents:
            agent.liquidate(sim, t_end)
    finally:
        if not verbose:
            sys.stdout = _old_stdout
        _devnull.close()

    in_dim = int(bundle["n_feat"]) + 3
    pooled_X, pooled_y, pooled_pred = [], [], []
    for agent in agents:
        feature_rows = agent.feature_log
        if feature_rows:
            pooled_X.append(np.vstack([row[0] for row in feature_rows]).astype(np.float32))
            pooled_y.append(np.asarray([row[1] for row in feature_rows], dtype=np.float32))
            pooled_pred.append(np.asarray([row[2] for row in feature_rows], dtype=np.float32))

    if pooled_X:
        X = np.vstack(pooled_X)
        y = np.concatenate(pooled_y)
        pred = np.concatenate(pooled_pred)
    else:
        X = np.empty((0, in_dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        pred = np.empty((0,), dtype=np.float32)

    return {
        "run_id": int(run_id), "seed": int(seed), "t_end": t_end,
        "n_agents": int(n_agents),
        "X": X, "y": y, "pred": pred, "n_windows": int(X.shape[0]),
    }


# --- Online trainer ---
def update_fill_nn(
    ckpt_path: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    out_path: str,
    epochs: int = 8,
    lr: float = 1e-4,
    batch_size: int = 65536,
    weight_decay: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
    training_num_threads: int = 1,
) -> Dict[str, Any]:
    """Warm-start the fill net from ``ckpt_path`` and fit ``X -> y`` (BCE).

    ``feat_mean`` / ``feat_std`` are kept frozen (only the weights adapt), so
    the logged normalised ``X`` rows stay valid across rounds.  Saves a drop-in
    checkpoint at ``out_path`` (all metadata copied, only ``state_dict``
    replaced) usable directly as a ``ckpt_path`` for the sim workers.

    ``training_num_threads`` applies only to this parent-process update. Live
    simulation workers continue to use one Torch thread each.

    Returns training losses, elapsed time, and the effective thread count.
    """
    import torch
    import torch.nn as nn

    if X.shape[0] == 0:
        raise ValueError("update_fill_nn received no training rows.")
    if int(training_num_threads) < 1:
        raise ValueError("training_num_threads must be at least one.")

    torch.manual_seed(int(seed))

    # Reuse the exact inference architecture (eval mode); switch to train.
    bundle = _load_nn_bundle(ckpt_path)
    previous_num_threads = torch.get_num_threads()
    training_started_at = time.perf_counter()
    try:
        torch.set_num_threads(int(training_num_threads))
        effective_num_threads = torch.get_num_threads()
        model = bundle["model"]
        model.train()

        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        n = Xt.shape[0]

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        rng = np.random.default_rng(int(seed))
        loss_first = None
        loss_last = None
        for ep in range(int(epochs)):
            perm = rng.permutation(n)
            ep_loss = 0.0
            ep_count = 0
            for start in range(0, n, int(batch_size)):
                idx = perm[start:start + int(batch_size)]
                xb = Xt[idx]
                yb = yt[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                bs = xb.shape[0]
                ep_loss += float(loss.item()) * bs
                ep_count += bs
            ep_loss /= max(1, ep_count)
            if loss_first is None:
                loss_first = ep_loss
            loss_last = ep_loss
            if verbose:
                print(
                    f"    [update] epoch {ep + 1}/{epochs}  BCE={ep_loss:.5f}",
                    flush=True,
                )

        model.eval()

        # Save a drop-in checkpoint with updated weights and unchanged metadata.
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw, out_path)
        training_seconds = time.perf_counter() - training_started_at
    finally:
        torch.set_num_threads(previous_num_threads)

    return {
        "out_path": str(out_path),
        "n": int(n),
        "loss_first": float(loss_first),
        "loss_last": float(loss_last),
        "training_seconds": float(training_seconds),
        "training_num_threads": int(effective_num_threads),
    }


# --- Calibration helpers ---
def bce_score(pred: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy of probabilities ``pred`` against labels."""
    p = np.clip(np.asarray(pred, dtype=np.float64), eps, 1.0 - eps)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p)))


def brier_score(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between probabilities and labels."""
    p = np.asarray(pred, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - yy) ** 2))


def calibration_table(pred: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Reliability-curve bins over predicted probability.

    Returns ``(mean_pred, mean_real, count)`` arrays (one entry per non-empty
    quantile bin), suitable for a predicted-vs-realized scatter / line.
    """
    p = np.asarray(pred, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        z = np.empty(0)
        return z, z, z
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return (
            np.array([p.mean()]),
            np.array([yy.mean()]),
            np.array([float(p.size)]),
        )
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    mean_pred, mean_real, count = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        mean_pred.append(float(p[m].mean()))
        mean_real.append(float(yy[m].mean()))
        count.append(float(m.sum()))
    return (np.asarray(mean_pred), np.asarray(mean_real), np.asarray(count))


def delta_ticks_from_X(X: np.ndarray, feat_mean, feat_std, n_feat: int,
                       tick: float) -> np.ndarray:
    """Recover quoted depth in ticks from the normalised input matrix ``X``.

    ``delta`` sits at column ``n_feat`` (PLN, z-scored with ``feat_mean`` /
    ``feat_std``); de-normalise and divide by the tick size.
    """
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    delta_pln = (
        X[:, n_feat].astype(np.float64) * float(feat_std[n_feat])
        + float(feat_mean[n_feat])
    )
    return delta_pln / float(tick)


def depth_calibration_table(
    X: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    feat_mean,
    feat_std,
    n_feat: int,
    tick: float,
    *,
    min_cnt: int = 10,
    max_depth: float = 20.0,
):
    """Predicted vs realized fill rate by quoted depth (ticks behind mid).

    Mirrors the competition ``fill_probe`` depth buckets used in
    ``simulation.ipynb`` Fig 5.  Returns a small frame with columns
    ``dbucket``, ``pred``, ``real``, ``cnt``.
    """
    import pandas as pd

    if X.shape[0] == 0:
        return pd.DataFrame(columns=["dbucket", "pred", "real", "cnt"])

    depth_ticks = delta_ticks_from_X(X, feat_mean, feat_std, n_feat, tick)
    depth_bucket = np.round(depth_ticks).clip(0.0, max_depth)
    frame = pd.DataFrame({"dbucket": depth_bucket, "pred": pred, "realized": y})
    g = (frame.groupby("dbucket", as_index=False)
         .agg(pred=("pred", "mean"), real=("realized", "mean"), cnt=("realized", "size")))
    return g[g["cnt"] >= min_cnt].sort_values("dbucket").reset_index(drop=True)
