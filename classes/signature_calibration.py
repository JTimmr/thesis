"""Sim-outcome (indirect-inference) calibration helpers for the
state-dependent limit-order placement mechanism.

The resiliency parameters ``(resil_kappa, resil_phi)`` are *effective*
reduced-form parameters: in the simulator the placement tilt is the only
reversion channel, so it must absorb the aggregate effect of every empirical
reversion channel (contrarian MO flow, cancellation asymmetry, depth
redistribution, ...).  They are therefore calibrated by matching *simulated
behaviour* — the normalized signature curve VR(tau) = C(tau)/C(1s) — to the
empirical 50-day KGHM panel, NOT by direct micro-MLE (see
``docs/signature_plot_calibration_problem.md`` and
``notebooks/empirical_placement_bias.ipynb`` for why the two disagree).

Anti-winner's-curse protocol (Smith & Winkler 2006, and the failed Optuna
searches documented in the doc):

* every configuration is measured with ``runs`` *unseeded* replicates and
  summarized by the mean curve and its standard error;
* every batch contains the control row ``(kappa, phi) = (0, 0)`` so
  comparisons are against a same-noise baseline;
* selection happens on mean curves over a tiny, pre-registered candidate
  set (a local linear dose-response solve), never on the noisy minimum of
  a large search;
* the selected point is re-validated with fresh runs.

All functions here are module-level so they pickle cleanly into a
``ProcessPoolExecutor`` on Windows (notebook-defined functions do not).
"""
from __future__ import annotations

import copy
import os

import numpy as np

# Session splitting identical to the evidence notebook / validation script.
SESSION_S = 7.8 * 3600.0
WARMUP_S = 60.0
TAUS_DENSE = np.arange(1, 301, dtype=float)


def load_empirical_vr():
    """Empirical VR panel from the 50-day dense signature cache.

    Returns ``(taus, vr_mean, vr_se)`` on the tau <= 300 s grid, with each
    day's curve normalized by its own C(1 s) before averaging.
    """
    from .helpers import data_dir

    z = np.load(os.path.join(str(data_dir()),
                             "_cache_signature_multiday_dense_300s.npz"),
                allow_pickle=True)
    taus = np.asarray(z["taus"], float)
    emp = np.asarray(z["emp_linear"], float)
    vr = emp / emp[:, [0]]
    m = taus <= 300
    mean = vr.mean(axis=0)[m]
    se = (vr.std(axis=0, ddof=1) / np.sqrt(vr.shape[0]))[m]
    return taus[m], mean, se


def run_signature_sessions(template, kappa, phi, tau_s=4.0, flow_tau_s=40.0, T=1_000_000, base_overrides=None):
    """One unseeded sim -> list of per-session dense signature curves.

    Deep-copies *template*, sets the resiliency parameters (and any
    ``base_overrides`` attributes, e.g. ``lo_p_best`` / ``lo_inside_c1`` /
    ``lo_inside_c0``), runs ``T`` events, splits the run into consecutive
    7.8 h sessions and returns one C(tau) curve (tau = 1..300 s) per
    session — the exact evidence-notebook methodology.
    """
    from .mean_reversion_metrics import signature_curve

    sim = copy.deepcopy(template)
    sim.T = int(T)
    sim.verbose = False
    sim.capture_mid = True
    sim.resil_kappa = float(kappa)
    sim.resil_phi = float(phi)
    sim.resil_tau_s = float(tau_s)
    sim.resil_flow_tau_s = float(flow_tau_s)
    if base_overrides:
        for name, value in base_overrides.items():
            setattr(sim, name, value)
    sim.run(verbose=False)
    t, mid = sim.get_mid_series()
    if len(t) < 1000:
        return []
    curves = []
    k = 0
    while True:
        a = WARMUP_S + k * SESSION_S
        b = a + SESSION_S
        if b > float(t[-1]):
            break
        m = (t >= a) & (t <= b)
        if m.sum() > 500:
            c = signature_curve(t[m], mid[m], TAUS_DENSE)
            if np.all(np.isfinite(c)):
                curves.append(c)
        k += 1
    return curves


def _run_job(args):
    """Worker wrapper: never raises, returns (config_key, curves)."""
    template, kappa, phi, tau_s, flow_tau_s, T, base_overrides = args
    try:
        curves = run_signature_sessions(template, kappa, phi, tau_s,
                                        flow_tau_s, T, base_overrides)
        return (kappa, phi), curves
    except Exception as exc:  # keep the batch alive; report and move on
        print(f"  [job kappa={kappa} phi={phi}] FAILED: {exc!r}", flush=True)
        return (kappa, phi), []


def run_batch(template, configs, *, runs=6, workers=10, T=1_000_000,
              tau_s=4.0, flow_tau_s=40.0, base_overrides=None, verbose=True):
    """Run ``runs`` unseeded replicates of every ``(kappa, phi)`` config.

    Returns ``{(kappa, phi): np.ndarray of shape (n_sessions, 300)}`` of raw
    per-session C(tau) curves.  Jobs from all configs are interleaved across
    one process pool, so the batch finishes in
    ``ceil(len(configs) * runs / workers)`` waves.
    """
    import concurrent.futures
    import time

    jobs = [(template, float(kappa), float(phi), tau_s, flow_tau_s, T, base_overrides)
            for (kappa, phi) in configs for _ in range(runs)]
    results = {(float(kappa), float(phi)): [] for (kappa, phi) in configs}

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_job, job_args) for job_args in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            key, curves = fut.result()
            results[key].extend(curves)
            if verbose:
                print(f"  [{i + 1}/{len(jobs)}] kappa={key[0]:g} phi={key[1]:g} "
                      f"+{len(curves)} sessions "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    return {k: np.asarray(v, float) for k, v in results.items()}


def vr_stats(curves):
    """Per-session-normalized VR mean and SE from raw C(tau) session curves."""
    c = np.asarray(curves, float)
    vr = c / c[:, [0]]
    return vr.mean(axis=0), vr.std(axis=0, ddof=1) / np.sqrt(vr.shape[0])


def solve_dose_response(taus, vr0, g_kappa, g_phi, emp_vr, target_taus, weights):
    """Weighted least-squares solve of the local linear dose-response model.

    ``VR(tau; kappa, phi) ~ VR0(tau) + kappa * g_kappa(tau) + phi * g_phi(tau)``
    is solved for ``(kappa, phi)`` on the ``target_taus`` horizons.
    """
    idx = np.array([int(np.argmin(np.abs(taus - t))) for t in target_taus])
    w = np.sqrt(np.asarray(weights, float))
    A = np.column_stack([g_kappa[idx], g_phi[idx]]) * w[:, None]
    b = (emp_vr[idx] - vr0[idx]) * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(sol[0]), float(sol[1])


# --- 4-parameter variants: (kappa, phi, tau_s, flow_tau_s) per config ---

def _run_job_4d(args):
    """Worker wrapper for 4-param configs."""
    template, kappa, phi, tau_s, flow_tau_s, T, base_overrides = args
    key = (kappa, phi, tau_s, flow_tau_s)
    try:
        curves = run_signature_sessions(template, kappa, phi, tau_s,
                                        flow_tau_s, T, base_overrides)
        return key, curves
    except Exception as exc:
        print(f"  [job {key}] FAILED: {exc!r}", flush=True)
        return key, []


def run_batch_4d(template, configs, *, runs=6, workers=10, T=1_000_000,
                 base_overrides=None, verbose=True):
    """Run ``runs`` unseeded replicates per ``(kappa, phi, tau_s, flow_tau_s)``.

    Returns ``{(kappa, phi, tau_s, flow_tau_s): ndarray(n_sessions, 300)}``.
    """
    import concurrent.futures
    import time

    cfgs = [tuple(float(x) for x in c) for c in configs]
    jobs = [(template, c[0], c[1], c[2], c[3], T, base_overrides)
            for c in cfgs for _ in range(runs)]
    results = {c: [] for c in cfgs}

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_job_4d, job_args) for job_args in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            key, curves = fut.result()
            results[key].extend(curves)
            if verbose:
                print(f"  [{i+1}/{len(jobs)}] "
                      f"\u03ba={key[0]:.3f} \u03c6={key[1]:.3f} "
                      f"\u03c4={key[2]:g} \u03c4f={key[3]:g} "
                      f"+{len(curves)} sess "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    return {k: np.asarray(v, float) for k, v in results.items()}


def solve_dose_response_nd(taus, vr0, gradients, emp_vr,
                           target_taus, weights):
    """Weighted least-squares solve for N parameters.

    Parameters
    ----------
    gradients : list of N arrays, each shape ``(len(taus),)`` — the
        finite-difference gradient of VR w.r.t. each parameter.

    Returns
    -------
    list of N float steps (additive corrections to the center).
    """
    idx = np.array([int(np.argmin(np.abs(taus - t))) for t in target_taus])
    w = np.sqrt(np.asarray(weights, float))
    A = np.column_stack([g[idx] for g in gradients]) * w[:, None]
    b = (emp_vr[idx] - vr0[idx]) * w
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return [float(s) for s in sol]


__all__ = [
    "SESSION_S",
    "WARMUP_S",
    "TAUS_DENSE",
    "load_empirical_vr",
    "run_signature_sessions",
    "run_batch",
    "run_batch_4d",
    "vr_stats",
    "solve_dose_response",
    "solve_dose_response_nd",
]
