"""Mean-reversion / signature / 2-D Hawkes microstructure metrics.

These pure metric, signature-curve, and 2-D Hawkes-fit functions are lifted
verbatim from ``notebooks/mean_reversion_evidence.ipynb`` so they can be shared
by that notebook and a separate calibration harness.  Only the imports were
adjusted; the numerics are unchanged.  Data-loading helpers that touch
SQLite/files (e.g. ``build_source``, ``build_mid_series``, ``list_db_days``)
are intentionally excluded -- this module is self-contained and pure.
"""

import numpy as np

# tick size (PLN) for KGHM on the WSE
TICK = 0.05


# --- Generic analytics ---

def sample_grid(t, mid, dt):
    """Last-observation-carried-forward sample of the mid on a uniform grid."""
    grid = np.arange(t[0], t[-1], dt)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(mid) - 1)
    return grid, mid[idx]


def acf(x, max_lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def xcorr(a, b, max_lag):
    """corr(a_t, b_{t+lag}) for lag = 0..max_lag (a leads b)."""
    a = np.asarray(a, float) - np.asarray(a, float).mean()
    b = np.asarray(b, float) - np.asarray(b, float).mean()
    denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom == 0.0:
        return np.zeros(max_lag + 1)
    n = len(a)
    return np.array([np.dot(a[:n - k], b[k:]) / denom for k in range(0, max_lag + 1)])


def return_acf(t, mid, dt, max_lag):
    _, mg = sample_grid(t, mid, dt)
    r = np.diff(mg)
    return acf(r, max_lag), len(r)


def realized_var_rate(t, mid, tau, mode="linear"):
    """Realised variance per unit time on a LOCF grid: (1/T) sum(r^2)."""
    _, mg = sample_grid(t, mid, tau)
    if mode == "log":
        mg = np.log(np.maximum(mg, 1e-12))
    r = np.diff(mg)
    T = float(t[-1] - t[0])
    if T <= 0 or len(r) == 0:
        return np.nan
    return float(np.sum(r * r) / T)


def signature_curve(t, mid, taus, mode="linear"):
    """Signature curve C(delta) on a LOCF mid grid.

    The dense paper-style panel uses integer-second horizons. For those we sample
    once at 1s and reuse every kth point, which is equivalent to the original
    non-overlapping grid but much faster for 1..300s curves.
    """
    taus = np.asarray(taus, float)
    tau_i = np.rint(taus).astype(int)
    T = float(t[-1] - t[0])
    if T <= 0:
        return np.full_like(taus, np.nan, dtype=float)

    if np.allclose(taus, tau_i) and np.all(tau_i >= 1):
        _, mg1 = sample_grid(t, mid, 1.0)
        if mode == "log":
            mg1 = np.log(np.maximum(mg1, 1e-12))
        out = []
        for tau in tau_i:
            sampled = mg1[::tau]
            r = np.diff(sampled)
            out.append(np.nan if len(r) == 0 else float(np.sum(r * r) / T))
        return np.asarray(out, float)

    return np.array([realized_var_rate(t, mid, tau, mode=mode) for tau in taus])


def mid_locf(t, mid, q):
    idx = np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(mid) - 1)
    return mid[idx]


def updown_per_bin(t, mid, dt):
    """Up / down mid-move magnitude (PLN) per uniform bin (paper's N^u, N^d)."""
    _, mg = sample_grid(t, mid, dt)
    dm = np.diff(mg)
    return np.maximum(dm, 0.0), np.maximum(-dm, 0.0)


# paper's closed-form signature plot  C(tau) = A[k2 + (1-k2)(1-e^{-g*tau})/(g*tau)]  (Eq. 36)
# C(0)=A (high-freq vol), C(inf)=A*k2 (low-freq vol).  k2<1 => mean reversion.
def paper_signature(tau, A, kappa2, gamma):
    tau = np.asarray(tau, float)
    return A * (kappa2 + (1.0 - kappa2) * (1.0 - np.exp(-gamma * tau)) / (gamma * tau))


# --- Symmetric 2-D up/down exponential Hawkes MLE ---

def hawkes2d_negloglik(params, up, dn, T):
    """Negative log-likelihood of the symmetric bivariate exp-Hawkes.

    lambda_u = mu + a_s * A_u + a_m * A_d ; lambda_d = mu + a_m * A_u + a_s * A_d
    where A_u, A_d are the decayed counts of past up / down jumps.  O(N).
    """
    mu, a_s, a_m, beta = params
    if min(mu, a_s, a_m, beta) <= 0.0:
        return 1e18
    times = np.concatenate([up, dn])
    types = np.concatenate([np.zeros(len(up), np.int8), np.ones(len(dn), np.int8)])
    order = np.argsort(times, kind="mergesort")
    times, types = times[order], types[order]
    Au = Ad = 0.0
    last = 0.0
    ll = 0.0
    for tm, ty in zip(times, types):
        dec = np.exp(-beta * (tm - last))
        Au *= dec
        Ad *= dec
        if ty == 0:
            lam = mu + a_s * Au + a_m * Ad
            Au += 1.0
        else:
            lam = mu + a_m * Au + a_s * Ad
            Ad += 1.0
        if lam <= 0.0:
            return 1e18
        ll += np.log(lam)
        last = tm
    comp = 2.0 * mu * T + (a_s + a_m) / beta * np.sum(1.0 - np.exp(-beta * (T - times)))
    return comp - ll


def fit_hawkes2d(t, mid, n_max=25000):
    """Extract up/down mid-jump times and MLE-fit the symmetric 2-D Hawkes."""
    from scipy.optimize import minimize
    dm = np.diff(mid)
    tt = t[1:]
    up = tt[dm > 0]
    dn = tt[dm < 0]
    # cap events to bound runtime; rebase to start at 0
    n = min(n_max, len(up) + len(dn))
    cut = np.sort(np.concatenate([up, dn]))[:n][-1] if n > 0 else 0.0
    up = up[up <= cut]
    dn = dn[dn <= cut]
    t0 = min(up.min() if len(up) else cut, dn.min() if len(dn) else cut)
    up, dn = up - t0, dn - t0
    T = float(cut - t0)
    rate = (len(up) + len(dn)) / max(T, 1e-9)
    x0 = [0.4 * rate, 0.2, 0.2, 1.0]
    bnds = [(1e-6, None), (1e-6, 20.0), (1e-6, 20.0), (1e-4, 200.0)]
    res = minimize(hawkes2d_negloglik, x0, args=(up, dn, T), method="L-BFGS-B", bounds=bnds)
    mu, a_s, a_m, beta = res.x
    return dict(mu=mu, alpha_s=a_s, alpha_m=a_m, beta=beta,
                margin=a_m - a_s, n_up=int(len(up)), n_dn=int(len(dn)),
                T=T, success=bool(res.success))


__all__ = [
    "TICK",
    "sample_grid",
    "acf",
    "xcorr",
    "return_acf",
    "realized_var_rate",
    "signature_curve",
    "mid_locf",
    "updown_per_bin",
    "paper_signature",
    "hawkes2d_negloglik",
    "fit_hawkes2d",
]
