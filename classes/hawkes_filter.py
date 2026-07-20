"""Online multivariate Hawkes intensity filter for replay-based backtesting.

Maintains the same recursive auxiliary state as
:meth:`research_core.classes.simulate.Simulate._record_event`:
for each kernel component ``k``,

    A^(k)_{:, j}(t+)  =  exp(-β^(k) Δt) ⊙ A^(k)_{:, j}(t-) + e_j[event]

so that the conditional intensity at any later time ``t`` is

    λ_i(t)  =  μ_i  +  Σ_k Σ_j  α^(k)_{ij} β^(k)_{ij} A^(k)_{ij}(t).

The filter is fed by the empirical / simulated event stream that
:class:`research_core.classes.backtest.MMBacktester` already replays, so
an MM agent can read live Hawkes intensities during a backtest without
running a parallel simulator.  Event types are mapped to kernel
dimensions using the same convention as ``Simulate``:

    0 = MO_bid (aggressive buy)
    1 = MO_ask (aggressive sell)
    2 = LO_bid
    3 = LO_ask
    4 = CXL_bid
    5 = CXL_ask
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_LABELS: Tuple[str, ...] = (
    "MO_bid", "MO_ask", "LO_bid", "LO_ask", "CXL_bid", "CXL_ask",
)


class HawkesFilter:
    """Online multivariate Hawkes intensity filter."""

    def __init__(
        self,
        baseline: np.ndarray,
        adjacency_list: Sequence[np.ndarray],
        decays_list: Sequence[np.ndarray],
        labels: Sequence[str] = DEFAULT_LABELS,
    ):
        baseline = np.asarray(baseline, dtype=np.float64).copy()
        adj = [np.asarray(a, dtype=np.float64).copy() for a in adjacency_list]
        dec = [np.asarray(b, dtype=np.float64).copy() for b in decays_list]

        if len(adj) != len(dec):
            raise ValueError(
                "adjacency_list and decays_list must have the same length "
                f"(got {len(adj)} and {len(dec)})"
            )
        if len(adj) == 0:
            raise ValueError("at least one kernel component is required")

        d = baseline.shape[0]
        for k, (a, b) in enumerate(zip(adj, dec)):
            if a.shape != (d, d) or b.shape != (d, d):
                raise ValueError(
                    f"kernel {k}: adjacency/decay shapes {a.shape}/{b.shape} "
                    f"do not match baseline dim {d}"
                )
        if len(labels) != d:
            raise ValueError(
                f"labels length {len(labels)} ≠ baseline dim {d}"
            )

        self.baseline = baseline
        self.adjacency_list = adj
        self.decays_list = dec
        self.labels: Tuple[str, ...] = tuple(labels)
        self.label_to_dim = {lbl: i for i, lbl in enumerate(self.labels)}
        self.n_kernels = len(adj)
        self.d = d
        self._A_list: List[np.ndarray] = [
            np.zeros((d, d), dtype=np.float64) for _ in range(self.n_kernels)
        ]
        self._t_last: float = 0.0

    @classmethod
    def kghm_multivariate_single(cls, alpha_scale: float = 0.9) -> "HawkesFilter":
        """Single-kernel multivariate KGHM Hawkes filter from ``Simulate``."""
        from .simulate import Simulate

        sim = Simulate(T=1, alpha_scale=float(alpha_scale))
        return cls.from_simulate(sim)

    @classmethod
    def from_simulate(cls, sim) -> "HawkesFilter":
        return cls(
            sim._baseline,
            sim._adjacency_list,
            sim._decays_list,
            sim.labels,
        )

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "HawkesFilter":
        """Load calibrated parameters written by ``HawkesCalibration.save_params``."""
        params = pickle.loads(Path(path).read_bytes())
        if "multi_single_tau" in params:
            params = params["multi_single_tau"]
        adjacency = np.asarray(params["adjacency"], dtype=np.float64)
        decays = np.asarray(params["decays"], dtype=np.float64)
        if adjacency.ndim == 2:
            adjacency_list = [adjacency]
            decays_list = [decays]
        else:
            adjacency_list = [adjacency[k] for k in range(adjacency.shape[0])]
            decays_list = [decays[k] for k in range(decays.shape[0])]
        return cls(params["baseline"], adjacency_list, decays_list)

    def clone(self) -> "HawkesFilter":
        return HawkesFilter(
            self.baseline,
            self.adjacency_list,
            self.decays_list,
            self.labels,
        )

    def reset(self, t0: float = 0.0) -> None:
        for k in range(self.n_kernels):
            self._A_list[k].fill(0.0)
        self._t_last = float(t0)

    def update(self, t: float, dim_idx: int) -> None:
        t = float(t)
        dim_idx = int(dim_idx)
        if not (0 <= dim_idx < self.d):
            raise IndexError(
                f"dim_idx {dim_idx} out of range for d={self.d}"
            )
        dt = t - self._t_last
        if dt < 0.0:
            dt = 0.0
        for k in range(self.n_kernels):
            self._A_list[k] *= np.exp(-self.decays_list[k] * dt)
            self._A_list[k][:, dim_idx] += 1.0
        self._t_last = t

    def update_label(self, t: float, label: str) -> None:
        self.update(t, self.label_to_dim[label])

    def intensity(self, t: float) -> np.ndarray:
        contribs = self.intensity_decomposed(t)
        out = self.baseline + contribs.sum(axis=0)
        np.maximum(out, 0.0, out=out)
        return out

    def intensity_decomposed(self, t: float) -> np.ndarray:
        dt = float(t) - self._t_last
        if dt < 0.0:
            dt = 0.0
        out = np.empty((self.n_kernels, self.d), dtype=np.float64)
        for k in range(self.n_kernels):
            A_decayed = self._A_list[k] * np.exp(-self.decays_list[k] * dt)
            out[k] = np.sum(
                self.adjacency_list[k] * self.decays_list[k] * A_decayed,
                axis=1,
            )
        return out

    def integrated_intensity(self, t0: float, t1: float) -> np.ndarray:
        t0 = max(float(t0), self._t_last)
        t1 = max(float(t1), t0)
        if t1 <= t0:
            return np.zeros(self.d, dtype=np.float64)
        dt_anchor = t0 - self._t_last
        dt = t1 - t0
        out = self.baseline * dt
        for k in range(self.n_kernels):
            B = self.decays_list[k]
            if dt_anchor > 0.0:
                A0 = self._A_list[k] * np.exp(-B * dt_anchor)
            else:
                A0 = self._A_list[k]
            factor = 1.0 - np.exp(-B * dt)
            out += np.sum(self.adjacency_list[k] * factor * A0, axis=1)
        np.maximum(out, 0.0, out=out)
        return out

    @property
    def t_last(self) -> float:
        return self._t_last

    def A_snapshot(self) -> List[np.ndarray]:
        return [A.copy() for A in self._A_list]


def classify_event(event_type: str, side) -> int:
    et = str(event_type)
    try:
        s = int(side)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown side: {side!r}") from exc

    if et == "LO":
        if s == 1:
            return 2
        if s == 2:
            return 3
    elif et == "CXL":
        if s == 1:
            return 4
        if s == 2:
            return 5
    elif et == "MO":
        if s == 1:
            return 0
        if s == 2:
            return 1
    raise ValueError(f"unknown (event_type={event_type!r}, side={side!r})")


def classify_mo(mo_side) -> int:
    if isinstance(mo_side, (bytes, bytearray)):
        mo_side = mo_side.decode("ascii", errors="ignore")
    if isinstance(mo_side, str):
        low = mo_side.strip().lower()
        if low == "buy":
            return 0
        if low == "sell":
            return 1
    else:
        try:
            s = int(mo_side)
        except (TypeError, ValueError):
            s = None
        if s == 1:
            return 0
        if s == 2:
            return 1
    raise ValueError(f"unknown mo_side: {mo_side!r}")


HawkesFilterFactory = Callable[[], HawkesFilter]


def default_hawkes_params_path() -> Path:
    from research_core.classes.helpers import resolve_data_path

    path = resolve_data_path("multivariate_hawkes_params_KGHM.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated Hawkes parameters not found at {path}. Run the final "
            f"'Save calibrated parameters' cell of notebooks/calibration.ipynb "
            f"to generate it."
        )
    return path


def resolve_filter_factory(
    spec: Union[None, bool, HawkesFilter, HawkesFilterFactory],
) -> Optional[HawkesFilterFactory]:
    if spec is None or spec is False:
        return None
    if spec is True:
        return lambda: HawkesFilter.from_pickle(default_hawkes_params_path())
    if isinstance(spec, HawkesFilter):
        template = spec
        return lambda: template.clone()
    if callable(spec):
        return spec
    raise TypeError(
        "hawkes must be None, bool, HawkesFilter, or a zero-arg callable; "
        f"got {type(spec).__name__}"
    )
