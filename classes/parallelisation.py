"""
Parallel Optuna optimisation for Hawkes process calibration.

Designed for Windows where multiprocessing uses 'spawn'.  Each worker
subprocess independently recreates its objective from a serialised data
file, avoiding pickling issues with complex objects.

Coordinator API
---------------
    run_parallel_optuna(data_dict, objective_type, n_workers, n_trials)

Worker entry point (invoked by the coordinator via subprocess)::

    python -m research_core.classes.parallelisation \
        --mode worker --objective-type sumexp \
        --data-file data.pkl --study-name my_study \
        --storage-url sqlite:///study.db --n-trials 50 --worker-id 0
"""

from __future__ import annotations

import argparse
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time as _time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.storages import RDBStorage

os.environ.setdefault("OMP_NUM_THREADS", "1")


# ═══════════════════════════════════════════════════════════════════════════════
# Worker functions — each runs inside a fresh subprocess
# ═══════════════════════════════════════════════════════════════════════════════

def _create_objective(objective_type: str, data: dict):
    """Instantiate the right Optuna objective from *data*.

    Imports from ``calibrate`` are deferred to avoid a circular dependency
    (calibrate imports parallelisation at the top level).  These only execute
    inside worker subprocesses, where no circular import can occur.
    """
    if objective_type == "sumexp":
        from research_core.classes.calibrate import _SumExpObjective
        return _SumExpObjective(
            beta_ranges=data["beta_ranges"],
            penalty=data["penalty"],
            C=data["C"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            events=data["events"],
            end_times=data["end_times"],
            slow_self_floor=data.get("slow_self_floor"),
            rho_target=data.get("rho_target", 0.95),
        )

    if objective_type == "sumexp_self":
        from research_core.classes.calibrate import _SumExpSelfObjective
        return _SumExpSelfObjective(
            beta_ranges=data["beta_ranges"],
            penalty=data["penalty"],
            C=data["C"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            events=data["events"],
            end_times=data["end_times"],
        )

    if objective_type == "single":
        from research_core.classes.calibrate import _MutualObjective
        return _MutualObjective(
            beta_min=data["beta_min"],
            beta_max=data["beta_max"],
            n_nodes=data["n_nodes"],
            marks_order=data["marks_order"],
            MAX_ITER=data["MAX_ITER"],
            TOL=data["TOL"],
            events=data["events_dense"],
            end_times=data["end_times_array"],
        )

    raise ValueError(f"Unknown objective_type: {objective_type!r}")


def _run_worker(
    storage_url: str,
    study_name: str,
    data_file: str,
    objective_type: str,
    n_trials: int,
):
    """Load data, build objective, run trials against the shared study."""
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    objective = _create_objective(objective_type, data)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinator — called from the main process (e.g. calibrate.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_parallel_optuna(
    data_dict: dict,
    objective_type: str,
    n_workers: int,
    n_trials: int,
    study_name: Optional[str] = None,
) -> dict:
    """Run an Optuna study across *n_workers* subprocesses.

    Parameters
    ----------
    data_dict : dict
        Serialisable keyword arguments for the objective constructor.
    objective_type : str
        ``"sumexp"``, ``"sumexp_self"``, or ``"single"``.
    n_workers : int
        Number of parallel worker processes.
    n_trials : int
        Total number of Optuna trials (split across workers).
    study_name : str, optional
        Defaults to a timestamped name.

    Returns
    -------
    dict with ``best_params`` (dict) and ``best_value`` (float).
    """
    tmp = Path(tempfile.gettempdir())
    ts = int(_time.time() * 1000)
    if study_name is None:
        study_name = f"hawkes_{objective_type}_{ts}"

    data_file = tmp / f"optuna_data_{ts}.pkl"
    output_file = tmp / f"optuna_results_{ts}.pkl"
    db_path = tmp / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"

    with open(data_file, "wb") as f:
        pickle.dump(data_dict, f)

    storage = RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    with storage.engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")

    trials_per_worker = int(np.ceil(n_trials / n_workers))

    # ── spawn workers ─────────────────────────────────────────────
    processes = []
    for i in range(n_workers):
        cmd = [
            sys.executable, "-m", "research_core.classes.parallelisation",
            "--mode", "worker",
            "--objective-type", objective_type,
            "--data-file", str(data_file),
            "--study-name", study_name,
            "--storage-url", storage_url,
            "--n-trials", str(trials_per_worker),
            "--worker-id", str(i),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        processes.append((i, proc))

    print(f"Launched {n_workers} workers "
          f"({trials_per_worker} trials each, {objective_type})")

    # ── merge stdout via threads ──────────────────────────────────
    out_q: queue.Queue = queue.Queue()

    def _reader(wid, p):
        try:
            for line in p.stdout:
                out_q.put((wid, line))
        finally:
            out_q.put((wid, None))

    for wid, proc in processes:
        t = threading.Thread(target=_reader, args=(wid, proc), daemon=True)
        t.start()

    done = 0
    while done < len(processes):
        try:
            wid, line = out_q.get(timeout=1.0)
            if line is None:
                done += 1
            else:
                print(f"[W{wid}] {line}", end="", flush=True)
        except queue.Empty:
            pass

    for wid, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            print(f"Worker {wid} exited with code {proc.returncode}")

    # ── collect results ───────────────────────────────────────────
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    print(f"\nCompleted {len(completed)} / {n_trials} trials")

    if not completed:
        results: Dict = {
            "best_params": None,
            "best_value": None,
            "error": "No trials completed",
        }
    else:
        best = study.best_trial
        results = {"best_params": best.params, "best_value": best.value}
        print(f"Best score: {best.value:.6f}")

    # ── cleanup ───────────────────────────────────────────────────
    for p in (data_file, db_path):
        try:
            p.unlink()
        except Exception:
            pass

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point for worker subprocesses
# ═══════════════════════════════════════════════════════════════════════════════

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["worker"], required=True)
    parser.add_argument("--objective-type", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage-url", required=True)
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args()

    print(f"Worker {args.worker_id} starting "
          f"({args.n_trials} trials, {args.objective_type})", flush=True)
    try:
        _run_worker(
            args.storage_url, args.study_name,
            args.data_file, args.objective_type, args.n_trials,
        )
        print(f"Worker {args.worker_id} completed", flush=True)
    except Exception as exc:
        print(f"Worker {args.worker_id} error: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    _main()
