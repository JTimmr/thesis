"""Parallel Optuna optimisation for Hawkes calibration.

The search runs across several worker subprocesses that share one
SQLite-backed Optuna study. This is the practical way to parallelise on
Windows, where multiprocessing uses spawn: each worker re-imports the package,
rebuilds its objective from a pickled data file, and writes its trials back to
the shared store. Pickling the objective directly would drag the whole event
dataset across the process boundary, so the coordinator passes a data file.

``run_parallel_optuna`` is the coordinator, called from ``HawkesCalibration``
for the single-exponential (``"single"``) path. The module also serves as the worker entry
point::

    python -m research_core.classes.parallelisation \
        --mode worker --objective-type single \
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
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.storages import RDBStorage

os.environ.setdefault("OMP_NUM_THREADS", "1")


# --- Worker functions ---

def create_objective(objective_type: str, data: dict):
    """Instantiate the right Optuna objective from *data*.

    Imports from ``calibrate`` are deferred to avoid a circular import
    (``calibrate`` imports this module at the top level). They run only inside
    worker subprocesses, where the circular import cannot occur.
    """
    if objective_type == "single":
        from research_core.classes.calibrate import MultivariateHawkesObjective
        return MultivariateHawkesObjective(
            beta_min=data["beta_min"],
            beta_max=data["beta_max"],
            n_nodes=data["n_nodes"],
            marks_order=data["marks_order"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            events=data["events_dense"],
            end_times=data["end_times_array"],
        )

    raise ValueError(f"Unknown objective_type: {objective_type!r}")


def run_worker(storage_url: str, study_name: str, data_file: str, objective_type: str, n_trials: int):
    """Load data, build objective, run trials against the shared study."""
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    objective = create_objective(objective_type, data)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


def read_worker_stdout(worker_id: int, process, out_q: queue.Queue) -> None:
    """Copy one worker's stdout into the coordinator queue."""
    try:
        if process.stdout is None:
            return
        for line in process.stdout:
            out_q.put((worker_id, line))
    finally:
        out_q.put((worker_id, None))


# --- Coordinator ---

def run_parallel_optuna(data_dict: dict, objective_type: str, n_workers: int, n_trials: int, study_name: Optional[str] = None) -> dict:
    """Run an Optuna study across *n_workers* subprocesses.

    Parameters
    ----------
    data_dict : dict
        Serialisable keyword arguments for the objective constructor.
    objective_type : str
        ``"single"`` (production single-exponential).
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
    temp_dir = Path(tempfile.gettempdir())
    ts = int(time.time() * 1000)
    if study_name is None:
        study_name = f"hawkes_{objective_type}_{ts}"

    data_file = temp_dir / f"optuna_data_{ts}.pkl"
    db_path = temp_dir / f"{study_name}.db"
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

    # Spawn workers.
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

    # Merge worker stdout without blocking the coordinator loop.
    out_q: queue.Queue = queue.Queue()

    for wid, proc in processes:
        t = threading.Thread(
            target=read_worker_stdout,
            args=(wid, proc, out_q),
            daemon=True,
        )
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

    # Collect results.
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

    # Cleanup.
    for p in (data_file, db_path):
        p.unlink(missing_ok=True)

    return results


# --- CLI entry point ---

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
        run_worker(
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
