"""Runnable scripts and reusable batch routines for ``research_core``.

Modules here double as command-line entry points (``python -m
research_core.scripts.<name>``) and as importable libraries. The quintet
notebook, for example, imports the simulated-annealing helpers from
``build_calibration_quintets`` directly instead of loading the file by path.
"""
