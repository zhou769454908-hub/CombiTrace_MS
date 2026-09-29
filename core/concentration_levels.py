"""Utilities for deterministic concentration decile labels."""
from __future__ import annotations
import math
from typing import Iterable, List, Tuple
import numpy as np


def concentration_deciles(values: Iterable[object]) -> Tuple[List[object], List[object]]:
    """Return deciles 1..10 (low to high) and labels L01..L10.

    Invalid values receive blank labels. Stable ordering is used for ties so
    every output row is assigned exactly one level when a positive finite value
    exists.
    """
    raw = list(values)
    valid = []
    for i, v in enumerate(raw):
        try:
            x = float(v)
            if math.isfinite(x) and x > 0:
                valid.append((i, x))
        except Exception:
            pass
    deciles: List[object] = [""] * len(raw)
    labels: List[object] = [""] * len(raw)
    if not valid:
        return deciles, labels
    order = np.argsort(np.asarray([v for _, v in valid], dtype=float), kind="mergesort")
    n = len(valid)
    for rank0, pos in enumerate(order.tolist()):
        row_idx = valid[int(pos)][0]
        # Equal-size ordinal bins; 1 is lowest and 10 is highest.
        level = min(10, max(1, int(math.ceil((rank0 + 1) * 10.0 / n))))
        deciles[row_idx] = level
        labels[row_idx] = f"L{level:02d}"
    return deciles, labels


def add_decile_columns(rows, concentration_key: str, *, prefix: str = "Concentration"):
    values = [r.get(concentration_key, "") for r in rows]
    deciles, labels = concentration_deciles(values)
    for i, row in enumerate(rows):
        row[f"{prefix}_decile_1_to_10"] = deciles[i]
        row[f"{prefix}_level"] = labels[i]
    return rows
