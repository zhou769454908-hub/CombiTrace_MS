"""Shared row-limit policy. A maximum of 100 is not a requirement to have 100."""
from __future__ import annotations
from typing import Mapping, Sequence

MAX_TRAINING_ROWS = 100


def is_nonblank_record(record: Mapping[str, object]) -> bool:
    return any(str(v).strip() for k, v in record.items()
               if not str(k).startswith('__') and v is not None)


def validate_training_count(records: Sequence[Mapping[str, object]], *, minimum: int = 0,
                            label: str = 'Training table') -> int:
    n = sum(is_nonblank_record(r) for r in records)
    if n > MAX_TRAINING_ROWS:
        raise ValueError(
            f'{label}: {n} nonblank data rows exceed the maximum of {MAX_TRAINING_ROWS}. Up to 100 nonempty calibration rows are accepted, including 98, 99 or 100. Rows are not silently truncated.')
    if n < minimum:
        raise ValueError(f'{label}: {n} rows; at least {minimum} usable rows are needed.')
    return n
