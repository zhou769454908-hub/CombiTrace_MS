
from __future__ import annotations
from .legacy_schema import canonical_header

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from .chemistry import format_formula_hill, parse_formula


@dataclass
class FilterStats:
    total_target_rows: int = 0
    kept_rows: int = 0
    removed_rows: int = 0
    ref_unique_formulas: int = 0
    ref_rows_with_formula: int = 0
    ref_parse_errors: int = 0
    target_parse_errors: int = 0
    target_blank_formula: int = 0


def read_csv_header(path: Path) -> List[str]:

    encodings = ["utf-8-sig", "utf-8", "gbk", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        return [str(x).strip() for x in row]
                return []
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV header: {path} ({last_err})")


def guess_formula_column(fieldnames: Sequence[str]) -> Optional[str]:

    if not fieldnames:
        return None
    candidates = [
        "formula",
        "molecular_formula",
        "chemical_formula",
        "mf",
        "sum_formula",
    ]
    lower_map = {canonical_header(n).strip().lower(): str(n).strip() for n in fieldnames}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    for k, v in lower_map.items():
        if "formula" in k:
            return v
    return None


def normalize_formula_str(formula: str) -> str:

    s = str(formula).strip()
    if not s:
        return ""
    try:
        counts = parse_formula(s)
        return format_formula_hill(counts)
    except Exception:
        return s


def load_formula_set(
    ref_csv: Path,
    *,
    formula_col: str,
    mode: str = "canonical",  # canonical | exact
) -> Tuple[Set[str], FilterStats]:

    stats = FilterStats()
    s: Set[str] = set()

    encodings = ["utf-8-sig", "utf-8", "gbk", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with ref_csv.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise RuntimeError("CSV has no header")
                for row in reader:
                    val = (row.get(formula_col) or "").strip()
                    if not val:
                        continue
                    stats.ref_rows_with_formula += 1
                    if mode == "exact":
                        key = val
                    else:
                        key = normalize_formula_str(val)
                        try:
                            parse_formula(val)
                        except Exception:
                            stats.ref_parse_errors += 1
                    if key:
                        s.add(key)
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise RuntimeError(f"Failed to read reference CSV: {ref_csv} ({last_err})")

    stats.ref_unique_formulas = len(s)
    return s, stats


def filter_csv_remove_matching_formulas(
    *,
    ref_csv: Path,
    target_csv: Path,
    out_csv: Path,
    ref_formula_col: str,
    target_formula_col: str,
    mode: str = "canonical",  # canonical | exact
    removed_csv: Optional[Path] = None,
) -> FilterStats:

    ref_set, stats = load_formula_set(ref_csv, formula_col=ref_formula_col, mode=mode)

    encodings = ["utf-8-sig", "utf-8", "gbk", "latin-1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            with target_csv.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise RuntimeError("Target CSV has no header")
                fieldnames = list(reader.fieldnames)

                out_csv.parent.mkdir(parents=True, exist_ok=True)
                out_f = out_csv.open("w", encoding="utf-8-sig", newline="")
                out_writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                out_writer.writeheader()

                rem_f = None
                rem_writer = None
                if removed_csv is not None:
                    removed_csv.parent.mkdir(parents=True, exist_ok=True)
                    rem_f = removed_csv.open("w", encoding="utf-8-sig", newline="")
                    rem_writer = csv.DictWriter(rem_f, fieldnames=fieldnames)
                    rem_writer.writeheader()

                try:
                    for row in reader:
                        stats.total_target_rows += 1
                        val = (row.get(target_formula_col) or "").strip()
                        if not val:
                            stats.target_blank_formula += 1
                            out_writer.writerow(row)
                            stats.kept_rows += 1
                            continue
                        if mode == "exact":
                            key = val
                        else:
                            key = normalize_formula_str(val)
                            try:
                                parse_formula(val)
                            except Exception:
                                stats.target_parse_errors += 1

                        if key in ref_set:
                            stats.removed_rows += 1
                            if rem_writer is not None:
                                rem_writer.writerow(row)
                        else:
                            out_writer.writerow(row)
                            stats.kept_rows += 1
                finally:
                    out_f.close()
                    if rem_f is not None:
                        rem_f.close()
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise RuntimeError(f"Failed to filter target CSV: {target_csv} ({last_err})")

    return stats


__all__ = [
    "FilterStats",
    "read_csv_header",
    "guess_formula_column",
    "normalize_formula_str",
    "filter_csv_remove_matching_formulas",
]
