'C = C_A + C_B + C_C\n      H = H_A + H_B + H_C - 1\n      N = N_A + N_B + N_C - 2\n      O = O_A + O_B + O_C + 1\n      P = P_A + P_B + P_C + 1'

from __future__ import annotations
from .legacy_schema import canonical_header

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .chemistry import SUPPORTED_ELEMENTS, format_formula_hill, monoisotopic_mass, parse_formula


@dataclass(frozen=True)
class ComponentRow:

    name: str
    formula_input: str
    counts: Dict[str, int]
    exact_mass: float


def _detect_col(header_lower: List[str], candidates: Iterable[str]) -> Optional[str]:
    cand = [c.lower() for c in candidates]
    for c in cand:
        if c in header_lower:
            return c
    return None


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows: List[Dict[str, str]] = []
                for r in reader:
                    rows.append({(k or "").strip(): ("" if v is None else str(v)).strip() for k, v in r.items()})
                return rows
        except Exception as e:
            last_err = e
    raise RuntimeError(f'Could not read CSV: {csv_path} (encodings attempted: {encodings})\nLast error: {last_err}')


def load_component_csv(csv_path: Path) -> List[ComponentRow]:

    rows = _read_csv_rows(csv_path)
    if not rows:
        return []

    headers = list(rows[0].keys())
    headers_lower = [canonical_header(h).strip().lower() for h in headers]
    lower_to_orig = {canonical_header(h).strip().lower(): h for h in headers}

    
    formula_col = _detect_col(headers_lower, [
        "formula",
        "chemical formula",
        "chemical_formula",
        "molecular_formula",
        "mf",
    ])
    name_col = _detect_col(headers_lower, [
        "name",
        "compound",
        "compound_name",
        "id",
        "label",
        "title",
    ])

    
    sym_by_lower = {sym.lower(): sym for sym in SUPPORTED_ELEMENTS}
    elem_cols: Dict[str, str] = {}
    for low, orig in lower_to_orig.items():
        if low in sym_by_lower:
            elem_cols[sym_by_lower[low]] = orig

    out: List[ComponentRow] = []
    for r in rows:
        name = r.get(lower_to_orig[name_col], "") if name_col else ""

        counts: Dict[str, int] = {}
        formula_input = ""

        if formula_col:
            formula_input = r.get(lower_to_orig[formula_col], "").strip()
            if not formula_input:
                continue
            counts = parse_formula(formula_input)
        else:
            
            if not elem_cols:
                raise RuntimeError(
                    f'{csv_path.name}: no formula column or elemental-count columns were found.'
                )
            for sym, col in elem_cols.items():
                v = r.get(col, "")
                if v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                try:
                    iv = int(float(s))
                except Exception:
                    continue
                if iv:
                    counts[sym] = int(iv)

            if not counts:
                continue

        
        hill = format_formula_hill(counts)
        mass = monoisotopic_mass(counts)
        out.append(ComponentRow(name=name, formula_input=hill or formula_input, counts=counts, exact_mass=mass))

    return out


def _add_counts(*parts: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in parts:
        for el, cnt in d.items():
            out[el] = out.get(el, 0) + int(cnt)
    return out


def _apply_transform(base: Dict[str, int], *, only_formula: bool) -> Optional[Dict[str, int]]:

    d = dict(base)
    d["H"] = int(d.get("H", 0)) - 1
    d["N"] = int(d.get("N", 0)) - 2
    d["O"] = int(d.get("O", 0)) + 1
    d["P"] = int(d.get("P", 0)) + 1

    if only_formula:
        d = {k: int(d.get(k, 0)) for k in ("C", "H", "N", "O", "P")}

    
    for el, cnt in d.items():
        if int(cnt) < 0:
            return None
    return d


def enumerate_abc(
    a_rows: Sequence[ComponentRow],
    b_rows: Sequence[ComponentRow],
    c_rows: Sequence[ComponentRow],
    *,
    only_formula: bool,
    deduplicate: bool,
) -> Tuple[List[Tuple[str, float, int]], int]:
    'results: List[(formula, exact_mass, count)]'

    if not a_rows or not b_rows or not c_rows:
        return [], 0

    skipped_invalid = 0

    if deduplicate:
        m: Dict[str, Tuple[float, int]] = {}
        for a, b, c in itertools.product(a_rows, b_rows, c_rows):
            base = _add_counts(a.counts, b.counts, c.counts)
            out_counts = _apply_transform(base, only_formula=only_formula)
            if out_counts is None:
                skipped_invalid += 1
                continue
            f = format_formula_hill(out_counts)
            if not f:
                skipped_invalid += 1
                continue
            if f in m:
                mass, cnt = m[f]
                m[f] = (mass, cnt + 1)
            else:
                m[f] = (monoisotopic_mass(out_counts), 1)

        
        results = [(f, m[f][0], m[f][1]) for f in m]
        results.sort(key=lambda x: x[1])
        return results, skipped_invalid

    
    results_all: List[Tuple[str, float, int]] = []
    for a, b, c in itertools.product(a_rows, b_rows, c_rows):
        base = _add_counts(a.counts, b.counts, c.counts)
        out_counts = _apply_transform(base, only_formula=only_formula)
        if out_counts is None:
            skipped_invalid += 1
            continue
        f = format_formula_hill(out_counts)
        if not f:
            skipped_invalid += 1
            continue
        results_all.append((f, monoisotopic_mass(out_counts), 1))

    results_all.sort(key=lambda x: x[1])
    return results_all, skipped_invalid


def write_results_csv(results: Sequence[Tuple[str, float, int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "exact_mass", "count"])
        for formula, mass, cnt in results:
            w.writerow([formula, f"{mass:.6f}", int(cnt)])


# ==========================
# Detailed (keep duplicates)
# ==========================


@dataclass(frozen=True)
class ABCDetailedRow:

    formula: str
    exact_mass: float
    count: int
    combo: str


def _label_component(x: ComponentRow) -> str:
    s = (x.name or "").strip()
    if s:
        return s
    return (x.formula_input or "").strip() or "item"


def enumerate_abc_detailed(
    a_rows: Sequence[ComponentRow],
    b_rows: Sequence[ComponentRow],
    c_rows: Sequence[ComponentRow],
    *,
    only_formula: bool,
) -> Tuple[List[ABCDetailedRow], int]:
    'results: List[ABCDetailedRow]'

    if not a_rows or not b_rows or not c_rows:
        return [], 0

    skipped_invalid = 0
    out: List[ABCDetailedRow] = []

    for ai, a in enumerate(a_rows, start=1):
        la = _label_component(a)
        for bi, b in enumerate(b_rows, start=1):
            lb = _label_component(b)
            for ci, c in enumerate(c_rows, start=1):
                lc = _label_component(c)

                base = _add_counts(a.counts, b.counts, c.counts)
                out_counts = _apply_transform(base, only_formula=only_formula)
                if out_counts is None:
                    skipped_invalid += 1
                    continue

                f = format_formula_hill(out_counts)
                if not f:
                    skipped_invalid += 1
                    continue

                mass = monoisotopic_mass(out_counts)
                
                
                combo = f"A#{ai}:{la} | B#{bi}:{lb} | C#{ci}:{lc}"
                out.append(ABCDetailedRow(formula=f, exact_mass=float(mass), count=1, combo=combo))

    out.sort(key=lambda r: r.exact_mass)
    return out, skipped_invalid


def write_results_csv_detailed(results: Sequence[ABCDetailedRow], out_csv: Path) -> None:

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "exact_mass", "count", "combo"])
        for r in results:
            w.writerow([r.formula, f"{float(r.exact_mass):.6f}", int(r.count), r.combo])
