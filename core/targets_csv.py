'- file / raw / filename / sample'

from __future__ import annotations
from .legacy_schema import canonical_header

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class TargetSpec:
    file_key: str  
    name: str
    formula: str
    adduct: str
    polarity: str
    theoretical_mz: Optional[float]
    ppm: Optional[float]

    
    raw_row: Dict[str, str]


def _try_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _normalize_file_key(val: str) -> str:
    s = str(val or "").strip()
    if not s:
        return ""
    
    try:
        p = Path(s)
        s = p.stem  
    except Exception:
        
        if s.lower().endswith(".raw"):
            s = s[:-4]
    return s.strip().lower()


def _detect_col(header_lower: List[str], candidates: Iterable[str]) -> Optional[str]:
    cand = [c.lower() for c in candidates]
    for c in cand:
        if c in header_lower:
            return c
    return None


def read_targets_csv(csv_path: Path) -> List[Dict[str, str]]:

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
            continue

    raise RuntimeError(f'Could not read CSV (tried {encodings} encodings): {csv_path}\nLast error: {last_err}')


def load_targets_mapping(csv_path: Path) -> Dict[str, List[TargetSpec]]:

    rows = read_targets_csv(csv_path)
    if not rows:
        raise RuntimeError('Targets CSV is empty.')

    headers = list(rows[0].keys())
    headers_lower = [canonical_header(h).strip().lower() for h in headers]
    lower_to_orig = {canonical_header(h).strip().lower(): h for h in headers}

    file_col = _detect_col(
        headers_lower,
        [
            "file",
            "filename",
            "raw",
            "rawfile",
            "raw_file",
            "sample",
            "sample_name",
            "sampleid",
            "sample_serial_number",
            "sample serial number",
        ],
    )

    if not file_col:
        raise RuntimeError(
            'Targets CSV has no file column. Use a header such as file, raw, filename or sample.'
        )

    name_col = _detect_col(headers_lower, ["name", "compound", "compound_name", "target", "analyte"])
    formula_col = _detect_col(headers_lower, ["formula", "molecular_formula", "mf"])
    
    adduct_col = None
    polarity_col = None
    theo_col = _detect_col(headers_lower, ["theoretical_mz", "theory_mz", "target_mz", "mz", "exact_mz"])
    ppm_col = _detect_col(headers_lower, ["ppm", "tolerance_ppm", "ppm_tol"])

    mapping: Dict[str, List[TargetSpec]] = {}

    for r in rows:
        fk = _normalize_file_key(r.get(lower_to_orig[file_col], ""))
        if not fk:
            
            continue

        name = r.get(lower_to_orig[name_col], "") if name_col else ""
        formula = r.get(lower_to_orig[formula_col], "") if formula_col else ""
        adduct = ""  # auto
        polarity = ""  # auto
        theo_mz = _try_float(r.get(lower_to_orig[theo_col], "")) if theo_col else None
        ppm = _try_float(r.get(lower_to_orig[ppm_col], "")) if ppm_col else None

        if not formula and theo_mz is None:
            
            continue

        spec = TargetSpec(
            file_key=fk,
            name=name,
            formula=formula,
            adduct=adduct,
            polarity=polarity,
            theoretical_mz=theo_mz,
            ppm=ppm,
            raw_row=r,
        )
        mapping.setdefault(fk, []).append(spec)

    return mapping