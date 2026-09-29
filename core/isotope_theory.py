
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .chemistry import Adduct, MONO_MASS, parse_formula, monoisotopic_mass




ISOTOPES: Dict[str, List[Tuple[float, float]]] = {
    "H":  [(1.00782503223, 0.999885), (2.01410177812, 0.000115)],
    "D":  [(2.01410177812, 1.0)],
    "C":  [(12.0, 0.9893), (13.00335483507, 0.0107)],
    "N":  [(14.00307400443, 0.99632), (15.00010889888, 0.00368)],
    "O":  [(15.99491461957, 0.99757), (16.99913175650, 0.00038), (17.99915961286, 0.00205)],
    "S":  [(31.9720711744, 0.9499), (32.9714589098, 0.0075), (33.967867004, 0.0425), (35.96708071, 0.0001)],
    "F":  [(18.99840316273, 1.0)],
    "Na": [(22.9897692820, 1.0)],
    "Mg": [(23.985041697, 0.7899), (24.985836976, 0.10), (25.982592968, 0.1101)],
    "Al": [(26.98153853, 1.0)],
    "Si": [(27.97692653465, 0.92223), (28.9764946649, 0.04685), (29.973770136, 0.03092)],
    "P":  [(30.97376199842, 1.0)],
    "Cl": [(34.968852682, 0.7576), (36.965902602, 0.2424)],
    "K":  [(38.9637064864, 0.932581), (39.963998166, 0.000117), (40.9618252579, 0.067302)],
    "Br": [(78.9183376, 0.5069), (80.9162906, 0.4931)],
    "I":  [(126.9044719, 1.0)],
    
}


def _normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in counts.items():
        try:
            iv = int(v)
        except Exception:
            continue
        if iv:
            out[str(k)] = iv
    return out


def calc_rdb(counts: Dict[str, int]) -> Optional[float]:
    'DBE = C - H/2 - X/2 + N/2 + 1'
    c = int(counts.get("C", 0))
    h = int(counts.get("H", 0)) + int(counts.get("D", 0))  # D counts as H for DBE
    n = int(counts.get("N", 0))
    x = int(counts.get("F", 0)) + int(counts.get("Cl", 0)) + int(counts.get("Br", 0)) + int(counts.get("I", 0))
    if c == 0 and h == 0 and n == 0 and x == 0:
        return None
    return float(c - h / 2.0 - x / 2.0 + n / 2.0 + 1.0)


def isotope_peaks(
    formula: str,
    *,
    max_states: int = 8000,
    max_peaks: int = 12,
    min_rel: float = 0.1,
    merge_decimals: int = 6,
) -> List[Tuple[float, float]]:
    counts = _normalize_counts(parse_formula(formula))
    base_mass = monoisotopic_mass(counts)

    # dist: list of (delta_mass, prob)
    dist: List[Tuple[float, float]] = [(0.0, 1.0)]

    def merge_states(states: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        acc: Dict[float, float] = {}
        for dm, pr in states:
            key = round(float(dm), int(merge_decimals))
            acc[key] = acc.get(key, 0.0) + float(pr)
        return [(k, v) for k, v in acc.items() if v > 0]

    
    for el, n in sorted(counts.items(), key=lambda kv: kv[0]):
        if n <= 0:
            continue
        iso = ISOTOPES.get(el)
        if not iso:
            
            continue
        if len(iso) <= 1:
            continue

        mono_mass_el = iso[0][0]
        deltas = [(m - mono_mass_el, float(p)) for m, p in iso]

        for _ in range(int(n)):
            new: List[Tuple[float, float]] = []
            for dm, pr in dist:
                for d, p in deltas:
                    new.append((dm + d, pr * p))

            
            new = merge_states(new)

            
            new.sort(key=lambda x: x[1], reverse=True)
            if len(new) > int(max_states):
                new = new[: int(max_states)]
            dist = new

    if not dist:
        return [(base_mass, 100.0)]

    masses = np.array([base_mass + dm for dm, _ in dist], dtype=float)
    probs = np.array([pr for _, pr in dist], dtype=float)
    psum = float(np.sum(probs))
    if psum > 0:
        probs = probs / psum

    
    pmax = float(np.max(probs)) if probs.size else 0.0
    rel = (probs / pmax * 100.0) if pmax > 0 else np.zeros_like(probs)

    
    idx = np.argsort(rel)[::-1]
    kept: List[Tuple[float, float]] = []
    for i in idx.tolist():
        if rel[i] < float(min_rel):
            continue
        kept.append((float(masses[i]), float(rel[i])))
        if len(kept) >= int(max_peaks):
            break

    kept.sort(key=lambda x: x[0])
    return kept


def isotope_peaks_mz(
    formula: str,
    adduct: Adduct,
    *,
    max_states: int = 5000,
    max_peaks: int = 12,
    min_rel: float = 0.1,
) -> List[Tuple[float, float]]:
    peaks = isotope_peaks(formula, max_states=max_states, max_peaks=max_peaks, min_rel=min_rel)
    z = abs(int(adduct.charge)) if adduct.charge else 1
    out = []
    for mass, rel in peaks:
        mz = (float(mass) + float(adduct.mass_shift)) / float(z)
        out.append((mz, float(rel)))
    return out


def gaussian_curve(
    x: np.ndarray,
    peak_mz: Sequence[float],
    peak_rel: Sequence[float],
    *,
    fwhm_da: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.asarray([], dtype=float)

    peak_mz = np.asarray(list(peak_mz), dtype=float)
    peak_rel = np.asarray(list(peak_rel), dtype=float)

    if peak_mz.size == 0:
        return np.zeros_like(x)

    fwhm = float(fwhm_da)
    if fwhm <= 0:
        fwhm = 0.01
    sigma = fwhm / 2.35482004503  # FWHM -> sigma

    y = np.zeros_like(x)
    for m, a in zip(peak_mz.tolist(), peak_rel.tolist()):
        y += float(a) * np.exp(-0.5 * ((x - float(m)) / sigma) ** 2)

    
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax > 0:
        y = y / ymax * 100.0
    return y


def choose_default_fwhm(mz_center: float) -> float:
    _ = mz_center
    return 0.01
