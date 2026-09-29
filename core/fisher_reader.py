
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import AveragedSpectrum


_RANGE_RE = re.compile(r"\[\s*([0-9.]+)\s*-\s*([0-9.]+)\s*\]")


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _parse_full_mz_range(filter_string: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not filter_string:
        return None, None
    m = _RANGE_RE.search(filter_string)
    if not m:
        return None, None
    return _safe_float(m.group(1)), _safe_float(m.group(2))


def _get_ms_scan_arrays(raw, ms_filter: str) -> Tuple[np.ndarray, np.ndarray]:

    mf = (ms_filter or "ms").lower()
    if mf in ("ms2", "msms"):
        scans = getattr(raw, "_ms2_scan_numbers", None)
        rts = getattr(raw, "_ms2_retention_times", None)
    else:
        scans = getattr(raw, "_ms1_scan_numbers", None)
        rts = getattr(raw, "_ms1_retention_times", None)

    if scans is None or rts is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    try:
        scans = np.asarray(scans, dtype=int)
        rts = np.asarray(rts, dtype=float)
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=float)

    if scans.size != rts.size:
        return np.array([], dtype=int), np.array([], dtype=float)

    return scans, rts


def _pick_scan_range_by_tic(raw, *, ms_filter: str, avg_scans: int) -> Tuple[List[int], Optional[float], Optional[float]]:

    from fisher_py.data.business import TraceType

    ms_filter = (ms_filter or "ms").lower()
    avg_scans = max(1, int(avg_scans))

    scans, rts = _get_ms_scan_arrays(raw, ms_filter)

    
    if scans.size == 0:
        first = int(getattr(raw, "first_scan", 1))
        last = int(getattr(raw, "last_scan", first))
        scans = np.arange(first, last + 1, dtype=int)
        
        rt_list: List[float] = []
        for sn in scans:
            try:
                rt_list.append(float(raw.get_retention_time_from_scan_number(int(sn))))
            except Exception:
                rt_list.append(float("nan"))
        rts = np.asarray(rt_list, dtype=float)

    # TIC apex RT
    rt_apex: Optional[float] = None
    try:
        rt_tic, ic_tic = raw.get_chromatogram(0.0, 0.0, TraceType.TIC, ms_filter=ms_filter)
        rt_tic = np.asarray(rt_tic, dtype=float)
        ic_tic = np.asarray(ic_tic, dtype=float)
        if rt_tic.size and ic_tic.size:
            rt_apex = float(rt_tic[int(np.argmax(ic_tic))])
    except Exception:
        rt_apex = None

    if rt_apex is None:
        
        rt_apex = float(rts[int(rts.size // 2)]) if rts.size else 0.0

    
    if rts.size and np.isfinite(rts).any():
        idx = int(np.nanargmin(np.abs(rts - rt_apex)))
    else:
        idx = 0

    half = avg_scans // 2
    start_idx = max(0, idx - half)
    end_idx = min(scans.size - 1, start_idx + avg_scans - 1)
    start_idx = max(0, end_idx - (avg_scans - 1))

    sel_scans = [int(x) for x in scans[start_idx : end_idx + 1].tolist()]
    rt_start = float(rts[start_idx]) if rts.size else None
    rt_end = float(rts[end_idx]) if rts.size else None

    return sel_scans, rt_start, rt_end


def extract_averaged_spectrum_from_raw(
    raw_path: Path,
    *,
    avg_scans: int = 8,
    bin_decimals: int = 4,
    ms_filter: str = "ms",
) -> AveragedSpectrum:

    try:
        from fisher_py import RawFile
    except Exception as e:
        raise RuntimeError(
            f'Could not import fisher_py for Thermo RAW access.\nInstall the supported backend in this environment:\n  python -m pip install fisher-py==1.0.22\n\nOriginal error: {e}'
        )

    raw = RawFile(str(raw_path))

    
    scan_numbers, rt_start, rt_end = _pick_scan_range_by_tic(raw, ms_filter=ms_filter, avg_scans=avg_scans)
    if not scan_numbers:
        raise RuntimeError('No eligible RAW scan was found; check file integrity and the MS filter.')

    
    filter_string: Optional[str] = None
    try:
        mid_scan = int(scan_numbers[len(scan_numbers) // 2])
        filter_string = raw.get_scan_event_str_from_scan_number(mid_scan)
    except Exception:
        filter_string = None

    full_mz_min, full_mz_max = _parse_full_mz_range(filter_string)

    
    bins: Dict[float, float] = {}
    for sn in scan_numbers:
        mz, inten, charges, event_str = raw.get_scan_from_scan_number(int(sn))
        mz = np.asarray(mz, dtype=float)
        inten = np.asarray(inten, dtype=float)
        if mz.size == 0 or inten.size == 0:
            continue

        # rounding bin
        for m, y in zip(mz.tolist(), inten.tolist()):
            key = round(float(m), int(bin_decimals))
            bins[key] = bins.get(key, 0.0) + float(y)

    if not bins:
        raise RuntimeError('The selected scan contains no valid spectrum.')

    mz_sorted = np.array(sorted(bins.keys()), dtype=float)
    inten_avg = np.array([bins[k] / float(len(scan_numbers)) for k in mz_sorted.tolist()], dtype=float)

    
    acq_dt: Optional[datetime] = None
    try:
        p = Path(raw_path)
        if p.exists():
            acq_dt = datetime.fromtimestamp(p.stat().st_mtime)
    except Exception:
        acq_dt = None

    return AveragedSpectrum(
        mz=mz_sorted,
        intensity=inten_avg,
        scan_start=int(scan_numbers[0]),
        scan_end=int(scan_numbers[-1]),
        rt_start_min=_safe_float(rt_start),
        rt_end_min=_safe_float(rt_end),
        filter_string=filter_string,
        full_mz_min=full_mz_min,
        full_mz_max=full_mz_max,
        acquisition_datetime=acq_dt,
        raw_path=str(raw_path),
        avg_scans=len(scan_numbers),
    )
