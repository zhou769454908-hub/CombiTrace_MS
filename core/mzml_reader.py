
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import AveragedSpectrum, ScanInfo
from .mzml_minireader import iter_spectra, read_run_start_time
from .utils import parse_mz_range_from_filter


def collect_ms1_scan_infos(mzml_path: Path) -> Tuple[List[ScanInfo], Optional[str]]:
    '- scan_infos'

    scan_infos: List[ScanInfo] = []

    
    fallback_scan = 0

    best_tic = -1.0
    best_filter: Optional[str] = None

    for rec in iter_spectra(mzml_path, decode_arrays=False, decode_intensity_if_tic_missing=True):
        if rec.ms_level != 1:
            continue

        fallback_scan += 1
        scan_no = rec.scan_number if rec.scan_number is not None else fallback_scan
        tic = float(rec.tic) if rec.tic is not None else 0.0

        scan_infos.append(
            ScanInfo(
                scan_number=int(scan_no),
                rt_min=rec.rt_min,
                tic=tic,
                filter_string=rec.filter_string,
            )
        )

        if tic > best_tic:
            best_tic = tic
            best_filter = rec.filter_string

    return scan_infos, best_filter


def choose_scan_window(scan_infos: List[ScanInfo], avg_scans: int) -> List[ScanInfo]:
    if not scan_infos:
        raise ValueError('No MS1 spectrum found in mzML.')

    n = max(1, int(avg_scans))
    n = min(n, len(scan_infos))

    
    tics = np.array([s.tic for s in scan_infos], dtype=float)
    best_i = int(np.argmax(tics))

    start_i = max(0, best_i - n // 2)
    end_i = start_i + n
    if end_i > len(scan_infos):
        end_i = len(scan_infos)
        start_i = max(0, end_i - n)

    return scan_infos[start_i:end_i]


def average_ms1_scans(
    mzml_path: Path,
    selected_scans: List[ScanInfo],
    *,
    bin_decimals: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:

    if not selected_scans:
        return np.array([], dtype=float), np.array([], dtype=float)

    selected_set = {s.scan_number for s in selected_scans}

    bins: Dict[float, float] = {}
    used = 0

    for rec in iter_spectra(mzml_path, decode_arrays=True, decode_intensity_if_tic_missing=False):
        if rec.ms_level != 1:
            continue
        if rec.scan_number is None:
            
            continue
        if int(rec.scan_number) not in selected_set:
            continue

        mz = rec.mz
        inten = rec.intensity
        if mz is None or inten is None:
            continue

        used += 1

        mz_round = np.round(mz.astype(float), int(bin_decimals))

        
        for m, i in zip(mz_round.tolist(), inten.astype(float).tolist()):
            bins[m] = bins.get(m, 0.0) + float(i)

    if used == 0:
        raise RuntimeError('No data matched the requested scan. Check that the mzML file contains scan identifiers.')

    mz_sorted = np.array(sorted(bins.keys()), dtype=float)
    inten_avg = np.array([bins[m] / float(used) for m in mz_sorted], dtype=float)
    return mz_sorted, inten_avg


def extract_averaged_spectrum_from_mzml(
    mzml_path: Path,
    *,
    avg_scans: int = 8,
    bin_decimals: int = 4,
    raw_path_hint: Optional[str] = None,
) -> AveragedSpectrum:

    scan_infos, best_filter = collect_ms1_scan_infos(mzml_path)
    selected = choose_scan_window(scan_infos, avg_scans=avg_scans)

    mz, inten = average_ms1_scans(mzml_path, selected, bin_decimals=bin_decimals)

    
    rt_start = selected[0].rt_min if selected else None
    rt_end = selected[-1].rt_min if selected else None

    
    filter_string = best_filter
    if not filter_string:
        
        for s in selected:
            if s.filter_string:
                filter_string = s.filter_string
                break

    mz_range = parse_mz_range_from_filter(filter_string or "")

    acquisition_dt = read_run_start_time(mzml_path)

    return AveragedSpectrum(
        mz=mz,
        intensity=inten,
        scan_start=selected[0].scan_number,
        scan_end=selected[-1].scan_number,
        rt_start_min=rt_start,
        rt_end_min=rt_end,
        filter_string=filter_string,
        full_mz_min=mz_range[0] if mz_range else None,
        full_mz_max=mz_range[1] if mz_range else None,
        acquisition_datetime=acquisition_dt,
        raw_path=raw_path_hint,
        avg_scans=len(selected),
    )
