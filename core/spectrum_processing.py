
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import math
import numpy as np

from .models import Peak


@dataclass(frozen=True)
class SpectrumSettings:

    avg_scans: int = 8
    bin_decimals: int = 4

    
    auto_zoom: bool = False
    zoom_min_rel: float = 1.0
    zoom_padding_da: float = 50.0

    
    label_top_n: int = 10
    label_min_rel: float = 0.0
    label_min_spacing_da: float = 0.2


def auto_zoom_range(
    mz: np.ndarray,
    rel: np.ndarray,
    *,
    min_rel: float = 1.0,
    padding_da: float = 50.0,
) -> Optional[Tuple[float, float]]:
    if mz.size == 0:
        return None
    if rel.size == 0:
        return None

    mask = rel >= float(min_rel)
    if not np.any(mask):
        return None

    mz_sel = mz[mask]
    xmin = float(np.min(mz_sel)) - float(padding_da)
    xmax = float(np.max(mz_sel)) + float(padding_da)

    
    if xmin < 0:
        xmin = 0.0
    if xmax <= xmin:
        return None

    return xmin, xmax


def pick_peaks_to_label(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    top_n: int = 15,
    min_rel: float = 3.0,
    min_spacing_da: float = 0.2,
) -> List[Peak]:

    if mz.size == 0:
        return []
    if intensity.size == 0:
        return []

    nl = float(np.max(intensity))
    if nl <= 0:
        return []

    rel = (intensity / nl) * 100.0

    idx = np.argsort(intensity)[::-1]

    peaks: List[Peak] = []
    used_mz: List[float] = []

    for i in idx:
        r = float(rel[i])
        if r < float(min_rel):
            break
        m = float(mz[i])
        # spacing
        if any(abs(m - um) < float(min_spacing_da) for um in used_mz):
            continue
        peaks.append(Peak(mz=m, intensity=float(intensity[i]), rel_intensity=r))
        used_mz.append(m)
        if len(peaks) >= int(top_n):
            break

    
    peaks.sort(key=lambda p: p.mz)
    return peaks


def pick_peaks_by_mz_bins(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    bin_width_da: float = 50.0,
    min_rel: float = 0.0,
) -> List[Peak]:
    if mz is None or intensity is None:
        return []
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if mz.size == 0 or intensity.size == 0:
        return []

    nl = float(np.max(intensity))
    if nl <= 0:
        return []

    rel = (intensity / nl) * 100.0

    bw = float(bin_width_da)
    if bw <= 0:
        bw = 50.0

    mn = float(np.min(mz))
    mx = float(np.max(mz))
    start = math.floor(mn / bw) * bw
    stop = math.ceil(mx / bw) * bw

    peaks: List[Peak] = []
    b = start
    
    while b < stop + 1e-9:
        b0 = float(b)
        b1 = float(b + bw)
        i0 = int(np.searchsorted(mz, b0, side="left"))
        i1 = int(np.searchsorted(mz, b1, side="right"))
        if i1 > i0:
            sub = intensity[i0:i1]
            if sub.size:
                j = int(np.argmax(sub))
                idx = i0 + j
                r = float(rel[idx])
                if r >= float(min_rel):
                    peaks.append(Peak(mz=float(mz[idx]), intensity=float(intensity[idx]), rel_intensity=r))
        b += bw

    peaks.sort(key=lambda p: p.mz)
    return peaks