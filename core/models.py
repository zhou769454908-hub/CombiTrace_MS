
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ScanInfo:

    scan_number: int
    rt_min: Optional[float]
    tic: float
    filter_string: Optional[str] = None


@dataclass
class AveragedSpectrum:

    mz: np.ndarray
    intensity: np.ndarray

    scan_start: int
    scan_end: int
    rt_start_min: Optional[float]
    rt_end_min: Optional[float]

    filter_string: Optional[str] = None
    full_mz_min: Optional[float] = None
    full_mz_max: Optional[float] = None
    acquisition_datetime: Optional[datetime] = None
    raw_path: Optional[str] = None

    avg_scans: int = 1

    @property
    def nl(self) -> float:
    
        if self.intensity is None or len(self.intensity) == 0:
            return 0.0
        return float(np.max(self.intensity))

    @property
    def relative_intensity(self) -> np.ndarray:
        nl = self.nl
        if nl <= 0:
            return np.zeros_like(self.intensity)
        return (self.intensity / nl) * 100.0


@dataclass(frozen=True)
class Peak:
    mz: float
    intensity: float
    rel_intensity: float


@dataclass
class ReportMeta:

    instrument: str
    card_serial_number: str
    sample_serial_number: str
    operator: str
    report_date: str  
    operation_mode: str

    
    org_lines: list[str] = field(
        default_factory=lambda: [
            "National Center for Organic Mass Spectrometry in Shanghai",
            "Shanghai Institute of Organic Chemistry",
            "Chinese Academic of Sciences",
            "High Resolution AP-MALDI-MS REPORT",
        ]
    )


@dataclass
class ReportPaths:

    output_dir: Path
    word_path: Path
    spectrum_pdf_path: Path
    package_json_path: Path

    reviewed_word_path: Optional[Path] = None
    reviewed_spectrum_pdf_path: Optional[Path] = None


@dataclass
class ReviewInfo:

    status: str = "pending"  # pending/approved/rejected
    reviewer: str = ""
    reviewed_at: str = ""  # ISO string
    comment: str = ""
