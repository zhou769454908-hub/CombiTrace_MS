
from __future__ import annotations

import base64
import gzip
import io
import re
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _open_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _parse_iso_datetime(s: str) -> Optional[datetime]:
    if not s:
        return None
    
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def read_run_start_time(mzml_path: Path) -> Optional[datetime]:

    with _open_maybe_gzip(mzml_path) as fh:
        
        context = ET.iterparse(fh, events=("start",))
        for _, elem in context:
            if _local_name(elem.tag) == "run":
                ts = elem.attrib.get("startTimeStamp")
                return _parse_iso_datetime(ts) if ts else None
    return None


@dataclass
class SpectrumRecord:
    spectrum_id: str
    scan_number: Optional[int]
    ms_level: Optional[int]
    rt_min: Optional[float]
    filter_string: Optional[str]
    tic: Optional[float]
    mz: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None


def _extract_scan_number(spec_id: str) -> Optional[int]:
    if not spec_id:
        return None
    m = re.search(r"scan=(\d+)", spec_id)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _rt_to_minutes(value: float, unit_name: Optional[str]) -> float:
    if not unit_name:
        return float(value)
    u = unit_name.lower()
    if "second" in u:
        return float(value) / 60.0
    
    return float(value)


def _decode_binary_array(bda_elem: ET.Element) -> Tuple[Optional[str], Optional[np.ndarray]]:
    "- kind: 'mz' | 'intensity' | None\n    - array: numpy array or None"

    kind: Optional[str] = None
    compressed = False
    dtype: Optional[np.dtype] = None
    little_endian = True
    numpress = False

    binary_text: Optional[str] = None

    for child in list(bda_elem):
        tag = _local_name(child.tag)
        if tag == "cvParam":
            name = child.attrib.get("name", "")
            if name == "m/z array":
                kind = "mz"
            elif name == "intensity array":
                kind = "intensity"
            elif name == "zlib compression":
                compressed = True
            elif name == "MS-Numpress linear prediction compression" or "numpress" in name.lower():
                numpress = True
            elif name == "64-bit float":
                dtype = np.dtype("<f8")  
            elif name == "32-bit float":
                dtype = np.dtype("<f4")
            elif name == "little endian":
                little_endian = True
            elif name == "big endian":
                little_endian = False
        elif tag == "binary":
            binary_text = child.text

    if numpress:
        raise RuntimeError('MS-Numpress compression is not supported by this reader. Disable Numpress during conversion.')

    if kind is None:
        return None, None
    if dtype is None:
        
        dtype = np.dtype("<f8")

    if not binary_text:
        return kind, np.array([], dtype=dtype)

    raw = base64.b64decode(binary_text.encode("utf-8"))
    if compressed:
        raw = zlib.decompress(raw)

    
    arr = np.frombuffer(raw, dtype=dtype)
    if not little_endian:
        arr = arr.byteswap().newbyteorder()

    return kind, arr


def _parse_spectrum_elem(
    spec_elem: ET.Element,
    *,
    decode_arrays: bool,
    decode_intensity_if_tic_missing: bool,
) -> SpectrumRecord:
    spec_id = spec_elem.attrib.get("id", "")
    scan_number = _extract_scan_number(spec_id)

    ms_level: Optional[int] = None
    rt_min: Optional[float] = None
    filter_string: Optional[str] = None
    tic: Optional[float] = None

    
    for elem in spec_elem.iter():
        if _local_name(elem.tag) != "cvParam":
            continue
        name = elem.attrib.get("name", "")
        if name == "ms level":
            try:
                ms_level = int(elem.attrib.get("value", ""))
            except Exception:
                ms_level = None
        elif name == "scan start time":
            try:
                v = float(elem.attrib.get("value", ""))
                rt_min = _rt_to_minutes(v, elem.attrib.get("unitName"))
            except Exception:
                rt_min = None
        elif name == "filter string":
            filter_string = elem.attrib.get("value")
        elif name == "total ion current":
            try:
                tic = float(elem.attrib.get("value", ""))
            except Exception:
                tic = None

    mz_arr: Optional[np.ndarray] = None
    int_arr: Optional[np.ndarray] = None

    if decode_arrays or (decode_intensity_if_tic_missing and tic is None):
        
        for elem in spec_elem.iter():
            if _local_name(elem.tag) != "binaryDataArray":
                continue
            kind, arr = _decode_binary_array(elem)
            if kind == "mz" and decode_arrays:
                mz_arr = arr
            elif kind == "intensity":
                int_arr = arr

        if tic is None and int_arr is not None:
            tic = float(np.sum(int_arr))

    return SpectrumRecord(
        spectrum_id=spec_id,
        scan_number=scan_number,
        ms_level=ms_level,
        rt_min=rt_min,
        filter_string=filter_string,
        tic=tic,
        mz=mz_arr,
        intensity=int_arr,
    )


def iter_spectra(
    mzml_path: Path,
    *,
    decode_arrays: bool = False,
    decode_intensity_if_tic_missing: bool = True,
) -> Iterator[SpectrumRecord]:

    with _open_maybe_gzip(mzml_path) as fh:
        
        context = ET.iterparse(fh, events=("end",))
        for _, elem in context:
            if _local_name(elem.tag) != "spectrum":
                continue

            rec = _parse_spectrum_elem(
                elem,
                decode_arrays=decode_arrays,
                decode_intensity_if_tic_missing=decode_intensity_if_tic_missing,
            )
            yield rec

            
            elem.clear()


