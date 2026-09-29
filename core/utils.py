
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_mmddyy_hhmmss(dt: datetime) -> str:
    
    return dt.strftime("%m/%d/%y %H:%M:%S")


def format_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y/%m/%d")


def sci_no_plus(x: float, sig: int = 3) -> str:

    if x == 0:
        return "0"
    s = f"{x:.2E}"  
    s = s.replace("E+0", "E").replace("E+", "E").replace("E-0", "E-")
    
    s = re.sub(r"E(-?)0+(\d+)", r"E\1\2", s)
    return s


def parse_scan_number(spec_id: str) -> Optional[int]:

    if not spec_id:
        return None
    m = re.search(r"scan=(\d+)", spec_id)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    
    m = re.search(r"(\d+)$", spec_id)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def parse_mz_range_from_filter(filter_string: str) -> Optional[Tuple[float, float]]:

    if not filter_string:
        return None
    m = re.search(r"\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]", filter_string)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def run_subprocess(cmd: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        shell=False,
    )


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def open_in_os(path: Path) -> None:

    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        
        pass


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


def safe_stem_from_raw_path(raw_path: Path) -> str:

    
    stem = raw_path.stem
    if stem:
        return stem
    
    return raw_path.name
