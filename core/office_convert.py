'soffice --headless --convert-to doc --outdir <outdir> <input.docx>'

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .utils import run_subprocess, which


def convert_docx_to_doc(docx_path: Path, out_dir: Path) -> Optional[Path]:

    soffice = which("soffice") or which("libreoffice")
    if not soffice:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [soffice, "--headless", "--convert-to", "doc", "--outdir", str(out_dir), str(docx_path)]
    proc = run_subprocess(cmd)
    if proc.returncode != 0:
        return None

    
    out_doc = out_dir / (docx_path.stem + ".doc")
    return out_doc if out_doc.exists() else None
