'- -i=<raw_path>\n- -o=<output_dir>\n- -f=2  (indexed mzML)\n- -m=0  (metadata JSON)'

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .utils import ensure_dir, run_subprocess, which


@dataclass
class ThermoRawFileParserConfig:
    exe_path: Path
    use_mono: bool = False
    mono_path: str = "mono"
    extra_args: Tuple[str, ...] = ()


class ThermoRawFileParserRunner:
    def __init__(self, config: ThermoRawFileParserConfig):
        self.config = config

    def check_available(self) -> None:
        if not self.config.exe_path.exists():
            raise FileNotFoundError(f'ThermoRawFileParser not found: {self.config.exe_path}')
        if self.config.use_mono:
            if which(self.config.mono_path) is None:
                raise RuntimeError(
                    f"use_mono is enabled, but the executable was not found: '{self.config.mono_path}'. Install Mono or disable use_mono."
                )

    def convert_to_mzml(
        self,
        raw_path: Path,
        out_dir: Path,
        *,
        mzml_format: int = 2,
        metadata_format: int = 0,
        gzip: bool = False,
    ) -> Tuple[Path, Optional[Path], str]:
        '- mzml_path\n        - log_text'

        self.check_available()
        ensure_dir(out_dir)

        cmd: list[str] = []
        if self.config.use_mono:
            cmd.append(self.config.mono_path)
        cmd.append(str(self.config.exe_path))

        
        cmd.append(f"-i={raw_path}")
        cmd.append(f"-o={out_dir}")
        cmd.append(f"-f={mzml_format}")
        cmd.append(f"-m={metadata_format}")
        if gzip:
            cmd.append("-g")
        cmd.extend(list(self.config.extra_args))

        proc = run_subprocess(cmd)
        log_text = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")

        if proc.returncode != 0:
            raise RuntimeError(
                'ThermoRawFileParser failed (exit code {code}).\n\nCommand:\n{cmd}\n\nOutput:\n{out}'.format(
                    code=proc.returncode,
                    cmd=" ".join(cmd),
                    out=log_text.strip(),
                )
            )

        mzml_path = self._find_mzml(out_dir, raw_path)
        meta_json_path = self._find_metadata_json(out_dir, raw_path)

        return mzml_path, meta_json_path, log_text

    @staticmethod
    def _find_mzml(out_dir: Path, raw_path: Path) -> Path:
        patterns = ["*.mzML", "*.mzML.gz", "*.mzml", "*.mzml.gz"]
        files: list[Path] = []
        for pat in patterns:
            files.extend(out_dir.glob(pat))
        if not files:
            for pat in patterns:
                files.extend(out_dir.rglob(pat))

        if not files:
            raise FileNotFoundError(f'No mzML file found in the output directory: {out_dir}')

        base = raw_path.stem
        
        for f in sorted(files):
            if base and base in f.stem:
                return f
        return sorted(files)[0]

    @staticmethod
    def _find_metadata_json(out_dir: Path, raw_path: Path) -> Optional[Path]:
        json_files = list(out_dir.glob("*.json")) + list(out_dir.rglob("*.json"))
        if not json_files:
            return None

        base = raw_path.stem
        for f in sorted(json_files):
            if base and base in f.stem:
                return f
        return sorted(json_files)[0]
