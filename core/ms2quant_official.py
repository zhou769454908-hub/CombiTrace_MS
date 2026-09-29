"""Bridge to the official public MS2Quant R package.

The KruveLab MS2Quant package bundles a pretrained xgbTree ionisation-
efficiency model.  This module does *not* redistribute that package or its
model object.  Instead it invokes a user-installed official R package through
Rscript, predicts logIE from Product SMILES under the programmed eluent
conditions, and then performs a Q Exactive HF / method-specific transfer:

    log10(normalised RF) = slope * predicted_logIE + intercept

The transfer is refitted inside every local CV training fold.  The unknown
sample is never used as supervised calibration data.

Important scope limits of the current official public model:
* positive ESI only;
* [M+H]+ and [M]+ ions;
* full MS2Quant_quantify() expects multi-level calibrant curves and raw areas.
  The present bridge uses the official pretrained IE predictor and the user's
  one-point-per-compound area/internal-standard ratio to fit a normalised-RF
  transfer.  This is aligned with the published instrument-transfer principle,
  but it is not a claim that the full official calibration-curve workflow was
  reproduced.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.model_selection import RepeatedKFold
    SKLEARN_AVAILABLE = True
    SKLEARN_ERROR = ""
except Exception as exc:  # pragma: no cover
    SKLEARN_AVAILABLE = False
    SKLEARN_ERROR = str(exc)


CONCENTRATION_UNIT_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ('Select the calibration concentration unit', ""),
    ('mol/L', "mol_L"),
    ('mmol/L', "mmol_L"),
    ("\u00B5mol/mL (micromole/milliliter)", "umol_mL"),
    ('umol/L', "umol_L"),
    ('nmol/L', "nmol_L"),
    ('mg/L', "mg_L"),
    ('ug/L', "ug_L"),
    ('ng/mL', "ng_mL"),
    ('ng/L', "ng_L"),
)
CONCENTRATION_UNIT_LABELS = [x[0] for x in CONCENTRATION_UNIT_OPTIONS]
CONCENTRATION_UNIT_MAP = {x[0]: x[1] for x in CONCENTRATION_UNIT_OPTIONS}
CONCENTRATION_UNIT_CODE_TO_LABEL = {x[1]: x[0] for x in CONCENTRATION_UNIT_OPTIONS}


@dataclass
class MS2QuantEnvironment:
    available: bool
    rscript_path: str = ""
    r_version: str = ""
    package_version: str = ""
    package_path: str = ""
    java_home: str = ""
    function_available: bool = False
    message: str = ""
    stdout: str = ""
    stderr: str = ""

    def rows(self) -> List[Dict[str, object]]:
        return [
            {"Item": "Official_MS2Quant_available", "Value": bool(self.available)},
            {"Item": "Rscript_path", "Value": self.rscript_path},
            {"Item": "R_version", "Value": self.r_version},
            {"Item": "MS2Quant_version", "Value": self.package_version},
            {"Item": "MS2Quant_package_path", "Value": self.package_path},
            {"Item": "JAVA_HOME", "Value": self.java_home},
            {"Item": "MS2Quant_predict_IE_available", "Value": bool(self.function_available)},
            {"Item": "Environment_message", "Value": self.message},
            {"Item": "R_stdout", "Value": self.stdout},
            {"Item": "R_stderr", "Value": self.stderr},
        ]


@dataclass
class MS2QuantPredictionResult:
    mapping: Dict[str, float] = field(default_factory=dict)
    audit_rows: List[Dict[str, object]] = field(default_factory=list)
    environment: Optional[MS2QuantEnvironment] = None
    warnings: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


@dataclass
class MS2QuantTransferResult:
    cv_rows: List[Dict[str, object]] = field(default_factory=list)
    target_rows: List[Dict[str, object]] = field(default_factory=list)
    transfer_rows: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _num(value: object) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def detect_rscript(explicit_path: str = "") -> str:
    """Return an existing Rscript executable path, or an empty string."""
    candidates: List[str] = []
    explicit = str(explicit_path or "").strip().strip('"')
    if explicit:
        candidates.append(explicit)
    found = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if found:
        candidates.append(found)

    if os.name == "nt":
        roots = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        patterns: List[str] = []
        for root in roots:
            if not root:
                continue
            patterns.extend([
                str(Path(root) / "R" / "R-*" / "bin" / "Rscript.exe"),
                str(Path(root) / "R" / "R-*" / "bin" / "x64" / "Rscript.exe"),
            ])
        for pattern in patterns:
            candidates.extend(sorted(glob.glob(pattern), reverse=True))
    else:
        candidates.extend(["/usr/bin/Rscript", "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript"])

    seen = set()
    for candidate in candidates:
        path = str(candidate or "").strip()
        if not path or path.lower() in seen:
            continue
        seen.add(path.lower())
        if Path(path).is_file():
            return str(Path(path).resolve())
    return ""


def _run_command(command: Sequence[str], *, timeout_sec: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(5, int(timeout_sec)),
        check=False,
    )


def check_ms2quant_environment(rscript_path: str = "", timeout_sec: int = 120) -> MS2QuantEnvironment:
    rscript = detect_rscript(rscript_path)
    if not rscript:
        return MS2QuantEnvironment(
            available=False,
            message="Rscript.exe was not found. Install 64-bit R and select Rscript.exe in the dual-model settings.",
        )

    expr = r'''
info <- list(
  r_version = paste(R.version$major, R.version$minor, sep="."),
  package_available = requireNamespace("MS2Quant", quietly=TRUE),
  package_version = "",
  package_path = "",
  java_home = Sys.getenv("JAVA_HOME"),
  function_available = FALSE
)
if (info$package_available) {
  info$package_version <- as.character(utils::packageVersion("MS2Quant"))
  info$package_path <- find.package("MS2Quant")
  ns <- asNamespace("MS2Quant")
  info$function_available <- exists("MS2Quant_predict_IE", envir=ns, inherits=FALSE)
}
cat(paste(
  paste0("r_version=", info$r_version),
  paste0("package_available=", info$package_available),
  paste0("package_version=", info$package_version),
  paste0("package_path=", info$package_path),
  paste0("java_home=", info$java_home),
  paste0("function_available=", info$function_available),
  sep="\n"
))
'''
    try:
        proc = _run_command([rscript, "--vanilla", "-e", expr], timeout_sec=timeout_sec)
    except Exception as exc:
        return MS2QuantEnvironment(
            available=False,
            rscript_path=rscript,
            message=f"Failed to execute Rscript: {exc}",
        )
    values: Dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    package_ok = values.get("package_available", "").upper() == "TRUE"
    function_ok = values.get("function_available", "").upper() == "TRUE"
    available = proc.returncode == 0 and package_ok and function_ok
    message = (
        "Official MS2Quant R package is available."
        if available else
        "R is available, but the official MS2Quant package/function is not ready. Run r_scripts/install_ms2quant.R and inspect the R output."
    )
    return MS2QuantEnvironment(
        available=available,
        rscript_path=rscript,
        r_version=values.get("r_version", ""),
        package_version=values.get("package_version", ""),
        package_path=values.get("package_path", ""),
        java_home=values.get("java_home", ""),
        function_available=function_ok,
        message=message,
        stdout=(proc.stdout or "")[-4000:],
        stderr=(proc.stderr or "")[-4000:],
    )


def _molecular_weight(record: Dict[str, object]) -> Optional[float]:
    for key in ("MolWt", "Exact_mass", "Product_Exact_Mass", "Product_exact_mass"):
        value = _num(record.get(key))
        if value is not None and value > 0:
            return float(value)
    return None


def concentration_to_molar(value: float, unit: str, molecular_weight: Optional[float]) -> float:
    x = float(value)
    code = str(unit or "mol_L")
    if code == "mol_L":
        return x
    if code == "mmol_L":
        return x * 1e-3
    if code == "umol_mL":
        return x * 1e-3
    if code == "umol_L":
        return x * 1e-6
    if code == "nmol_L":
        return x * 1e-9
    mw = float(molecular_weight or 0.0)
    if mw <= 0:
        raise ValueError("Molecular weight is required for a mass-concentration unit")
    if code == "mg_L":
        return x * 1e-3 / mw
    if code in {"ug_L", "ng_mL"}:
        return x * 1e-6 / mw
    if code == "ng_L":
        return x * 1e-9 / mw
    raise ValueError(f"Unsupported concentration unit: {unit}")


def concentration_from_molar(value_molar: float, unit: str, molecular_weight: Optional[float]) -> float:
    x = float(value_molar)
    code = str(unit or "mol_L")
    if code == "mol_L":
        return x
    if code == "mmol_L":
        return x / 1e-3
    if code == "umol_mL":
        return x / 1e-3
    if code == "umol_L":
        return x / 1e-6
    if code == "nmol_L":
        return x / 1e-9
    mw = float(molecular_weight or 0.0)
    if mw <= 0:
        raise ValueError("Molecular weight is required for a mass-concentration unit")
    if code == "mg_L":
        return x * mw / 1e-3
    if code in {"ug_L", "ng_mL"}:
        return x * mw / 1e-6
    if code == "ng_L":
        return x * mw / 1e-9
    raise ValueError(f"Unsupported concentration unit: {unit}")


def _private_smiles(record: Dict[str, object]) -> str:
    for key in ("__Product_SMILES_Private", "Product_SMILES", "Selected_Product_SMILES"):
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _record_key(record: Dict[str, object], index: int) -> str:
    key = _text(record.get("ABC_Formula_Key") or record.get("Product_Master_Formula_Key"))
    if key:
        return key
    source_bits = [
        _text(record.get("Source_file")),
        _text(record.get("Source_sheet")),
        _text(record.get("Source_row")),
        _text(record.get("Combo")),
    ]
    source = "|".join(bit for bit in source_bits if bit)
    if source:
        return "source|" + source
    return "row_%d" % int(index)


def _prediction_key(
    record: Dict[str, object],
    index: int,
    *,
    organic_rounding_pct: float = 0.5,
) -> str:
    """Stable key for one structure under one apex-eluent condition.

    MS2Quant predictions depend on both structure and organic percentage.  The
    calibration and unknown-sample occurrences of the same compound therefore
    share a prediction only when their rounded apex %%B is also the same.
    """
    base = _record_key(record, index)
    smiles = _private_smiles(record)
    digest = hashlib.sha1(smiles.encode("utf-8", errors="ignore")).hexdigest()[:12] if smiles else "nosmiles"
    b_pct = _num(record.get("Mobile_phase_B_pct"))
    if b_pct is None:
        b_tag = "missing"
    else:
        step = max(0.01, float(organic_rounding_pct))
        b_tag = f"{round(float(b_pct) / step) * step:.4f}"
    return f"{base}|B={b_tag}|S={digest}"


def _official_modifier_code(organic_modifier: str) -> str:
    text = str(organic_modifier or "").strip().lower()
    if text in {"acetonitrile", "mecn", "acn"}:
        return "MeCN"
    if text in {"methanol", "meoh"}:
        return "MeOH"
    return ""


def run_official_ms2quant_predictions(
    records: Sequence[Dict[str, object]],
    *,
    rscript_path: str,
    organic_modifier: str,
    aqueous_pH: float,
    output_xlsx: Path,
    timeout_sec: int = 3600,
    organic_rounding_pct: float = 0.5,
) -> MS2QuantPredictionResult:
    """Predict official pretrained logIE for unique input records."""
    environment = check_ms2quant_environment(rscript_path, timeout_sec=min(timeout_sec, 180))
    result = MS2QuantPredictionResult(environment=environment)
    if not environment.available:
        result.warnings.append(environment.message)
        return result
    modifier_code = _official_modifier_code(organic_modifier)
    if not modifier_code:
        result.warnings.append("Official MS2Quant currently accepts MeCN or MeOH as organic_modifier.")
        return result

    rows: List[Dict[str, object]] = []
    seen: Dict[str, int] = {}
    for idx, record in enumerate(records):
        abc_key = _record_key(record, idx)
        key = _prediction_key(record, idx, organic_rounding_pct=organic_rounding_pct)
        smiles = _private_smiles(record)
        b_pct = _num(record.get("Mobile_phase_B_pct"))
        status = "READY"
        reason = ""
        if not smiles:
            status, reason = "SKIPPED", "Product SMILES unavailable"
        elif b_pct is None:
            status, reason = "SKIPPED", "Mobile_phase_B_pct unavailable"
        row = {
            "row_id": key,
            "ABC_Formula_Key": abc_key,
            "SMILES": smiles,
            "organic_percentage": (round(float(b_pct) / organic_rounding_pct) * organic_rounding_pct if b_pct is not None else ""),
            "Source_file": record.get("Source_file", ""),
            "Source_sheet": record.get("Source_sheet", ""),
            "Source_row": record.get("Source_row", ""),
            "Formula": record.get("Formula", ""),
            "Combo": record.get("Combo", ""),
            "Status": status,
            "Reason": reason,
        }
        result.audit_rows.append(dict(row))
        if status != "READY":
            continue
        # One structure/condition key should be predicted once.  Duplicate
        # records are mapped back to the same value.
        dedupe_key = key
        if dedupe_key in seen:
            continue
        seen[dedupe_key] = len(rows)
        rows.append({
            "row_id": key,
            "ABC_Formula_Key": abc_key,
            "SMILES": smiles,
            "organic_percentage": row["organic_percentage"],
        })

    if not rows:
        result.warnings.append("No rows had both Product SMILES and apex organic percentage for official MS2Quant prediction.")
        return result

    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "r_scripts" / "ms2quant_bridge.R"
    if not script_path.exists():
        result.warnings.append(f"Official MS2Quant bridge script is missing: {script_path}")
        return result

    temp_root = Path(tempfile.mkdtemp(prefix="ct_ms2q_"))
    input_csv = temp_root / "input.csv"
    output_csv = temp_root / "output.csv"
    meta_json = temp_root / "metadata.json"
    with input_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "ABC_Formula_Key", "SMILES", "organic_percentage"])
        writer.writeheader()
        writer.writerows(rows)

    command = [
        environment.rscript_path,
        "--vanilla",
        str(script_path),
        str(input_csv),
        str(output_csv),
        str(meta_json),
        modifier_code,
        repr(float(aqueous_pH)),
    ]
    try:
        proc = _run_command(command, timeout_sec=timeout_sec)
    except Exception as exc:
        result.warnings.append(f"Official MS2Quant R bridge failed to execute: {exc}")
        return result
    result.stdout = (proc.stdout or "")[-12000:]
    result.stderr = (proc.stderr or "")[-12000:]
    if proc.returncode != 0:
        result.warnings.append(
            "Official MS2Quant R bridge returned a non-zero status. See MS2Quant_Environment and MS2Quant_IE_Audit. "
            + (result.stderr[-1500:] if result.stderr else "")
        )
        return result
    if not output_csv.exists():
        result.warnings.append("Official MS2Quant R bridge completed without creating its output CSV.")
        return result

    output_rows: List[Dict[str, str]] = []
    with output_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        output_rows = list(csv.DictReader(handle))
    for row in output_rows:
        key = _text(row.get("row_id"))
        value = _num(row.get("predicted_logIE"))
        if key and value is not None:
            result.mapping[key] = float(value)

    by_key = {str(r.get("row_id")): r for r in output_rows}
    for audit in result.audit_rows:
        key = _text(audit.get("row_id"))
        if audit.get("Status") != "READY":
            continue
        out = by_key.get(key)
        if out is None:
            audit["Status"] = "FAILED"
            audit["Reason"] = "No prediction returned by MS2Quant"
            continue
        audit["Status"] = _text(out.get("status")) or "PREDICTED"
        audit["Reason"] = _text(out.get("reason"))
        audit["predicted_logIE"] = out.get("predicted_logIE", "")
        audit["prediction_column"] = out.get("prediction_column", "")
        audit["organic_percentage_used"] = out.get("organic_percentage", audit.get("organic_percentage", ""))

    metadata: Dict[str, object] = {}
    if meta_json.exists():
        try:
            metadata = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    if metadata:
        result.audit_rows.insert(0, {
            "row_id": "SOFTWARE_METADATA",
            "Status": "INFO",
            "Reason": json.dumps(metadata, ensure_ascii=False),
        })
    missing = len(rows) - len(result.mapping)
    if missing:
        result.warnings.append(f"Official MS2Quant returned predictions for {len(result.mapping)}/{len(rows)} unique rows; inspect MS2Quant_IE_Audit.")
    return result


def _rank_targets(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    values = np.asarray([float(r["Estimated_concentration"]) for r in rows], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(rows), dtype=int)
    for rank, index in enumerate(order, start=1):
        ranks[int(index)] = rank
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for idx, row in enumerate(rows):
        row["Corrected_rank_low_to_high"] = int(ranks[idx])
        row["Corrected_percentile"] = float(ranks[idx] / len(rows) * 100.0)
        row["Concentration_class"] = "Low" if values[idx] <= q1 else "Medium" if values[idx] <= q2 else "High"


def _aggregate_cv_predictions(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["Calibration_Index"]), []).append(row)
    out: List[Dict[str, object]] = []
    for index in sorted(grouped):
        group = grouped[index]
        predicted = float(np.median([float(r["Predicted_concentration"]) for r in group]))
        actual = float(group[0]["Actual_concentration"])
        fold = max(predicted / actual, actual / predicted)
        row = dict(group[0])
        row["Predicted_concentration"] = predicted
        row["Fold_error"] = fold
        row["CV_prediction_repeats"] = len(group)
        row["Response_corrected_score_log10"] = math.log10(predicted)
        out.append(row)
    return out


def fit_ms2quant_instrument_transfer(
    calibration: Sequence[Dict[str, object]],
    targets: Sequence[Dict[str, object]],
    logie_map: Dict[str, float],
    *,
    concentration_unit: str,
    instrument_name: str,
    cv_splits: int = 5,
    cv_repeats: int = 3,
    random_state: int = 42,
) -> MS2QuantTransferResult:
    """Fit Q Exactive HF-specific normalised RF transfer from official logIE."""
    result = MS2QuantTransferResult()
    if not SKLEARN_AVAILABLE:
        result.warnings.append("scikit-learn is required for repeated-CV instrument transfer: " + SKLEARN_ERROR)
        return result

    usable: List[Dict[str, object]] = []
    for idx, record in enumerate(calibration):
        abc_key = _record_key(record, idx)
        prediction_key = _prediction_key(record, idx)
        logie = logie_map.get(prediction_key)
        ratio = _num(record.get("Measured_ratio"))
        concentration = _num(record.get("Actual_concentration"))
        mw = _molecular_weight(record)
        reason = ""
        conc_m = None
        if logie is None:
            reason = "official predicted logIE missing"
        elif ratio is None or ratio <= 0:
            reason = "measured area/internal-standard ratio missing or <=0"
        elif concentration is None or concentration <= 0:
            reason = "actual standard concentration missing or <=0"
        else:
            try:
                conc_m = concentration_to_molar(float(concentration), concentration_unit, mw)
            except Exception as exc:
                reason = str(exc)
        if reason or conc_m is None or conc_m <= 0:
            result.transfer_rows.append({
                "Stage": "Calibration_audit", "Calibration_Index": idx, "ABC_Formula_Key": abc_key,
                "MS2Quant_Prediction_Key": prediction_key,
                "Status": "SKIPPED", "Reason": reason or "molar concentration invalid",
                "Concentration_unit": concentration_unit, "Molecular_weight": mw if mw is not None else "",
            })
            continue
        item = dict(record)
        item["__official_key__"] = abc_key
        item["__prediction_key__"] = prediction_key
        item["__predicted_logIE__"] = float(logie)
        item["__concentration_molar__"] = float(conc_m)
        item["__molecular_weight__"] = mw
        usable.append(item)

    if len(usable) < max(10, int(cv_splits)):
        result.warnings.append(f"Only {len(usable)} calibration rows are usable for official MS2Quant transfer; at least 10 are recommended.")
        return result

    x_all = np.asarray([float(r["__predicted_logIE__"]) for r in usable], dtype=float)
    y_all = np.asarray([
        math.log10(float(r["Measured_ratio"]) / float(r["__concentration_molar__"]))
        for r in usable
    ], dtype=float)
    if len(np.unique(x_all)) < 2:
        result.warnings.append("Official predicted logIE has no variation; instrument transfer cannot be fitted.")
        return result

    n_splits = max(2, min(int(cv_splits), len(usable)))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=max(1, int(cv_repeats)), random_state=int(random_state))
    cv_raw: List[Dict[str, object]] = []
    for fold_no, (train_idx, test_idx) in enumerate(rkf.split(x_all), start=1):
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        if len(np.unique(x_train)) < 2:
            continue
        slope, intercept = np.polyfit(x_train, y_train, 1)
        for test_index in test_idx:
            record = usable[int(test_index)]
            pred_log_rrf = float(slope * float(record["__predicted_logIE__"]) + intercept)
            pred_molar = float(record["Measured_ratio"]) / (10.0 ** pred_log_rrf)
            pred_original = concentration_from_molar(pred_molar, concentration_unit, record.get("__molecular_weight__"))
            cv_raw.append({
                "Calibration_Index": int(test_index),
                "Fold": fold_no,
                "Source_file": record.get("Source_file", ""),
                "Source_sheet": record.get("Source_sheet", ""),
                "Source_row": record.get("Source_row", ""),
                "Combo": record.get("Combo", ""),
                "ABC_Formula_Key": record.get("__official_key__", ""),
                "MS2Quant_Prediction_Key": record.get("__prediction_key__", ""),
                "Formula": record.get("Formula", ""),
                "Actual_concentration": float(record["Actual_concentration"]),
                "Concentration_unit": concentration_unit,
                "Actual_concentration_M": float(record["__concentration_molar__"]),
                "Measured_ratio": float(record["Measured_ratio"]),
                "Official_predicted_logIE": float(record["__predicted_logIE__"]),
                "Transfer_slope": float(slope),
                "Transfer_intercept": float(intercept),
                "Predicted_log10_normalized_RF": pred_log_rrf,
                "Predicted_concentration_M": pred_molar,
                "Predicted_concentration": pred_original,
                "Instrument": instrument_name,
                "Response_basis": "area/internal-standard ratio",
            })
    result.cv_rows = _aggregate_cv_predictions(cv_raw)

    slope, intercept = np.polyfit(x_all, y_all, 1)
    fitted = slope * x_all + intercept
    ss_res = float(np.sum((y_all - fitted) ** 2))
    ss_tot = float(np.sum((y_all - float(np.mean(y_all))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    result.transfer_rows.extend([
        {"Stage": "Final_transfer", "Parameter": "Instrument", "Value": instrument_name},
        {"Stage": "Final_transfer", "Parameter": "Response_basis", "Value": "area/internal-standard ratio"},
        {"Stage": "Final_transfer", "Parameter": "Concentration_unit_input_output", "Value": concentration_unit},
        {"Stage": "Final_transfer", "Parameter": "Calibration_rows", "Value": len(usable)},
        {"Stage": "Final_transfer", "Parameter": "Slope_logRF_vs_logIE", "Value": float(slope)},
        {"Stage": "Final_transfer", "Parameter": "Intercept_logRF_vs_logIE", "Value": float(intercept)},
        {"Stage": "Final_transfer", "Parameter": "R2_calibration_logRF_vs_logIE", "Value": r2},
        {"Stage": "Final_transfer", "Parameter": "Equation", "Value": "log10(normalized RF)=slope*official predicted logIE+intercept"},
        {"Stage": "Final_transfer", "Parameter": "Instrument_transfer_scope", "Value": "Thermo Q Exactive HF method-specific transfer from official universal pretrained IE"},
        {"Stage": "Final_transfer", "Parameter": "Official_full_quantify_equivalence", "Value": "No: one-point-per-compound normalized RF transfer; not multi-level MS2Quant_quantify calibration curves"},
    ])

    target_rows: List[Dict[str, object]] = []
    for idx, record in enumerate(targets):
        key = _record_key(record, idx)
        prediction_key = _prediction_key(record, idx)
        logie = logie_map.get(prediction_key)
        ratio = _num(record.get("Measured_ratio"))
        mw = _molecular_weight(record)
        if logie is None or ratio is None or ratio <= 0:
            continue
        pred_log_rrf = float(slope * float(logie) + intercept)
        pred_molar = float(ratio) / (10.0 ** pred_log_rrf)
        try:
            pred_original = concentration_from_molar(pred_molar, concentration_unit, mw)
        except Exception:
            continue
        target_rows.append({
            "Method": "Official_MS2Quant_pretrained_xgbTree_QEHF_transfer",
            "Source_file": record.get("Source_file", ""),
            "Source_sheet": record.get("Source_sheet", ""),
            "Source_row": record.get("Source_row", ""),
            "Name": record.get("Name", ""),
            "Formula": record.get("Formula", ""),
            "Combo": record.get("Combo", ""),
            "ABC_Formula_Key": key,
            "MS2Quant_Prediction_Key": prediction_key,
            "Raw_area_IS_ratio": float(ratio),
            "Official_predicted_logIE": float(logie),
            "Transfer_slope": float(slope),
            "Transfer_intercept": float(intercept),
            "Predicted_log10_RRF": pred_log_rrf,
            "Predicted_normalized_RF": 10.0 ** pred_log_rrf,
            "Predicted_concentration_M": pred_molar,
            "Estimated_concentration": pred_original,
            "Concentration_unit": concentration_unit,
            "Response_corrected_score_log10": math.log10(pred_original),
            "Instrument": instrument_name,
            "Response_basis": "area/internal-standard ratio",
            "Apex_RT_min": record.get("Apex_RT_min", ""),
            "Mobile_phase_B_pct": record.get("Mobile_phase_B_pct", ""),
        })
    _rank_targets(target_rows)
    result.target_rows = target_rows
    return result
