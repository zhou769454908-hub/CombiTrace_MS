"""Dual validation against a published ionization-efficiency workflow.

This module provides three auditable reference paths next to CombiTrace-IE:

1. ``official_ms2quant_r``
   Invokes the public KruveLab/MS2Quant R package and its bundled pretrained
   xgbTree model (1191-chemical IE training set), then fits a local
   instrument/method transfer for the Thermo Scientific Q Exactive HF.

2. ``published_rf_surrogate``
   A local Random-Forest benchmark that follows the *published protocol* as
   closely as the locally available RDKit descriptors permit.  It combines
   molecular numeric descriptors with the five eluent descriptors used in the
   RandFor-IE literature (viscosity, surface tension, polarity index, aqueous
   pH and NH4 presence), applies paper-aligned descriptor filtering, and learns
   log10(RRF) only from the user's known standards.

   Important: this is a protocol surrogate, not the authors' exact pretrained
   universal RandFor-IE model.  The original model was trained on a much larger
   PaDEL/experimental-IE dataset and its fitted weights are not bundled here.

3. ``external_logie_transfer``
   Reproduces the published instrument-transfer step when externally predicted
   logIE values are supplied: log10(RF) = slope * predicted_logIE + intercept.
   The slope/intercept are fitted inside every CV training fold and finally on
   all standards before prediction of the unknown sample.

Both paths use only the separate known-standard injections for supervision.
The unknown real sample is never paired with a standard concentration.
"""
from __future__ import annotations

import csv
import hashlib
import math
import statistics
import tempfile
import warnings as pywarnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from .concentration_levels import add_decile_columns

from .combitrace_ie import _combo_formula_key
from .response_predictor import read_table, _norm_header
from .esi_model_benchmark import _metrics, _trend_metrics, _num, _text, _key
from .ms2quant_official import (
    CONCENTRATION_UNIT_CODE_TO_LABEL,
    MS2QuantEnvironment,
    check_ms2quant_environment,
    fit_ms2quant_instrument_transfer,
    run_official_ms2quant_predictions,
)
from .negative_esi_literature import (
    NEGATIVE_ESI_FEATURES,
    add_negative_esi_proxy_features,
    run_dynamic_range_correction,
    literature_context_rows,
    make_negative_esi_plots,
)

SKLEARN_AVAILABLE = False
SKLEARN_ERROR = ""
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RepeatedKFold
    SKLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency guard
    SKLEARN_ERROR = str(exc)


PUBLISHED_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Negative-ESI literature RF framework (local reconstruction)", "negative_rf_literature"),
    ("Official MS2Quant pretrained xgbTree (positive ions only)", "official_ms2quant_r"),
    ("External predicted logIE + instrument transfer", "external_logie_transfer"),
    ("Generic published-protocol RF surrogate", "published_rf_surrogate"),
)

PUBLISHED_MODE_LABELS = [x[0] for x in PUBLISHED_MODE_OPTIONS]
PUBLISHED_MODE_MAP = {x[0]: x[1] for x in PUBLISHED_MODE_OPTIONS}

ION_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Negative ESI", "negative"),
    ("Positive ESI", "positive"),
)
ION_MODE_LABELS = [x[0] for x in ION_MODE_OPTIONS]
ION_MODE_MAP = {x[0]: x[1] for x in ION_MODE_OPTIONS}

ADDUCT_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("[M-H]-", "[M-H]-"),
    ("[M+H]+", "[M+H]+"),
    ("[M]+", "[M]+"),
    ("Other / outside the principal literature model scope", "other"),
)
ADDUCT_LABELS = [x[0] for x in ADDUCT_OPTIONS]
ADDUCT_MAP = {x[0]: x[1] for x in ADDUCT_OPTIONS}

ORGANIC_MODIFIER_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Acetonitrile", "acetonitrile"),
    ("Methanol", "methanol"),
    ("Isopropanol", "isopropanol"),
    ("Acetone", "acetone"),
    ("Other / use only programmed A/B, pH and ammonium fields", "other"),
)
ORGANIC_MODIFIER_LABELS = [x[0] for x in ORGANIC_MODIFIER_OPTIONS]
ORGANIC_MODIFIER_MAP = {x[0]: x[1] for x in ORGANIC_MODIFIER_OPTIONS}


# Coefficients transcribed from the Supporting Information of the 2020
# RandFor-IE paper.  Viscosity uses organic percentage (0..100); surface
# tension and polarity use organic fraction (0..1).
_ELUENT_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "acetonitrile": {
        "A": -1.04e-4, "B": 4.36e-3, "C": 0.884,
        "D": -2.9, "E": 7.14, "F": 5.8, "G": 10.0,
        "sigma1": 71.8, "sigma2": 27.9,
    },
    "methanol": {
        "A": -3.59e-4, "B": 3.20e-2, "C": 0.903,
        "D": -2.2, "E": 5.62, "F": 5.1, "G": 10.0,
        "sigma1": 71.8, "sigma2": 22.1,
    },
    "isopropanol": {
        "A": -4.74e-4, "B": 5.89e-2, "C": 0.788,
        "D": -3.9, "E": 15.6, "F": 3.9, "G": 10.0,
        "sigma1": 71.8, "sigma2": 17.0,
    },
    "acetone": {
        "A": -3.13e-4, "B": 2.47e-2, "C": 0.902,
        "D": -2.5, "E": 6.84, "F": 5.1, "G": 10.0,
        "sigma1": 71.8, "sigma2": 22.2,
    },
}

PUBLISHED_ELUENT_FEATURES: Tuple[str, ...] = (
    "Published_Viscosity_mPa_s",
    "Published_Surface_Tension_mN_m",
    "Published_Polarity_Index",
    "Published_Aqueous_pH",
    "Published_NH4_Present",
    "Published_Formic_Acid_pct",
    "Published_Organic_Formic_Acid_pct",
    "Published_Apex_Formic_Acid_pct",
)


@dataclass
class PublishedIEConfig:
    enabled: bool = True
    mode: str = "negative_rf_literature"
    ion_mode: str = "negative"
    adduct: str = "[M-H]-"
    organic_modifier: str = "acetonitrile"
    aqueous_phase_name: str = "Water + 0.1% formic acid"
    organic_phase_name: str = "Acetonitrile"
    formic_acid_pct: float = 0.1
    organic_formic_acid_pct: float = 0.0
    aqueous_pH: float = 2.7
    nh4_present: bool = False
    negative_proxies_enabled: bool = True
    dynamic_range_mode: str = "auto"
    cv_splits: int = 5
    cv_repeats: int = 3
    random_state: int = 42
    output_language: str = "en"
    external_logie_file: Optional[Path] = None
    external_logie_sheet: str = ""
    external_combo_col: str = ""
    external_logie_col: str = ""
    rscript_path: str = ""
    instrument_name: str = "Thermo Scientific Q Exactive HF"
    concentration_unit: str = "mol_L"
    official_timeout_sec: int = 3600


@dataclass
class PublishedIEResult:
    enabled: bool
    method_code: str
    method_label: str
    n_calibration: int
    n_target: int
    comparison_rows: List[Dict[str, object]] = field(default_factory=list)
    cv_rows: List[Dict[str, object]] = field(default_factory=list)
    target_rows: List[Dict[str, object]] = field(default_factory=list)
    eluent_rows: List[Dict[str, object]] = field(default_factory=list)
    feature_rows: List[Dict[str, object]] = field(default_factory=list)
    external_logie_audit: List[Dict[str, object]] = field(default_factory=list)
    environment_rows: List[Dict[str, object]] = field(default_factory=list)
    transfer_rows: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)
    raw_metrics: Dict[str, object] = field(default_factory=dict)
    plot_paths: List[Path] = field(default_factory=list)
    negative_feature_rows: List[Dict[str, object]] = field(default_factory=list)
    dynamic_comparison_rows: List[Dict[str, object]] = field(default_factory=list)
    dynamic_cv_rows: List[Dict[str, object]] = field(default_factory=list)
    dynamic_target_rows: List[Dict[str, object]] = field(default_factory=list)
    high_range_bias_rows: List[Dict[str, object]] = field(default_factory=list)
    literature_context_rows: List[Dict[str, object]] = field(default_factory=list)
    negative_decision_rows: List[Dict[str, object]] = field(default_factory=list)
    selected_dynamic_method: str = "none"


@dataclass
class DualIEValidationResult:
    enabled: bool
    custom_model: str
    published_method: str
    comparison_rows: List[Dict[str, object]] = field(default_factory=list)
    target_rows: List[Dict[str, object]] = field(default_factory=list)
    decision_rows: List[Dict[str, object]] = field(default_factory=list)
    method_rows: List[Dict[str, object]] = field(default_factory=list)
    plot_paths: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    target_rank_spearman: Optional[float] = None
    decision_label: str = ""
    decision_summary: str = ""


def published_eluent_descriptors(
    b_pct: Optional[float],
    *,
    organic_modifier: str = "acetonitrile",
    aqueous_pH: float = 2.7,
    nh4_present: bool = False,
    formic_acid_pct: float = 0.1,
    organic_formic_acid_pct: float = 0.0,
) -> Dict[str, object]:
    """Calculate the five published eluent descriptors at an LC apex.

    ``b_pct`` is the programmed organic-mobile-phase percentage at the apex.
    When the organic solvent is not one of the four coefficient sets published
    in the SI, physical properties are left blank rather than guessed.
    """
    b = _num(b_pct)
    out: Dict[str, object] = {
        "Published_Organic_Modifier": str(organic_modifier or "other"),
        "Published_Aqueous_pH": float(aqueous_pH),
        "Published_NH4_Present": 1.0 if bool(nh4_present) else 0.0,
        "Published_Formic_Acid_pct": float(formic_acid_pct),
        "Published_Organic_Formic_Acid_pct": float(organic_formic_acid_pct),
        "Published_Apex_Formic_Acid_pct": "",
        "Published_Viscosity_mPa_s": "",
        "Published_Surface_Tension_mN_m": "",
        "Published_Polarity_Index": "",
    }
    if b is None:
        return out
    p = max(0.0, min(100.0, float(b)))
    x = p / 100.0
    coeff = _ELUENT_COEFFICIENTS.get(str(organic_modifier or "").lower())
    if not coeff:
        return out

    viscosity = coeff["A"] * p * p + coeff["B"] * p + coeff["C"]
    surface = (
        coeff["sigma1"]
        + coeff["D"] * coeff["sigma1"] * x
        + (coeff["E"] * coeff["sigma2"] - coeff["D"] * coeff["sigma1"] - coeff["sigma1"]) * x * x
        + (coeff["sigma2"] - coeff["E"] * coeff["sigma2"]) * x * x * x
    )
    polarity = coeff["F"] * x + coeff["G"] * (1.0 - x)
    apex_fa = (1.0 - x) * float(formic_acid_pct) + x * float(organic_formic_acid_pct)
    out.update({
        "Published_Apex_Formic_Acid_pct": float(apex_fa),
        "Published_Viscosity_mPa_s": float(viscosity),
        "Published_Surface_Tension_mN_m": float(surface),
        "Published_Polarity_Index": float(polarity),
    })
    return out


def add_published_eluent_features(records: Sequence[Dict[str, object]], config: PublishedIEConfig) -> None:
    for record in records:
        record.update(published_eluent_descriptors(
            _num(record.get("Mobile_phase_B_pct")),
            organic_modifier=config.organic_modifier,
            aqueous_pH=config.aqueous_pH,
            nh4_present=config.nh4_present,
            formic_acid_pct=config.formic_acid_pct,
            organic_formic_acid_pct=config.organic_formic_acid_pct,
        ))
        record["Published_Aqueous_Phase"] = config.aqueous_phase_name
        record["Published_Organic_Phase"] = config.organic_phase_name
        record["Published_Ion_Mode"] = config.ion_mode
        record["Published_Adduct"] = config.adduct


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(xx) < 2 or len(xx) != len(yy):
        return float("nan")
    if float(np.std(xx)) <= 0 or float(np.std(yy)) <= 0:
        return float("nan")
    # Average-rank implementation with stable tie handling.
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), dtype=float)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            out[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        return out
    return float(np.corrcoef(ranks(xx), ranks(yy))[0, 1])


def _dominant_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0
    _uniq, counts = np.unique(values, return_counts=True)
    return float(np.max(counts) / len(values))


def _candidate_molecular_features(records: Sequence[Dict[str, object]], descriptor_names: Sequence[str]) -> List[str]:
    # The published protocol uses molecular descriptors plus five eluent
    # descriptors.  Raw RT/%B and CombiTrace-specific gradient interactions are
    # deliberately excluded from this reference path.
    excluded = {
        "Apex_RT_min", "Effective_gradient_time_min", "Mobile_phase_A_pct",
        "Mobile_phase_B_pct", "B_slope_pct_per_min", "Gradient_state",
        "HydrophobicGradientExposure_proxy", "PolarAqueousExposure_proxy",
        "IonizableAqueousExposure_proxy", "BasicAqueousExposure_proxy",
        "AcidicAqueousExposure_proxy", "GradientChangeExposure_proxy",
    }
    names: List[str] = []
    for name in descriptor_names:
        text = str(name or "")
        if not text or text in excluded or text in names:
            continue
        # Categorical/text fields are excluded.  Formula_* counts are numeric.
        if text in {"A_Formula", "B_Formula", "C_Formula"}:
            continue
        names.append(text)
    for name in PUBLISHED_ELUENT_FEATURES:
        if name not in names:
            names.append(name)
    return names


def _paper_filter_features(
    records: Sequence[Dict[str, object]],
    candidate_names: Sequence[str],
    *,
    correlation_r2_threshold: float = 0.80,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Paper-aligned descriptor filtering fitted on one training fold."""
    audit: List[Dict[str, object]] = []
    provisional: List[Tuple[str, np.ndarray, float]] = []
    n = len(records)
    for name in candidate_names:
        vals = np.asarray([np.nan if _num(r.get(name)) is None else float(_num(r.get(name))) for r in records], dtype=float)
        finite = np.isfinite(vals)
        missing_pct = float((1.0 - np.mean(finite)) * 100.0) if n else 100.0
        row = {
            "Feature": name,
            "Calibration_rows": n,
            "Missing_pct": missing_pct,
            "Dominant_value_pct": "",
            "Variance": "",
            "Status": "",
            "Reason": "",
            "Correlated_with": "",
        }
        # Original workflow removed descriptors containing NA.  This strict
        # rule is deliberately retained for the published benchmark.
        if not np.all(finite):
            row.update({"Status": "Excluded", "Reason": "contains missing/NA values in calibration fold"})
            audit.append(row)
            continue
        dom = _dominant_fraction(vals)
        var = float(np.var(vals))
        row["Dominant_value_pct"] = dom * 100.0
        row["Variance"] = var
        if dom >= 0.95 or var <= 1e-14:
            row.update({"Status": "Excluded", "Reason": ">=95% identical values or near-zero variance"})
            audit.append(row)
            continue
        provisional.append((name, vals, var))
        row.update({"Status": "Candidate", "Reason": "passed NA and 95%-constant filters"})
        audit.append(row)

    # Retain higher-variance representatives first, then discard any feature
    # with R^2 > 0.8 to a retained descriptor, matching the paper's threshold.
    threshold_r = math.sqrt(max(0.0, min(0.999999, float(correlation_r2_threshold))))
    provisional.sort(key=lambda item: (-item[2], item[0]))
    retained: List[Tuple[str, np.ndarray]] = []
    status_by_name = {str(r["Feature"]): r for r in audit}
    for name, vals, _var in provisional:
        correlated_with = ""
        max_abs_r = 0.0
        for kept_name, kept_vals in retained:
            if float(np.std(vals)) <= 0 or float(np.std(kept_vals)) <= 0:
                continue
            corr = float(np.corrcoef(vals, kept_vals)[0, 1])
            if math.isfinite(corr) and abs(corr) > max_abs_r:
                max_abs_r = abs(corr)
                correlated_with = kept_name
        row = status_by_name[name]
        if correlated_with and max_abs_r > threshold_r:
            row.update({
                "Status": "Excluded",
                "Reason": f"correlated descriptor; r^2={max_abs_r ** 2:.4f} > {correlation_r2_threshold:.2f}",
                "Correlated_with": correlated_with,
            })
        else:
            retained.append((name, vals))
            row.update({"Status": "Selected", "Reason": "retained by published-protocol filtering"})
    return [name for name, _vals in retained], audit


def _matrix(records: Sequence[Dict[str, object]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([
        [float(_num(record.get(name))) if _num(record.get(name)) is not None else np.nan for name in names]
        for record in records
    ], dtype=float)


def _aggregate_repeated_predictions(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        idx = int(row["Calibration_Index"])
        grouped.setdefault(idx, []).append(row)
    out: List[Dict[str, object]] = []
    for idx in sorted(grouped):
        group = grouped[idx]
        pred_log_rrf = float(statistics.median(float(r["Predicted_log10_RRF"]) for r in group))
        ratio = float(group[0]["Measured_ratio"])
        actual = float(group[0]["Actual_concentration"])
        predicted = float(ratio / (10.0 ** pred_log_rrf))
        fold = max(predicted / actual, actual / predicted)
        out.append({
            "Calibration_Index": idx,
            "Source_file": group[0].get("Source_file", ""),
            "Source_sheet": group[0].get("Source_sheet", ""),
            "Source_row": group[0].get("Source_row", ""),
            "Combo": group[0].get("Combo", ""),
            "ABC_Formula_Key": group[0].get("ABC_Formula_Key", ""),
            "Actual_concentration": actual,
            "Measured_ratio": ratio,
            "Predicted_log10_RRF": pred_log_rrf,
            "Predicted_concentration": predicted,
            "Response_corrected_score_log10": math.log10(predicted),
            "Fold_error": fold,
            "CV_prediction_repeats": len(group),
        })
    return out


def _comparison_metrics(cv_rows: Sequence[Dict[str, object]], calibration: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    actual = [float(r["Actual_concentration"]) for r in cv_rows]
    predicted = [float(r["Predicted_concentration"]) for r in cv_rows]
    metrics = _metrics(actual, predicted)
    metrics.update(_trend_metrics(actual, predicted))
    raw_actual = [float(r["Actual_concentration"]) for r in calibration]
    raw_ratio = [float(r["Measured_ratio"]) for r in calibration]
    raw = _trend_metrics(raw_actual, raw_ratio)
    for key, value in raw.items():
        metrics["Raw_" + key] = value
    for key in (
        "Trend_Spearman_r", "Trend_Kendall_tau", "Trend_Pairwise_Concordance_pct",
        "Trend_Top10_Overlap_pct", "Trend_Top20_Overlap_pct", "Trend_Tertile_Accuracy_pct",
    ):
        value = _num(metrics.get(key))
        base = _num(raw.get(key))
        metrics["Delta_" + key + "_vs_raw"] = float(value - base) if value is not None and base is not None else ""
    return metrics, raw


def _load_external_logie(config: PublishedIEConfig) -> Tuple[Dict[str, float], List[Dict[str, object]], List[str]]:
    mapping: Dict[str, float] = {}
    audit: List[Dict[str, object]] = []
    warnings: List[str] = []
    path = Path(config.external_logie_file) if config.external_logie_file else None
    if not path:
        return mapping, audit, ["External logIE transfer selected but no external logIE table was provided."]
    table = read_table(path, sheet_name=config.external_logie_sheet)
    headers = table.headers
    combo_col = config.external_combo_col
    logie_col = config.external_logie_col
    if not combo_col:
        candidates = [h for h in headers if "combo" in _norm_header(h) or 'Combo' in str(h)]
        combo_col = candidates[0] if candidates else ""
    if not logie_col:
        candidates = [h for h in headers if "logie" in _norm_header(h) or "ionizationefficiency" in _norm_header(h)]
        logie_col = candidates[0] if candidates else ""
    if not combo_col or not logie_col:
        warnings.append(f"Could not identify Combo/logIE columns in {path.name}; specify them explicitly.")
        return mapping, audit, warnings
    for raw in table.rows:
        combo = _text(raw.get(combo_col))
        key = _combo_formula_key(combo)
        value = _num(raw.get(logie_col))
        status = "USED"
        reason = ""
        if not key:
            status, reason = "SKIPPED", "Combo could not be converted to ordered A/B/C Formula key"
        elif value is None:
            status, reason = "SKIPPED", "invalid or missing logIE"
        elif key in mapping:
            status, reason = "SKIPPED", "duplicate A/B/C Formula key; first value retained"
        else:
            mapping[key] = float(value)
        audit.append({
            "Source_file": raw.get("__source_file__", path.name),
            "Source_sheet": raw.get("__source_sheet__", table.sheet_name),
            "Source_row": raw.get("__source_row__", ""),
            "Combo": combo,
            "ABC_Formula_Key": key,
            "Predicted_logIE": value if value is not None else "",
            "Status": status,
            "Reason": reason,
        })
    return mapping, audit, warnings


def _external_transfer_cv(
    calibration: Sequence[Dict[str, object]],
    logie_map: Dict[str, float],
    *, cv_splits: int, cv_repeats: int, random_state: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    warnings: List[str] = []
    usable = [(i, r, logie_map.get(_key(r))) for i, r in enumerate(calibration)]
    usable = [(i, r, v) for i, r, v in usable if v is not None and math.isfinite(float(v))]
    if len(usable) < max(10, cv_splits * 2):
        return [], [f"Only {len(usable)} calibration standards have external logIE; not enough for repeated CV."]
    idxs = np.asarray([u[0] for u in usable], dtype=int)
    logie = np.asarray([float(u[2]) for u in usable], dtype=float)
    y = np.asarray([math.log10(float(u[1]["Measured_ratio"]) / float(u[1]["Actual_concentration"])) for u in usable], dtype=float)
    splitter = RepeatedKFold(
        n_splits=max(2, min(int(cv_splits), len(usable))),
        n_repeats=max(1, int(cv_repeats)),
        random_state=int(random_state),
    )
    rows: List[Dict[str, object]] = []
    for fold_no, (train_pos, test_pos) in enumerate(splitter.split(logie), start=1):
        xtr, ytr = logie[train_pos], y[train_pos]
        if len(np.unique(xtr)) < 2:
            warnings.append(f"External logIE fold {fold_no}: no logIE variation; skipped")
            continue
        slope, intercept = np.polyfit(xtr, ytr, 1)
        pred = slope * logie[test_pos] + intercept
        for pos, pred_y in zip(test_pos, pred):
            original_idx = int(idxs[pos])
            record = calibration[original_idx]
            rows.append({
                "Calibration_Index": original_idx,
                "Fold": fold_no,
                "Source_file": record.get("Source_file", ""),
                "Source_sheet": record.get("Source_sheet", ""),
                "Source_row": record.get("Source_row", ""),
                "Combo": record.get("Combo", ""),
                "ABC_Formula_Key": _key(record),
                "Actual_concentration": float(record["Actual_concentration"]),
                "Measured_ratio": float(record["Measured_ratio"]),
                "External_predicted_logIE": float(logie[pos]),
                "Transfer_slope": float(slope),
                "Transfer_intercept": float(intercept),
                "Predicted_log10_RRF": float(pred_y),
            })
    return _aggregate_repeated_predictions(rows), warnings


def _rf_surrogate_cv(
    calibration: Sequence[Dict[str, object]],
    candidate_features: Sequence[str],
    *, cv_splits: int, cv_repeats: int, random_state: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    warnings: List[str] = []
    n = len(calibration)
    splitter = RepeatedKFold(
        n_splits=max(2, min(int(cv_splits), n)),
        n_repeats=max(1, int(cv_repeats)),
        random_state=int(random_state),
    )
    rows: List[Dict[str, object]] = []
    feature_events: List[Dict[str, object]] = []
    y_all = np.asarray([
        math.log10(float(r["Measured_ratio"]) / float(r["Actual_concentration"]))
        for r in calibration
    ], dtype=float)
    for fold_no, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n)), start=1):
        train_records = [calibration[int(i)] for i in train_idx]
        selected, audit = _paper_filter_features(train_records, candidate_features)
        for row in audit:
            rr = dict(row)
            rr["Fold"] = fold_no
            feature_events.append(rr)
        if len(selected) < 3:
            warnings.append(f"Published RF surrogate fold {fold_no}: fewer than 3 filtered descriptors; skipped")
            continue
        Xtr = _matrix(train_records, selected)
        Xte = _matrix([calibration[int(i)] for i in test_idx], selected)
        # Test-only missing values can occur when a descriptor is finite in the
        # fold training rows but unavailable for one held-out row.  Median fill
        # does not influence training feature selection.
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        model = RandomForestRegressor(
            n_estimators=100,
            max_features="sqrt",
            min_samples_leaf=2,
            bootstrap=True,
            random_state=int(random_state) + fold_no,
            n_jobs=-1,
        )
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            model.fit(Xtr, y_all[train_idx])
            pred = model.predict(Xte)
        for original_idx, pred_y in zip(test_idx, pred):
            record = calibration[int(original_idx)]
            rows.append({
                "Calibration_Index": int(original_idx),
                "Fold": fold_no,
                "Source_file": record.get("Source_file", ""),
                "Source_sheet": record.get("Source_sheet", ""),
                "Source_row": record.get("Source_row", ""),
                "Combo": record.get("Combo", ""),
                "ABC_Formula_Key": _key(record),
                "Actual_concentration": float(record["Actual_concentration"]),
                "Measured_ratio": float(record["Measured_ratio"]),
                "Predicted_log10_RRF": float(pred_y),
                "Selected_feature_count": len(selected),
                "Selected_features": "; ".join(selected),
            })
    return _aggregate_repeated_predictions(rows), feature_events, warnings


def _fit_rf_surrogate_targets(
    calibration: Sequence[Dict[str, object]],
    targets: Sequence[Dict[str, object]],
    candidate_features: Sequence[str],
    *, random_state: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    selected, audit = _paper_filter_features(calibration, candidate_features)
    warnings: List[str] = []
    if len(selected) < 3:
        return [], audit, ["Published RF surrogate: fewer than 3 descriptors remained after paper-aligned filtering."]
    Xtr = _matrix(calibration, selected)
    Xtar = _matrix(targets, selected)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xtar = np.where(np.isfinite(Xtar), Xtar, med)
    y = np.asarray([
        math.log10(float(r["Measured_ratio"]) / float(r["Actual_concentration"]))
        for r in calibration
    ], dtype=float)
    model = RandomForestRegressor(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(Xtr, y)
    pred_y = model.predict(Xtar)
    importance = getattr(model, "feature_importances_", np.zeros(len(selected)))
    importance_map = {name: float(value) for name, value in zip(selected, importance)}
    for row in audit:
        row["Full_fit_importance"] = importance_map.get(str(row.get("Feature")), "")
    out: List[Dict[str, object]] = []
    for record, log_rrf in zip(targets, pred_y):
        ratio = float(record["Measured_ratio"])
        corrected = math.log10(ratio) - float(log_rrf)
        concentration = 10.0 ** corrected
        out.append({
            "Method": "Published_RF_IE_protocol_surrogate",
            "Source_file": record.get("Source_file", ""),
            "Source_sheet": record.get("Source_sheet", ""),
            "Source_row": record.get("Source_row", ""),
            "Name": record.get("Name", ""),
            "Formula": record.get("Formula", ""),
            "Combo": record.get("Combo", ""),
            "ABC_Formula_Key": _key(record),
            "Raw_area_IS_ratio": ratio,
            "Predicted_log10_RRF": float(log_rrf),
            "Predicted_RRF": 10.0 ** float(log_rrf),
            "Response_corrected_score_log10": corrected,
            "Estimated_concentration": concentration,
            "Apex_RT_min": record.get("Apex_RT_min", ""),
            "Mobile_phase_B_pct": record.get("Mobile_phase_B_pct", ""),
            "Published_Viscosity_mPa_s": record.get("Published_Viscosity_mPa_s", ""),
            "Published_Surface_Tension_mN_m": record.get("Published_Surface_Tension_mN_m", ""),
            "Published_Polarity_Index": record.get("Published_Polarity_Index", ""),
            "Published_Aqueous_pH": record.get("Published_Aqueous_pH", ""),
            "Published_NH4_Present": record.get("Published_NH4_Present", ""),
            "Published_Selected_feature_count": len(selected),
        })
    _add_target_ranks(out)
    return out, audit, warnings


def _fit_external_targets(
    calibration: Sequence[Dict[str, object]], targets: Sequence[Dict[str, object]], logie_map: Dict[str, float],
) -> Tuple[List[Dict[str, object]], List[str]]:
    usable_cal = [(r, logie_map.get(_key(r))) for r in calibration]
    usable_cal = [(r, v) for r, v in usable_cal if v is not None]
    if len(usable_cal) < 10:
        return [], [f"Only {len(usable_cal)} standards have external logIE; final transfer model not fitted."]
    x = np.asarray([float(v) for _r, v in usable_cal], dtype=float)
    y = np.asarray([
        math.log10(float(r["Measured_ratio"]) / float(r["Actual_concentration"]))
        for r, _v in usable_cal
    ], dtype=float)
    if len(np.unique(x)) < 2:
        return [], ["External logIE values have no variation; transfer regression not fitted."]
    slope, intercept = np.polyfit(x, y, 1)
    out: List[Dict[str, object]] = []
    missing = 0
    for record in targets:
        logie = logie_map.get(_key(record))
        if logie is None:
            missing += 1
            continue
        log_rrf = float(slope * float(logie) + intercept)
        ratio = float(record["Measured_ratio"])
        corrected = math.log10(ratio) - log_rrf
        concentration = 10.0 ** corrected
        out.append({
            "Method": "Published_external_logIE_instrument_transfer",
            "Source_file": record.get("Source_file", ""),
            "Source_sheet": record.get("Source_sheet", ""),
            "Source_row": record.get("Source_row", ""),
            "Name": record.get("Name", ""),
            "Formula": record.get("Formula", ""),
            "Combo": record.get("Combo", ""),
            "ABC_Formula_Key": _key(record),
            "Raw_area_IS_ratio": ratio,
            "External_predicted_logIE": float(logie),
            "Transfer_slope": float(slope),
            "Transfer_intercept": float(intercept),
            "Predicted_log10_RRF": log_rrf,
            "Predicted_RRF": 10.0 ** log_rrf,
            "Response_corrected_score_log10": corrected,
            "Estimated_concentration": concentration,
        })
    _add_target_ranks(out)
    warnings = [f"External logIE missing for {missing}/{len(targets)} target rows."] if missing else []
    return out, warnings


def _add_target_ranks(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    values = np.asarray([float(r["Estimated_concentration"]) for r in rows], dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(rows), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = rank
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for i, row in enumerate(rows):
        row["Corrected_rank_low_to_high"] = int(ranks[i])
        row["Corrected_percentile"] = float(ranks[i] / len(rows) * 100.0)
        value = float(values[i])
        row["Concentration_class"] = "Low" if value <= q1 else "Medium" if value <= q2 else "High"
    add_decile_columns(rows, "Estimated_concentration", prefix="Concentration")


def run_published_ie_benchmark(
    calibration_records: Sequence[Dict[str, object]],
    target_records: Sequence[Dict[str, object]],
    descriptor_names: Sequence[str],
    output_xlsx: Path,
    config: PublishedIEConfig,
) -> PublishedIEResult:
    if not config.enabled:
        return PublishedIEResult(False, config.mode, "Disabled", 0, 0)
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("Published IE benchmark needs scikit-learn: " + SKLEARN_ERROR)

    calibration = [dict(r) for r in calibration_records]
    targets = [dict(r) for r in target_records]
    add_published_eluent_features(calibration, config)
    add_published_eluent_features(targets, config)
    if config.negative_proxies_enabled and (config.ion_mode == "negative" or config.mode == "negative_rf_literature"):
        add_negative_esi_proxy_features(
            calibration,
            aqueous_pH=config.aqueous_pH,
            aqueous_formic_acid_pct=config.formic_acid_pct,
            organic_formic_acid_pct=config.organic_formic_acid_pct,
        )
        add_negative_esi_proxy_features(
            targets,
            aqueous_pH=config.aqueous_pH,
            aqueous_formic_acid_pct=config.formic_acid_pct,
            organic_formic_acid_pct=config.organic_formic_acid_pct,
        )
    warnings: List[str] = []
    if config.adduct not in {"[M+H]+", "[M]+", "[M-H]-"}:
        warnings.append(
            "Selected adduct is outside the principal ion forms used by the published RandFor-IE workflow; interpret the benchmark as exploratory."
        )
    if config.organic_modifier not in _ELUENT_COEFFICIENTS:
        warnings.append(
            "Organic modifier has no published coefficient set; viscosity, surface tension and polarity index are blank."
        )
    if abs(float(config.formic_acid_pct) - 0.1) < 1e-9 and abs(float(config.aqueous_pH) - 2.7) < 0.15:
        warnings.append(
            "Aqueous pH=2.7 is an editable paper-based approximation for water with 0.1% formic acid, not a measured pH."
        )

    candidate_features = _candidate_molecular_features(calibration, descriptor_names)
    if config.negative_proxies_enabled and (config.ion_mode == "negative" or config.mode == "negative_rf_literature"):
        for feature_name in NEGATIVE_ESI_FEATURES:
            if feature_name not in candidate_features:
                candidate_features.append(feature_name)
    external_audit: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    environment_rows: List[Dict[str, object]] = []
    transfer_rows: List[Dict[str, object]] = []
    official_model_used = False
    official_package_version = ""
    negative_feature_rows: List[Dict[str, object]] = []
    dynamic_comparison_rows: List[Dict[str, object]] = []
    dynamic_cv_rows: List[Dict[str, object]] = []
    dynamic_target_rows: List[Dict[str, object]] = []
    high_range_bias_rows: List[Dict[str, object]] = []
    literature_rows = literature_context_rows()
    negative_decision_rows: List[Dict[str, object]] = []
    selected_dynamic_method = "none"
    published_plot_paths: List[Path] = []

    if config.mode == "negative_rf_literature":
        if config.ion_mode != "negative":
            warnings.append("Negative-ESI literature mode was selected while ion mode is not negative; results are exploratory.")
        if config.adduct != "[M-H]-":
            warnings.append("The primary negative-ion framework is parameterized for [M-H]-; other adducts require separate validation.")
        cv_rows_base, cv_feature_events, cv_warnings = _rf_surrogate_cv(
            calibration, candidate_features,
            cv_splits=config.cv_splits, cv_repeats=config.cv_repeats, random_state=config.random_state,
        )
        warnings.extend(cv_warnings)
        target_rows_base, feature_rows, target_warnings = _fit_rf_surrogate_targets(
            calibration, targets, candidate_features, random_state=config.random_state,
        )
        warnings.extend(target_warnings)
        counts: Dict[str, Dict[str, float]] = {}
        for row in cv_feature_events:
            name = str(row.get("Feature", ""))
            stat = counts.setdefault(name, {"selected": 0.0, "total": 0.0})
            stat["total"] += 1.0
            if str(row.get("Status")) == "Selected":
                stat["selected"] += 1.0
        full_map = {str(r.get("Feature")): r for r in feature_rows}
        for name, stat in counts.items():
            row = full_map.setdefault(name, {"Feature": name})
            row["CV_fold_selection_frequency_pct"] = stat["selected"] / stat["total"] * 100.0 if stat["total"] else ""
        feature_rows = sorted(full_map.values(), key=lambda r: (
            str(r.get("Status")) != "Selected",
            -float(_num(r.get("Full_fit_importance")) or 0.0),
            str(r.get("Feature")),
        ))
        negative_feature_rows = [
            {
                "Feature": name,
                "Description": "Negative-ion structural/eluent proxy used by the literature-aligned local reconstruction",
                "Calibration_nonmissing_pct": float(np.mean([_num(r.get(name)) is not None for r in calibration]) * 100.0) if calibration else 0.0,
            }
            for name in NEGATIVE_ESI_FEATURES
        ]
        dynamic = run_dynamic_range_correction(
            cv_rows_base, target_rows_base,
            requested_mode=config.dynamic_range_mode,
            cv_splits=config.cv_splits,
            random_state=config.random_state,
        )
        warnings.extend(dynamic.warnings)
        cv_rows = dynamic.cv_rows
        target_rows = dynamic.target_rows
        dynamic_comparison_rows = dynamic.comparison_rows
        dynamic_cv_rows = dynamic.cv_rows
        dynamic_target_rows = dynamic.target_rows
        high_range_bias_rows = dynamic.high_range_bias_rows
        selected_dynamic_method = dynamic.selected_method
        base_metrics, _base_raw = _comparison_metrics(cv_rows_base, calibration) if cv_rows_base else ({"N": 0}, {})
        corrected_metrics, _corr_raw = _comparison_metrics(cv_rows, calibration) if cv_rows else ({"N": 0}, {})
        negative_decision_rows = [
            {"Item": "Framework", "Value": "Negative-ESI literature-aligned local RF reconstruction"},
            {"Item": "Instrument", "Value": config.instrument_name},
            {"Item": "Ion mode", "Value": config.ion_mode},
            {"Item": "Adduct", "Value": config.adduct},
            {"Item": "Selected dynamic-range method", "Value": selected_dynamic_method},
            {"Item": "Base median fold error", "Value": base_metrics.get("Median_Fold_Error", "")},
            {"Item": "Corrected median fold error", "Value": corrected_metrics.get("Median_Fold_Error", "")},
            {"Item": "Base P80 fold error", "Value": base_metrics.get("P80_Fold_Error", "")},
            {"Item": "Corrected P80 fold error", "Value": corrected_metrics.get("P80_Fold_Error", "")},
            {"Item": "Base trend Spearman", "Value": base_metrics.get("Trend_Spearman_r", "")},
            {"Item": "Corrected trend Spearman", "Value": corrected_metrics.get("Trend_Spearman_r", "")},
            {"Item": "Interpretation", "Value": "Empirical semiquantitative response correction. One concentration per analyte cannot establish analyte-specific saturation."},
        ]
        method_label = "Negative-ESI literature-aligned RF + cross-fitted dynamic-range calibration"
        method_code = "Negative_ESI_literature_RF_dynamic_range"
        published_plot_paths, plot_warnings = make_negative_esi_plots(output_xlsx, dynamic, feature_rows)
        warnings.extend(plot_warnings)

    elif config.mode == "official_ms2quant_r":
        # The public MS2Quant pretrained model currently supports positive ESI
        # [M+H]+ and [M]+.  Unsupported ion forms are reported explicitly and
        # are never silently passed to the model.
        if config.ion_mode != "positive" or config.adduct not in {"[M+H]+", "[M]+"}:
            warnings.append(
                "Official MS2Quant v1.1.0 public model is limited to positive ESI [M+H]+ and [M]+; "
                "the official pretrained route was not run for the selected ion mode/adduct."
            )
            cv_rows = []
            target_rows = []
            env = check_ms2quant_environment(config.rscript_path)
            environment_rows = env.rows()
        elif config.organic_modifier not in {"acetonitrile", "methanol"}:
            warnings.append(
                "Official MS2Quant accepts MeCN or MeOH as organic_modifier; select the actual B phase before running."
            )
            cv_rows = []
            target_rows = []
            env = check_ms2quant_environment(config.rscript_path)
            environment_rows = env.rows()
        else:
            # Preserve calibration and unknown-sample occurrences separately.
            # Official IE depends on the apex organic percentage, so the same
            # compound may legitimately require a different prediction under a
            # different retention-time/gradient condition.  The R bridge itself
            # deduplicates only identical structure + rounded %%B conditions.
            combined: List[Dict[str, object]] = list(calibration) + list(targets)
            prediction = run_official_ms2quant_predictions(
                combined,
                rscript_path=config.rscript_path,
                organic_modifier=config.organic_modifier,
                aqueous_pH=config.aqueous_pH,
                output_xlsx=output_xlsx,
                timeout_sec=config.official_timeout_sec,
            )
            warnings.extend(prediction.warnings)
            external_audit = prediction.audit_rows
            if prediction.environment is not None:
                environment_rows = prediction.environment.rows()
                official_package_version = prediction.environment.package_version
            transfer = fit_ms2quant_instrument_transfer(
                calibration, targets, prediction.mapping,
                concentration_unit=config.concentration_unit,
                instrument_name=config.instrument_name,
                cv_splits=config.cv_splits,
                cv_repeats=config.cv_repeats,
                random_state=config.random_state,
            )
            warnings.extend(transfer.warnings)
            cv_rows = transfer.cv_rows
            target_rows = transfer.target_rows
            transfer_rows = transfer.transfer_rows
            official_model_used = bool(prediction.mapping and cv_rows)
        method_label = f"Official MS2Quant pretrained xgbTree + {config.instrument_name} normalised-RF transfer"
        method_code = "Official_MS2Quant_pretrained_xgbTree_QEHF_transfer"

    elif config.mode == "external_logie_transfer":
        logie_map, external_audit, load_warnings = _load_external_logie(config)
        warnings.extend(load_warnings)
        cv_rows, cv_warnings = _external_transfer_cv(
            calibration, logie_map,
            cv_splits=config.cv_splits, cv_repeats=config.cv_repeats, random_state=config.random_state,
        )
        warnings.extend(cv_warnings)
        target_rows, target_warnings = _fit_external_targets(calibration, targets, logie_map)
        warnings.extend(target_warnings)
        method_label = "Published external logIE + instrument-transfer regression"
        method_code = "Published_external_logIE_transfer"
    else:
        cv_rows, cv_feature_events, cv_warnings = _rf_surrogate_cv(
            calibration, candidate_features,
            cv_splits=config.cv_splits, cv_repeats=config.cv_repeats, random_state=config.random_state,
        )
        warnings.extend(cv_warnings)
        target_rows, feature_rows, target_warnings = _fit_rf_surrogate_targets(
            calibration, targets, candidate_features, random_state=config.random_state,
        )
        warnings.extend(target_warnings)
        # Aggregate fold selection frequency for transparency.
        counts: Dict[str, Dict[str, float]] = {}
        for row in cv_feature_events:
            name = str(row.get("Feature", ""))
            stat = counts.setdefault(name, {"selected": 0.0, "total": 0.0})
            stat["total"] += 1.0
            if str(row.get("Status")) == "Selected":
                stat["selected"] += 1.0
        full_map = {str(r.get("Feature")): r for r in feature_rows}
        for name, stat in counts.items():
            row = full_map.setdefault(name, {"Feature": name})
            row["CV_fold_selection_frequency_pct"] = (
                stat["selected"] / stat["total"] * 100.0 if stat["total"] else ""
            )
        feature_rows = sorted(full_map.values(), key=lambda r: (
            str(r.get("Status")) != "Selected",
            -float(_num(r.get("Full_fit_importance")) or 0.0),
            str(r.get("Feature")),
        ))
        method_label = "Published-protocol RF-IE surrogate (local RDKit descriptors)"
        method_code = "Published_RF_IE_protocol_surrogate"

    metrics, raw = _comparison_metrics(cv_rows, calibration) if cv_rows else ({"N": 0}, _trend_metrics(
        [float(r["Actual_concentration"]) for r in calibration],
        [float(r["Measured_ratio"]) for r in calibration],
    ))
    comp = dict(metrics)
    comp.update({
        "Method": method_code,
        "Method_label": method_label,
        "Calibration_rows": len(calibration),
        "Target_rows": len(target_rows),
        "Ion_mode": config.ion_mode,
        "Adduct": config.adduct,
        "Organic_modifier": config.organic_modifier,
        "Aqueous_phase": config.aqueous_phase_name,
        "Aqueous_formic_acid_pct": config.formic_acid_pct,
        "Organic_formic_acid_pct": config.organic_formic_acid_pct,
        "Aqueous_pH": config.aqueous_pH,
        "Dynamic_range_method": selected_dynamic_method,
        "NH4_present": bool(config.nh4_present),
        "Instrument": config.instrument_name,
        "Concentration_unit": config.concentration_unit,
        "Official_MS2Quant_package_version": official_package_version,
        "Official_pretrained_model_used": bool(official_model_used),
        "Interpretation": (
            "Negative-ion literature-aligned local reconstruction using molecular, eluent and transparent anion-formation proxy descriptors, followed by cross-fitted dynamic-range calibration."
            if config.mode == "negative_rf_literature" else
            f"Official KruveLab/MS2Quant pretrained xgbTree IE prediction followed by fold-specific and final {config.instrument_name} normalised-RF transfer."
            if config.mode == "official_ms2quant_r" else
            "Published instrument-transfer equation applied to externally supplied logIE predictions."
            if config.mode == "external_logie_transfer" else
            "Protocol surrogate using local RDKit descriptors; not the official pretrained MS2Quant/RandFor-IE model."
        ),
    })
    eluent_rows = [{
        "Aqueous_phase_name": config.aqueous_phase_name,
        "Organic_phase_name": config.organic_phase_name,
        "Organic_modifier": config.organic_modifier,
        "Aqueous_formic_acid_pct": config.formic_acid_pct,
        "Organic_formic_acid_pct": config.organic_formic_acid_pct,
        "Aqueous_pH": config.aqueous_pH,
        "Dynamic_range_method": selected_dynamic_method,
        "NH4_present": bool(config.nh4_present),
        "Ion_mode": config.ion_mode,
        "Adduct": config.adduct,
        "Instrument": config.instrument_name,
        "Concentration_unit": config.concentration_unit,
        "MS2Quant_package_version": official_package_version,
        "Note": "pH is editable; use a measured value when available. Official MS2Quant requires molar calibrant concentrations; the selected unit is converted internally.",
    }]
    return PublishedIEResult(
        enabled=True,
        method_code=method_code,
        method_label=method_label,
        n_calibration=len(calibration),
        n_target=len(target_rows),
        comparison_rows=[comp],
        cv_rows=cv_rows,
        target_rows=target_rows,
        eluent_rows=eluent_rows,
        feature_rows=feature_rows,
        external_logie_audit=external_audit,
        environment_rows=environment_rows,
        transfer_rows=transfer_rows,
        warnings=warnings,
        metrics=metrics,
        raw_metrics=raw,
        plot_paths=published_plot_paths,
        negative_feature_rows=negative_feature_rows,
        dynamic_comparison_rows=dynamic_comparison_rows,
        dynamic_cv_rows=dynamic_cv_rows,
        dynamic_target_rows=dynamic_target_rows,
        high_range_bias_rows=high_range_bias_rows,
        literature_context_rows=literature_rows,
        negative_decision_rows=negative_decision_rows,
        selected_dynamic_method=selected_dynamic_method,
    )


def _custom_cv_rows(custom_result) -> List[Dict[str, object]]:
    best = str(custom_result.best_model)
    return [r for r in custom_result.cv_predictions if str(r.get("Model")) == best]


def _dual_comparison(custom_result, published: PublishedIEResult) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    raw = published.raw_metrics or custom_result.raw_trend_metrics
    rows.append({
        "Framework": "Raw_area_IS_ratio",
        "Model": "No response correction",
        "Published_or_custom": "Baseline",
        "Trend_Spearman_r": raw.get("Trend_Spearman_r", ""),
        "Trend_Kendall_tau": raw.get("Trend_Kendall_tau", ""),
        "Trend_Pairwise_Concordance_pct": raw.get("Trend_Pairwise_Concordance_pct", ""),
        "Trend_Top10_Overlap_pct": raw.get("Trend_Top10_Overlap_pct", ""),
        "Trend_Top20_Overlap_pct": raw.get("Trend_Top20_Overlap_pct", ""),
    })
    custom = dict(custom_result.best_metrics)
    custom.update({
        "Framework": "CombiTrace_IE",
        "Model": custom_result.best_model,
        "Published_or_custom": "Custom chemistry-space model",
    })
    rows.append(custom)
    pub = dict(published.metrics)
    pub.update({
        "Framework": "Published_IE_reference",
        "Model": published.method_code,
        "Published_or_custom": "Published-protocol benchmark",
    })
    rows.append(pub)
    return rows


def _target_map(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        key = _text(row.get("ABC_Formula_Key"))
        if key and key not in out:
            out[key] = row
    return out


def _dual_target_rows(custom_result, published: PublishedIEResult) -> Tuple[List[Dict[str, object]], Optional[float]]:
    custom_map = _target_map(custom_result.target_predictions)
    published_map = _target_map(published.target_rows)
    keys = sorted(set(custom_map).intersection(published_map))
    rows: List[Dict[str, object]] = []
    custom_scores: List[float] = []
    published_scores: List[float] = []
    consensus_logs: List[float] = []
    for key in keys:
        c = custom_map[key]
        p = published_map[key]
        c_est = _num(c.get("Estimated_concentration"))
        p_est = _num(p.get("Estimated_concentration"))
        if c_est is None or p_est is None or c_est <= 0 or p_est <= 0:
            continue
        c_log = math.log10(c_est)
        p_log = math.log10(p_est)
        consensus_log = float(statistics.median([c_log, p_log]))
        consensus = 10.0 ** consensus_log
        agreement = max(c_est / p_est, p_est / c_est)
        confidence = "High" if agreement <= 2.0 else "Medium" if agreement <= 5.0 else "Low"
        rows.append({
            "ABC_Formula_Key": key,
            "Source_file": c.get("Source_file", p.get("Source_file", "")),
            "Source_sheet": c.get("Source_sheet", p.get("Source_sheet", "")),
            "Source_row": c.get("Source_row", p.get("Source_row", "")),
            "Name": c.get("Name", p.get("Name", "")),
            "Formula": c.get("Formula", p.get("Formula", "")),
            "Combo": c.get("Combo", p.get("Combo", "")),
            "Raw_area_IS_ratio": c.get("Raw_area_IS_ratio", p.get("Raw_area_IS_ratio", "")),
            "CombiTrace_model": custom_result.best_model,
            "CombiTrace_corrected_score_log10": c.get("Response_corrected_score_log10", ""),
            "CombiTrace_estimated_concentration": c_est,
            "CombiTrace_rank": c.get("Corrected_rank_low_to_high", ""),
            "CombiTrace_class": c.get("Concentration_class", ""),
            "CombiTrace_decile_1_to_10": c.get("Concentration_decile_1_to_10", ""),
            "CombiTrace_level": c.get("Concentration_level", ""),
            "CombiTrace_applicability": c.get("Applicability_domain", ""),
            "Published_method": published.method_code,
            "Published_corrected_score_log10": p.get("Response_corrected_score_log10", ""),
            "Published_estimated_concentration": p_est,
            "Published_rank": p.get("Corrected_rank_low_to_high", ""),
            "Published_class": p.get("Concentration_class", ""),
            "Published_decile_1_to_10": p.get("Concentration_decile_1_to_10", ""),
            "Published_level": p.get("Concentration_level", ""),
            "Model_agreement_fold": agreement,
            "Consensus_corrected_score_log10": consensus_log,
            "Consensus_estimated_concentration": consensus,
            "Consensus_confidence": confidence,
        })
        custom_scores.append(c_log)
        published_scores.append(p_log)
        consensus_logs.append(consensus_log)
    # Add consensus rank/class after all rows are known.
    if rows:
        vals = np.asarray([float(r["Consensus_estimated_concentration"]) for r in rows], dtype=float)
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(rows), dtype=int)
        for rank, idx in enumerate(order, start=1):
            ranks[int(idx)] = rank
        q1, q2 = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
        for i, row in enumerate(rows):
            value = float(vals[i])
            row["Consensus_rank_low_to_high"] = int(ranks[i])
            row["Consensus_percentile"] = float(ranks[i] / len(rows) * 100.0)
            row["Consensus_class"] = "Low" if value <= q1 else "Medium" if value <= q2 else "High"
        add_decile_columns(rows, "Consensus_estimated_concentration", prefix="Consensus_concentration")
    rho = _safe_spearman(custom_scores, published_scores) if len(custom_scores) >= 2 else None
    return rows, rho


def _dual_decision(comparison: Sequence[Dict[str, object]], target_rho: Optional[float]) -> Tuple[str, str, List[Dict[str, object]]]:
    custom = next((r for r in comparison if r.get("Framework") == "CombiTrace_IE"), {})
    published = next((r for r in comparison if r.get("Framework") == "Published_IE_reference"), {})
    c_rho = _num(custom.get("Trend_Spearman_r"))
    p_rho = _num(published.get("Trend_Spearman_r"))
    c_delta = _num(custom.get("Delta_Trend_Spearman_r_vs_raw"))
    p_delta = _num(published.get("Delta_Trend_Spearman_r_vs_raw"))
    c_med = _num(custom.get("Median_Fold_Error"))
    p_med = _num(published.get("Median_Fold_Error"))
    both_positive = c_rho is not None and p_rho is not None and c_rho > 0 and p_rho > 0
    both_improve = c_delta is not None and p_delta is not None and c_delta > 0 and p_delta > 0
    if both_positive and both_improve and target_rho is not None and target_rho >= 0.60:
        label = "DUAL-SUPPORT \u2013 both frameworks support trend correction within the calibrated domain"
    elif (c_rho is not None and c_rho > 0) or (p_rho is not None and p_rho > 0):
        label = "DUAL-CAUTION \u2013 at least one framework shows trend gain; use consensus/ranges only"
    else:
        label = "DUAL-NO-GO \u2013 dual validation does not support reliable response correction"
    summary = (
        f"CombiTrace Spearman={c_rho if c_rho is not None else 'NA'}, median fold={c_med if c_med is not None else 'NA'}; "
        f"published benchmark Spearman={p_rho if p_rho is not None else 'NA'}, median fold={p_med if p_med is not None else 'NA'}; "
        f"unknown-sample rank agreement={target_rho if target_rho is not None else 'NA'}. "
        + (
            "The literature path used the official public MS2Quant pretrained xgbTree and a fold-specific instrument transfer."
            if str(published.get("Model", "")) == "Official_MS2Quant_pretrained_xgbTree_QEHF_transfer" else
            "The literature path used externally supplied logIE values and an instrument transfer."
            if str(published.get("Model", "")) == "Published_external_logIE_transfer" else
            "The local literature-protocol route is an exploratory surrogate, not the official pretrained model."
        )
    )
    rows = [
        {"Criterion": "OVERALL DUAL DECISION", "Observed": label, "Interpretation": summary},
        {"Criterion": "CombiTrace corrected Spearman", "Observed": c_rho if c_rho is not None else "", "Interpretation": "OOF standard trend"},
        {"Criterion": "Published benchmark corrected Spearman", "Observed": p_rho if p_rho is not None else "", "Interpretation": "OOF standard trend"},
        {"Criterion": "Both improve over raw ratio", "Observed": "Yes" if both_improve else "No", "Interpretation": "Primary evidence for response correction"},
        {"Criterion": "Unknown-sample model rank agreement", "Observed": target_rho if target_rho is not None else "", "Interpretation": "Agreement is supportive, not external truth"},
    ]
    return label, summary, rows


def _short_plot_dir(output_xlsx: Path) -> Path:
    token = hashlib.sha1(str(Path(output_xlsx).resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8]
    out = Path(output_xlsx).parent / ("_dualplots_" + token)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return out
    except Exception:
        tmp = Path(tempfile.gettempdir()) / ("combitrace_dual_" + token)
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


def _make_dual_plots(
    output_xlsx: Path,
    comparison: Sequence[Dict[str, object]],
    target_rows: Sequence[Dict[str, object]],
    *, output_language: str = "en",
) -> Tuple[List[Path], List[str]]:
    paths: List[Path] = []
    warnings: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return paths, [f"Dual-model plots skipped: {exc}"]
    out_dir = _short_plot_dir(output_xlsx)
    zh = str(output_language).lower().startswith("zh")

    def save(fig, name: str):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    model_rows = [r for r in comparison if r.get("Framework") != "Raw_area_IS_ratio"]
    try:
        labels = [str(r.get("Framework")) for r in model_rows]
        x = np.arange(len(labels))
        raw_rho = _num(next((r for r in comparison if r.get("Framework") == "Raw_area_IS_ratio"), {}).get("Trend_Spearman_r"))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - 0.18, [float(_num(r.get("Trend_Spearman_r")) or 0.0) for r in model_rows], width=0.36, label='Corrected' if zh else "Corrected")
        ax.bar(x + 0.18, [float(raw_rho or 0.0)] * len(labels), width=0.36, label='Raw ratio' if zh else "Raw ratio")
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel('Spearman rank correlation' if zh else "Spearman rank correlation")
        ax.set_title('Dual models: out-of-fold trend recovery' if zh else "Dual models: out-of-fold trend recovery")
        ax.legend()
        save(fig, "01_dual_trend.png")
    except Exception as exc:
        warnings.append(f"Dual trend plot skipped: {exc}")

    try:
        labels = [str(r.get("Framework")) for r in model_rows]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - 0.18, [float(_num(r.get("Median_Fold_Error")) or np.nan) for r in model_rows], width=0.36, label='Median fold' if zh else "Median fold")
        ax.bar(x + 0.18, [float(_num(r.get("P80_Fold_Error")) or np.nan) for r in model_rows], width=0.36, label="P80")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel('Fold error' if zh else "Fold error")
        ax.set_title('Dual models: concentration-range error' if zh else "Dual models: concentration-range error")
        ax.legend()
        save(fig, "02_dual_fold.png")
    except Exception as exc:
        warnings.append(f"Dual fold plot skipped: {exc}")

    try:
        if target_rows:
            x = np.asarray([float(r["CombiTrace_rank"]) for r in target_rows if _num(r.get("CombiTrace_rank")) is not None and _num(r.get("Published_rank")) is not None])
            y = np.asarray([float(r["Published_rank"]) for r in target_rows if _num(r.get("CombiTrace_rank")) is not None and _num(r.get("Published_rank")) is not None])
            if len(x):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(x, y, alpha=0.6)
                lo = min(float(np.min(x)), float(np.min(y)))
                hi = max(float(np.max(x)), float(np.max(y)))
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                rho = _safe_spearman(x, y)
                ax.set_xlabel('CombiTrace rank' if zh else "CombiTrace rank")
                ax.set_ylabel('Published benchmark rank' if zh else "Published benchmark rank")
                ax.set_title(('Unknown-sample rank agreement' if zh else "Unknown-sample rank agreement") + f" (rho={rho:.3f})")
                save(fig, "03_target_rank_agreement.png")
    except Exception as exc:
        warnings.append(f"Target agreement plot skipped: {exc}")

    try:
        if target_rows:
            counts = {"High": 0, "Medium": 0, "Low": 0}
            for row in target_rows:
                counts[str(row.get("Consensus_confidence", "Low"))] = counts.get(str(row.get("Consensus_confidence", "Low")), 0) + 1
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(list(counts.keys()), list(counts.values()))
            ax.set_ylabel('Compound count' if zh else "Compound count")
            ax.set_title('Dual-model agreement confidence' if zh else "Dual-model agreement confidence")
            save(fig, "04_consensus_confidence.png")
    except Exception as exc:
        warnings.append(f"Consensus plot skipped: {exc}")
    return paths, warnings


def run_dual_ie_validation(
    custom_result,
    published_result: PublishedIEResult,
    output_xlsx: Path,
    *, output_language: str = "en",
) -> DualIEValidationResult:
    comparison = _dual_comparison(custom_result, published_result)
    target_rows, target_rho = _dual_target_rows(custom_result, published_result)
    label, summary, decision_rows = _dual_decision(comparison, target_rho)
    method_rows = [
        {
            "Framework": "CombiTrace_IE",
            "Method": custom_result.best_model,
            "Training_target": "log10 RRF from known standards",
            "Supervision": "known-standard injections only",
            "Feature_strategy": "automatic nested-CV selection from structure/gradient descriptors",
            "Status": "custom chemistry-space model",
        },
        {
            "Framework": "Published_IE_reference",
            "Method": published_result.method_code,
            "Training_target": "log10 RRF or external logIE-to-RF transfer",
            "Supervision": "known-standard injections only",
            "Feature_strategy": "published eluent descriptors + paper-aligned RF filtering",
            "Status": (
                "official public pretrained xgbTree + local instrument transfer"
                if published_result.method_code == "Official_MS2Quant_pretrained_xgbTree_QEHF_transfer"
                else "external published-logIE transfer"
                if published_result.method_code == "Published_external_logIE_transfer"
                else "negative-ion literature-aligned local reconstruction"
                if published_result.method_code == "Negative_ESI_literature_RF_dynamic_range"
                else "protocol surrogate; not original pretrained model"
            ),
        },
    ]
    plots, plot_warnings = _make_dual_plots(output_xlsx, comparison, target_rows, output_language=output_language)
    warnings = list(published_result.warnings) + plot_warnings
    return DualIEValidationResult(
        enabled=True,
        custom_model=custom_result.best_model,
        published_method=published_result.method_code,
        comparison_rows=comparison,
        target_rows=target_rows,
        decision_rows=decision_rows,
        method_rows=method_rows,
        plot_paths=plots,
        warnings=warnings,
        target_rank_spearman=target_rho,
        decision_label=label,
        decision_summary=summary,
    )


def _append_dict_sheet(wb, title: str, rows: Sequence[Dict[str, object]], headers: Optional[Sequence[str]] = None):
    ws = wb.create_sheet(title)
    if headers is None:
        headers = list(rows[0].keys()) if rows else ["Message"]
    ws.append(list(headers))
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    return ws


def append_published_ie_to_workbook(wb, published: PublishedIEResult, dual: DualIEValidationResult) -> None:
    _append_dict_sheet(wb, "Dual_Model_Decision", dual.decision_rows)
    _append_dict_sheet(wb, "Dual_Model_Comparison", dual.comparison_rows)
    _append_dict_sheet(wb, "Dual_Model_Targets", dual.target_rows)
    _append_dict_sheet(wb, "Dual_Model_Methods", dual.method_rows)
    _append_dict_sheet(wb, "Published_IE_CV", published.cv_rows)
    _append_dict_sheet(wb, "Published_IE_Targets", published.target_rows)
    _append_dict_sheet(wb, "Published_IE_Eluent", published.eluent_rows)
    _append_dict_sheet(wb, "Published_IE_Features", published.feature_rows)
    _append_dict_sheet(wb, "MS2Quant_IE_Audit", published.external_logie_audit)
    _append_dict_sheet(wb, "MS2Quant_Environment", published.environment_rows)
    _append_dict_sheet(wb, "MS2Quant_QEHF_Transfer", published.transfer_rows)
    _append_dict_sheet(wb, "NegESI_Decision", published.negative_decision_rows)
    _append_dict_sheet(wb, "Negative_ESI_Features", published.negative_feature_rows)
    _append_dict_sheet(wb, "Dynamic_Range_Comparison", published.dynamic_comparison_rows)
    _append_dict_sheet(wb, "Dynamic_Range_CV", published.dynamic_cv_rows)
    _append_dict_sheet(wb, "Dynamic_Range_Targets", published.dynamic_target_rows)
    _append_dict_sheet(wb, "High_Range_Bias", published.high_range_bias_rows)
    _append_dict_sheet(wb, "Literature_Context", published.literature_context_rows)

    ws = wb.create_sheet("Published_IE_Readme")
    rows = [
        ["Item", "Value"],
        ["Published benchmark mode", published.method_code],
        ["Published benchmark label", published.method_label],
        ["Known-standard calibration rows", published.n_calibration],
        ["Unknown target rows", published.n_target],
        ["Supervision", "Only separate known-standard injections; unknown real-sample ratios are never paired with standard concentrations"],
        ["Aqueous phase", published.eluent_rows[0].get("Aqueous_phase_name", "") if published.eluent_rows else ""],
        ["Formic acid (%)", published.eluent_rows[0].get("Formic_acid_pct", "") if published.eluent_rows else ""],
        ["Aqueous pH", published.eluent_rows[0].get("Aqueous_pH", "") if published.eluent_rows else ""],
        ["Organic modifier", published.eluent_rows[0].get("Organic_modifier", "") if published.eluent_rows else ""],
        ["NH4 present", published.eluent_rows[0].get("NH4_present", "") if published.eluent_rows else ""],
        ["Ion mode", published.eluent_rows[0].get("Ion_mode", "") if published.eluent_rows else ""],
        ["Adduct", published.eluent_rows[0].get("Adduct", "") if published.eluent_rows else ""],
        ["Official software", "KruveLab/MS2Quant R package installed from the public GitHub repository; the package bundles the pretrained xgbTree IE model."],
        ["Official training scope", "1191 unique chemicals, 6049 IE data points, approximately 8 orders of magnitude; current public model supports positive ESI [M+H]+ and [M]+."],
        ["Instrument adaptation", "Official predicted logIE is converted to this Thermo Scientific Q Exactive HF method by fold-specific and final log10(normalized RF)=slope*logIE+intercept transfer using known standards."],
        ["Response basis", "Area/internal-standard ratio. This is a normalized-RF transfer, not the full multi-level MS2Quant_quantify calibration-curve workflow."],
        ["Concentration units", "MS2Quant uses molar concentration. Mass concentration is converted with molecular weight; verify the selected unit."],
        ["Official installation", "Run r_scripts/install_ms2quant.R or install with devtools::install_github('kruvelab/MS2Quant', ref='main', INSTALL_opts='--no-multiarch')."],
        ["External logIE option", "Import predicted logIE values from another official/validated tool; the software then fits the instrument transfer inside CV and on all standards."],
        ["Local surrogate option", "The local RF protocol surrogate remains available only as an exploratory fallback and must not be described as the official pretrained model."],
        ["Negative-ion framework", "Uses literature-aligned eluent variables plus transparent acidity, ionization and anion-delocalization proxies. It is a local reconstruction, not the original fitted RandFor-IE weights."],
        ["Aqueous formic acid (%)", published.eluent_rows[0].get("Aqueous_formic_acid_pct", "") if published.eluent_rows else ""],
        ["Organic-phase formic acid (%)", published.eluent_rows[0].get("Organic_formic_acid_pct", "") if published.eluent_rows else ""],
        ["Dynamic-range method", published.selected_dynamic_method],
        ["Dynamic-range limitation", "A global cross-compound correction cannot prove analyte-specific saturation when each analyte has only one concentration level."],
        ["0.1% formic acid", "Default aqueous pH 2.7 is an editable approximation; replace with a measured pH when available."],
        ["Interpretation", "Use fold-external trend gain, dual agreement, applicability domain and concentration ranges; neither framework replaces analyte-specific calibration curves."],
    ]
    for row in rows:
        ws.append(row)
    ws.append([])
    ws.append(["Warnings"])
    for warning in dual.warnings:
        ws.append([warning])

    for title in ("Published_IE_CV", "Published_IE_Targets", "MS2Quant_IE_Audit"):
        if title in wb.sheetnames:
            wb[title].sheet_state = "hidden"

    if published.plot_paths:
        nps = wb.create_sheet("Negative_ESI_Plots")
        nps.column_dimensions["A"].width = 110
        row = 1
        buffers = []
        try:
            from openpyxl.drawing.image import Image as XLImage
            from io import BytesIO
            from PIL import Image as PILImage
            for path in published.plot_paths:
                try:
                    data = Path(path).read_bytes()
                    buffer = BytesIO(data)
                    buffers.append(buffer)
                    with PILImage.open(BytesIO(data)) as image:
                        width, height = image.size
                    img = XLImage(buffer)
                    max_width = 1050
                    if width > max_width:
                        scale = max_width / width
                        img.width = max_width
                        img.height = int(height * scale)
                    nps.add_image(img, f"A{row}")
                    row += max(20, int(img.height / 20) + 3)
                except Exception as exc:
                    nps.cell(row=row, column=1, value=f"Plot skipped: {path} ({exc})")
                    row += 2
            nps._negative_image_buffers = buffers
        except Exception as exc:
            nps.append([f"Could not embed plots: {exc}"])

    if dual.plot_paths:
        ps = wb.create_sheet("Dual_Model_Plots")
        ps.column_dimensions["A"].width = 110
        row = 1
        buffers = []
        try:
            from openpyxl.drawing.image import Image as XLImage
            from io import BytesIO
            from PIL import Image as PILImage
            for path in dual.plot_paths:
                try:
                    data = Path(path).read_bytes()
                    buffer = BytesIO(data)
                    buffers.append(buffer)
                    # Validate image before embedding; use a second buffer for openpyxl.
                    with PILImage.open(BytesIO(data)) as image:
                        width, height = image.size
                    img = XLImage(buffer)
                    max_width = 1050
                    if width > max_width:
                        scale = max_width / width
                        img.width = max_width
                        img.height = int(height * scale)
                    ps.add_image(img, f"A{row}")
                    row += max(20, int(img.height / 20) + 3)
                except Exception as exc:
                    ps.cell(row=row, column=1, value=f"Plot skipped: {path} ({exc})")
                    row += 2
            ps._dual_image_buffers = buffers  # keep buffers alive through workbook save
        except Exception as exc:
            ps.append([f"Could not embed plots: {exc}"])
