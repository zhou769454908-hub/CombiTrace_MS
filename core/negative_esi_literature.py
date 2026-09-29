"""Negative-ESI literature-aligned response modelling utilities.

This module adds two transparent layers to the local IE benchmark:

1. negative-ion-specific structural and eluent proxy descriptors;
2. an optional cross-fitted dynamic-range calibration for response compression.

The proxies are intentionally labelled as proxies. They are not experimental
pKa values, COSMO-RS descriptors, or a replacement for analyte-specific
calibration curves.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from .concentration_levels import add_decile_columns

from .esi_model_benchmark import _metrics, _trend_metrics, _num

SKLEARN_AVAILABLE = False
SKLEARN_ERROR = ""
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import HuberRegressor, Ridge
    from sklearn.model_selection import KFold
    SKLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    SKLEARN_ERROR = str(exc)


NEGATIVE_ESI_FEATURES: Tuple[str, ...] = (
    "Published_Apex_Formic_Acid_pct",
    "NegESI_DeprotonatableSiteCount_proxy",
    "NegESI_StrongAcidSiteCount_proxy",
    "NegESI_MediumAcidSiteCount_proxy",
    "NegESI_WeakAcidSiteCount_proxy",
    "NegESI_pKaClass_proxy",
    "NegESI_pH_minus_pKa_proxy",
    "NegESI_SolutionIonizedFraction_proxy",
    "NegESI_EWGCount_proxy",
    "NegESI_AcidicSiteDensity_proxy",
    "NegESI_AnionChargeDelocalization_proxy",
    "NegESI_AnionStability_proxy",
    "NegESI_IonizationDelocalization_proxy",
    "NegESI_OrganicFractionDelocalization_proxy",
    "NegESI_SurfaceTensionChargeRelease_proxy",
    "NegESI_ViscosityDesolvation_proxy",
    "NegESI_PolarityIonization_proxy",
    "NegESI_HydrophobicIonRelease_proxy",
)

DYNAMIC_RANGE_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Automatic cross-validated selection", "auto"),
    ("No dynamic-range correction", "none"),
    ("Robust power-law decompression", "robust_linear"),
    ("Monotonic isotonic calibration", "isotonic"),
    ("High-range piecewise calibration", "piecewise"),
)
DYNAMIC_RANGE_MODE_LABELS = [x[0] for x in DYNAMIC_RANGE_MODE_OPTIONS]
DYNAMIC_RANGE_MODE_MAP = {x[0]: x[1] for x in DYNAMIC_RANGE_MODE_OPTIONS}


def _f(record: Dict[str, object], key: str, default: float = 0.0) -> float:
    value = _num(record.get(key))
    return float(value) if value is not None and math.isfinite(float(value)) else float(default)


def add_negative_esi_proxy_features(
    records: Sequence[Dict[str, object]],
    *,
    aqueous_pH: float,
    aqueous_formic_acid_pct: float,
    organic_formic_acid_pct: float,
) -> None:
    """Add negative-ion-specific transparent proxy descriptors in-place."""
    for record in records:
        a_pct = max(0.0, min(100.0, _f(record, "Mobile_phase_A_pct", 100.0)))
        b_pct = max(0.0, min(100.0, _f(record, "Mobile_phase_B_pct", 0.0)))
        a_frac = a_pct / 100.0
        b_frac = b_pct / 100.0
        apex_fa = a_frac * float(aqueous_formic_acid_pct) + b_frac * float(organic_formic_acid_pct)

        strong = _f(record, "SulfonicAcidCount") + _f(record, "PhosphoricOHCount")
        medium = _f(record, "CarboxylicAcidCount") + _f(record, "ImideNHCount")
        weak = _f(record, "PhenolCount") + _f(record, "ThiolCount")
        deprot = _f(record, "AcidicSiteCount_proxy", strong + medium + weak)

        # Coarse acid-class proxy only. It is intentionally not named predicted pKa.
        if strong > 0:
            pka_proxy = 1.5
        elif medium > 0:
            pka_proxy = 4.5
        elif weak > 0:
            pka_proxy = 9.5
        else:
            pka_proxy = 12.0
        delta = float(aqueous_pH) - pka_proxy
        exponent = max(-30.0, min(30.0, pka_proxy - float(aqueous_pH)))
        ionized_fraction = 1.0 / (1.0 + 10.0 ** exponent)

        heavy = max(1.0, _f(record, "HeavyAtomCount", 1.0))
        aromatic = _f(record, "AromaticRingCount")
        logp = _f(record, "MolLogP")
        neg_charge = _f(record, "GasteigerNegativeChargeAbsSum")
        charge_range = _f(record, "GasteigerChargeRange")
        charge_sep = _f(record, "GasteigerChargeSeparation")
        ewg = _f(record, "ElectronWithdrawingGroupCount_proxy")
        if ewg <= 0:
            # Fallback proxy when the explicit SMARTS descriptor is not present.
            ewg = (
                _f(record, "Formula_O")
                + 0.5 * _f(record, "Formula_N")
                + _f(record, "Formula_F")
                + _f(record, "Formula_Cl")
                + _f(record, "Formula_Br")
                + _f(record, "Formula_I")
            )
        asa = max(1e-6, _f(record, "LabuteASA", 1.0))
        viscosity = max(1e-6, _f(record, "Published_Viscosity_mPa_s", 1.0))
        surface = max(1e-6, _f(record, "Published_Surface_Tension_mN_m", 72.0))
        polarity = _f(record, "Published_Polarity_Index", 10.0)

        delocalization = (
            neg_charge + 0.5 * charge_range + 0.25 * charge_sep
        ) / math.sqrt(heavy)
        delocalization *= 1.0 + 0.12 * aromatic + 0.06 * ewg
        acid_density = deprot / heavy
        stability = ionized_fraction * (1.0 + delocalization) * (1.0 + 0.05 * max(0.0, logp))

        record.update({
            "Published_Apex_Formic_Acid_pct": float(apex_fa),
            "NegESI_DeprotonatableSiteCount_proxy": float(deprot),
            "NegESI_StrongAcidSiteCount_proxy": float(strong),
            "NegESI_MediumAcidSiteCount_proxy": float(medium),
            "NegESI_WeakAcidSiteCount_proxy": float(weak),
            "NegESI_pKaClass_proxy": float(pka_proxy),
            "NegESI_pH_minus_pKa_proxy": float(delta),
            "NegESI_SolutionIonizedFraction_proxy": float(ionized_fraction),
            "NegESI_EWGCount_proxy": float(ewg),
            "NegESI_AcidicSiteDensity_proxy": float(acid_density),
            "NegESI_AnionChargeDelocalization_proxy": float(delocalization),
            "NegESI_AnionStability_proxy": float(stability),
            "NegESI_IonizationDelocalization_proxy": float(ionized_fraction * delocalization),
            "NegESI_OrganicFractionDelocalization_proxy": float(b_frac * delocalization),
            "NegESI_SurfaceTensionChargeRelease_proxy": float((1.0 + delocalization) / surface),
            "NegESI_ViscosityDesolvation_proxy": float((1.0 + max(0.0, logp)) / viscosity),
            "NegESI_PolarityIonization_proxy": float(polarity * ionized_fraction),
            "NegESI_HydrophobicIonRelease_proxy": float(max(0.0, logp) * b_frac * (1.0 + delocalization)),
        })


@dataclass
class DynamicRangeResult:
    selected_method: str
    comparison_rows: List[Dict[str, object]]
    cv_rows: List[Dict[str, object]]
    target_rows: List[Dict[str, object]]
    high_range_bias_rows: List[Dict[str, object]]
    fit_info: Dict[str, object]
    warnings: List[str]


def _log_arrays(cv_rows: Sequence[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    actual: List[float] = []
    base: List[float] = []
    indices: List[int] = []
    for i, row in enumerate(cv_rows):
        a = _num(row.get("Actual_concentration"))
        p = _num(row.get("Predicted_concentration"))
        if a is None or p is None or a <= 0 or p <= 0:
            continue
        actual.append(math.log10(float(a)))
        base.append(math.log10(float(p)))
        indices.append(i)
    return np.asarray(actual, dtype=float), np.asarray(base, dtype=float), indices


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    if len(x) < 3:
        return {"kind": "identity"}
    try:
        model = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500)
        model.fit(x.reshape(-1, 1), y)
        return {"kind": "robust_linear", "intercept": float(model.intercept_), "slope": float(model.coef_[0])}
    except Exception:
        slope, intercept = np.polyfit(x, y, 1)
        return {"kind": "robust_linear", "intercept": float(intercept), "slope": float(slope)}


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    if len(np.unique(x)) < 3:
        return _fit_linear(x, y)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(x, y)
    return {
        "kind": "isotonic",
        "x": [float(v) for v in model.X_thresholds_],
        "y": [float(v) for v in model.y_thresholds_],
    }


def _piecewise_design(x: np.ndarray, knee: float) -> np.ndarray:
    return np.column_stack([x, np.maximum(0.0, x - float(knee))])


def _fit_piecewise(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict[str, object]:
    if len(x) < 8 or len(np.unique(x)) < 4:
        return _fit_linear(x, y)
    candidates = sorted(set(float(np.quantile(x, q)) for q in (0.50, 0.60, 0.70, 0.80)))
    n_splits = max(2, min(4, len(x) // 4))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    best = None
    for knee in candidates:
        errors: List[float] = []
        for tr, te in splitter.split(x):
            model = Ridge(alpha=1e-3)
            model.fit(_piecewise_design(x[tr], knee), y[tr])
            pred = model.predict(_piecewise_design(x[te], knee))
            errors.extend(np.abs(pred - y[te]).tolist())
        score = float(np.median(errors)) if errors else float("inf")
        if best is None or score < best[0]:
            best = (score, knee)
    knee = best[1] if best is not None else float(np.median(x))
    model = Ridge(alpha=1e-3)
    model.fit(_piecewise_design(x, knee), y)
    return {
        "kind": "piecewise",
        "intercept": float(model.intercept_),
        "slope_low": float(model.coef_[0]),
        "slope_high_delta": float(model.coef_[1]),
        "knee": float(knee),
    }


def _fit_calibrator(method: str, x: np.ndarray, y: np.ndarray, random_state: int) -> Dict[str, object]:
    if method == "none":
        return {"kind": "identity"}
    if method == "robust_linear":
        return _fit_linear(x, y)
    if method == "isotonic":
        return _fit_isotonic(x, y)
    if method == "piecewise":
        return _fit_piecewise(x, y, random_state=random_state)
    return {"kind": "identity"}


def _apply_calibrator(info: Dict[str, object], x: np.ndarray) -> np.ndarray:
    kind = str(info.get("kind", "identity"))
    if kind == "robust_linear":
        return float(info.get("intercept", 0.0)) + float(info.get("slope", 1.0)) * x
    if kind == "isotonic":
        xx = np.asarray(info.get("x", []), dtype=float)
        yy = np.asarray(info.get("y", []), dtype=float)
        if len(xx) >= 2:
            return np.interp(x, xx, yy, left=yy[0], right=yy[-1])
        return x
    if kind == "piecewise":
        knee = float(info.get("knee", 0.0))
        return (
            float(info.get("intercept", 0.0))
            + float(info.get("slope_low", 1.0)) * x
            + float(info.get("slope_high_delta", 0.0)) * np.maximum(0.0, x - knee)
        )
    return x


def _crossfit_method(method: str, x: np.ndarray, y: np.ndarray, cv_splits: int, random_state: int) -> np.ndarray:
    if method == "none":
        return x.copy()
    n_splits = max(2, min(int(cv_splits), len(x)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    pred = np.full(len(x), np.nan, dtype=float)
    for fold_no, (tr, te) in enumerate(splitter.split(x), start=1):
        info = _fit_calibrator(method, x[tr], y[tr], random_state + fold_no)
        pred[te] = _apply_calibrator(info, x[te])
    return pred


def _dynamic_metrics(actual_log: np.ndarray, pred_log: np.ndarray) -> Dict[str, object]:
    actual = 10.0 ** actual_log
    pred = 10.0 ** pred_log
    out = _metrics(actual, pred)
    out.update(_trend_metrics(actual, pred))
    signed = pred_log - actual_log
    abs_log = np.abs(signed)
    out["Median_abs_log10_error"] = float(np.median(abs_log))
    out["P80_abs_log10_error"] = float(np.quantile(abs_log, 0.80))
    q3 = float(np.quantile(actual_log, 0.75))
    high = signed[actual_log >= q3]
    out["High_quartile_median_log10_bias"] = float(np.median(high)) if len(high) else float("nan")
    spearman = _num(out.get("Trend_Spearman_r"))
    out["Selection_score"] = (
        float(out["Median_abs_log10_error"])
        + 0.25 * float(out["P80_abs_log10_error"])
        + 0.25 * abs(float(out["High_quartile_median_log10_bias"]))
        - 0.10 * max(0.0, float(spearman or 0.0))
    )
    return out


def _high_range_rows(actual_log: np.ndarray, base_log: np.ndarray, corrected_log: np.ndarray) -> List[Dict[str, object]]:
    q = np.quantile(actual_log, [0.0, 0.25, 0.50, 0.75, 1.0])
    rows: List[Dict[str, object]] = []
    labels = ("Q1_low", "Q2", "Q3", "Q4_high")
    for i, label in enumerate(labels):
        if i < 3:
            mask = (actual_log >= q[i]) & (actual_log < q[i + 1])
        else:
            mask = (actual_log >= q[i]) & (actual_log <= q[i + 1])
        for stage, pred_log in (("Base", base_log), ("Dynamic_corrected", corrected_log)):
            signed = pred_log[mask] - actual_log[mask]
            fold = 10.0 ** np.abs(signed)
            rows.append({
                "Concentration_quartile": label,
                "Stage": stage,
                "N": int(np.sum(mask)),
                "Median_signed_log10_error": float(np.median(signed)) if len(signed) else "",
                "Geometric_bias_fold": float(10.0 ** np.median(signed)) if len(signed) else "",
                "Median_fold_error": float(np.median(fold)) if len(fold) else "",
                "Underprediction_pct": float(np.mean(signed < 0) * 100.0) if len(signed) else "",
                "Overprediction_pct": float(np.mean(signed > 0) * 100.0) if len(signed) else "",
            })
    return rows


def _rank_target_rows(rows: List[Dict[str, object]], concentration_key: str) -> None:
    valid = [(i, _num(r.get(concentration_key))) for i, r in enumerate(rows)]
    valid = [(i, float(v)) for i, v in valid if v is not None and v > 0]
    if not valid:
        return
    values = np.asarray([v for _i, v in valid], dtype=float)
    order = np.argsort(values, kind="mergesort")
    rank_by_pos = np.empty(len(valid), dtype=int)
    for rank, pos in enumerate(order, start=1):
        rank_by_pos[int(pos)] = rank
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for pos, (row_idx, value) in enumerate(valid):
        row = rows[row_idx]
        row["Dynamic_rank_low_to_high"] = int(rank_by_pos[pos])
        row["Dynamic_percentile"] = float(rank_by_pos[pos] / len(valid) * 100.0)
        row["Dynamic_concentration_class"] = "Low" if value <= q1 else "Medium" if value <= q2 else "High"
    add_decile_columns(rows, concentration_key, prefix="Dynamic_concentration")


def run_dynamic_range_correction(
    cv_rows: Sequence[Dict[str, object]],
    target_rows: Sequence[Dict[str, object]],
    *,
    requested_mode: str = "auto",
    cv_splits: int = 5,
    random_state: int = 42,
) -> DynamicRangeResult:
    warnings: List[str] = []
    if not SKLEARN_AVAILABLE:
        return DynamicRangeResult("none", [], list(cv_rows), list(target_rows), [], {"kind": "identity"}, [
            "Dynamic-range correction was skipped because scikit-learn is unavailable: " + SKLEARN_ERROR
        ])
    actual_log, base_log, original_indices = _log_arrays(cv_rows)
    if len(actual_log) < 12:
        return DynamicRangeResult("none", [], list(cv_rows), list(target_rows), [], {"kind": "identity"}, [
            "Dynamic-range correction requires at least 12 valid fold-external calibration predictions."
        ])

    methods = ["none", "robust_linear", "isotonic", "piecewise"]
    comparison: List[Dict[str, object]] = []
    predictions: Dict[str, np.ndarray] = {}
    for method in methods:
        try:
            pred_log = _crossfit_method(method, base_log, actual_log, cv_splits, random_state)
            metrics = _dynamic_metrics(actual_log, pred_log)
            predictions[method] = pred_log
            row = {"Method": method}
            row.update(metrics)
            comparison.append(row)
        except Exception as exc:
            warnings.append(f"Dynamic-range method {method} failed: {exc}")

    available = {str(r["Method"]): r for r in comparison}
    requested = str(requested_mode or "auto")
    if requested != "auto" and requested in predictions:
        selected = requested
    else:
        selectable = [r for r in comparison if str(r.get("Method")) in predictions]
        selected = str(min(selectable, key=lambda r: float(r.get("Selection_score", float("inf"))))["Method"]) if selectable else "none"
    selected_pred_log = predictions.get(selected, base_log.copy())
    full_fit = _fit_calibrator(selected, base_log, actual_log, random_state)

    corrected_cv_rows: List[Dict[str, object]] = []
    selected_by_index = {idx: pred for idx, pred in zip(original_indices, selected_pred_log)}
    for i, row in enumerate(cv_rows):
        rr = dict(row)
        base_est = _num(rr.get("Predicted_concentration"))
        rr["Base_estimated_concentration"] = base_est if base_est is not None else ""
        rr["Dynamic_range_method"] = selected
        if i in selected_by_index:
            corr_log = float(selected_by_index[i])
            corr = 10.0 ** corr_log
            rr["Dynamic_corrected_score_log10"] = corr_log
            rr["Dynamic_corrected_concentration"] = corr
            actual = float(rr["Actual_concentration"])
            rr["Dynamic_fold_error"] = max(corr / actual, actual / corr)
            # Compatibility with downstream dual-model evaluation.
            rr["Predicted_concentration"] = corr
            rr["Response_corrected_score_log10"] = corr_log
        corrected_cv_rows.append(rr)

    corrected_targets: List[Dict[str, object]] = []
    for row in target_rows:
        rr = dict(row)
        base_est = _num(rr.get("Estimated_concentration"))
        rr["Base_estimated_concentration"] = base_est if base_est is not None else ""
        rr["Dynamic_range_method"] = selected
        if base_est is not None and base_est > 0:
            base_x = np.asarray([math.log10(float(base_est))], dtype=float)
            corr_log = float(_apply_calibrator(full_fit, base_x)[0])
            corr = 10.0 ** corr_log
            rr["Dynamic_corrected_score_log10"] = corr_log
            rr["Dynamic_corrected_concentration"] = corr
            # Compatibility with downstream dual-model evaluation.
            rr["Estimated_concentration"] = corr
            rr["Response_corrected_score_log10"] = corr_log
        corrected_targets.append(rr)
    _rank_target_rows(corrected_targets, "Dynamic_corrected_concentration")

    high_rows = _high_range_rows(actual_log, base_log, selected_pred_log)
    for row in comparison:
        row["Selected"] = str(row.get("Method")) == selected
        row["Requested_mode"] = requested
    return DynamicRangeResult(
        selected_method=selected,
        comparison_rows=comparison,
        cv_rows=corrected_cv_rows,
        target_rows=corrected_targets,
        high_range_bias_rows=high_rows,
        fit_info=full_fit,
        warnings=warnings,
    )


def literature_context_rows() -> List[Dict[str, object]]:
    return [
        {
            "Reference_framework": "Negative-ion RandFor-IE literature",
            "Scope": "Large multi-condition negative-ESI ionization-efficiency modelling",
            "Key_inputs": "Molecular descriptors plus viscosity, surface tension, polarity index, aqueous pH and ammonium presence",
            "Use_in_this_software": "Local literature-aligned reconstruction with transparent RDKit and negative-ion proxy descriptors",
            "Important_limit": "This package does not contain the authors' original fitted negative-ion weights or their complete proprietary descriptor matrix",
        },
        {
            "Reference_framework": "Negative-ion mechanistic IE models",
            "Scope": "Acidity, solution ionization and anion charge delocalization",
            "Key_inputs": "Acidic-site class, pH relation, hydrophobicity and charge-delocalization descriptors",
            "Use_in_this_software": "Transparent proxy features labelled with _proxy",
            "Important_limit": "Gasteiger/SMARTS proxies are not experimental pKa, COSMO-RS alpha, or WAPS values",
        },
        {
            "Reference_framework": "Dynamic-range correction",
            "Scope": "Empirical correction of cross-compound response compression at the high end",
            "Key_inputs": "Fold-external base concentration estimates and known standard concentrations",
            "Use_in_this_software": "Cross-fitted robust linear, isotonic, or piecewise monotonic calibration",
            "Important_limit": "One concentration per analyte cannot prove analyte-specific saturation; multi-level curves are required for that claim",
        },
    ]


def make_negative_esi_plots(
    output_xlsx,
    result: DynamicRangeResult,
    feature_rows: Sequence[Dict[str, object]],
) -> Tuple[List[object], List[str]]:
    """Create compact English diagnostic plots for the negative-ESI branch."""
    from pathlib import Path
    import hashlib
    warnings: List[str] = []
    paths: List[Path] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], [f"Negative-ESI plots were skipped: {exc}"]

    out = Path(output_xlsx)
    tag = hashlib.sha1(str(out).encode("utf-8", errors="ignore")).hexdigest()[:8]
    folder = out.parent / f"_negplots_{tag}"
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str):
        path = folder / name
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    try:
        rows = [r for r in result.comparison_rows if _num(r.get("Median_Fold_Error")) is not None]
        if rows:
            names = [str(r.get("Method")) for r in rows]
            med = [float(r.get("Median_Fold_Error")) for r in rows]
            p80 = [float(r.get("P80_Fold_Error")) for r in rows]
            x = np.arange(len(names))
            width = 0.38
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(x - width / 2, med, width, label="Median fold error")
            ax.bar(x + width / 2, p80, width, label="P80 fold error")
            ax.set_xticks(x, names, rotation=25, ha="right")
            ax.set_ylabel("Fold error")
            ax.set_title("Cross-fitted dynamic-range calibration")
            ax.legend()
            save(fig, "01_dynamic_methods.png")
    except Exception as exc:
        warnings.append(f"Dynamic-method plot skipped: {exc}")

    try:
        rows = result.high_range_bias_rows
        if rows:
            quartiles = ["Q1_low", "Q2", "Q3", "Q4_high"]
            base = []
            corr = []
            for q in quartiles:
                br = next((r for r in rows if r.get("Concentration_quartile") == q and r.get("Stage") == "Base"), {})
                cr = next((r for r in rows if r.get("Concentration_quartile") == q and r.get("Stage") == "Dynamic_corrected"), {})
                base.append(float(_num(br.get("Median_signed_log10_error")) or 0.0))
                corr.append(float(_num(cr.get("Median_signed_log10_error")) or 0.0))
            x = np.arange(len(quartiles))
            width = 0.38
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(x - width / 2, base, width, label="Base model")
            ax.bar(x + width / 2, corr, width, label="Dynamic corrected")
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_xticks(x, quartiles)
            ax.set_ylabel("Median log10(predicted / actual)")
            ax.set_title("Concentration-range bias")
            ax.legend()
            save(fig, "02_high_range_bias.png")
    except Exception as exc:
        warnings.append(f"High-range plot skipped: {exc}")

    try:
        rows = result.cv_rows
        actual = [float(r["Actual_concentration"]) for r in rows if _num(r.get("Actual_concentration")) and _num(r.get("Predicted_concentration"))]
        pred = [float(r["Predicted_concentration"]) for r in rows if _num(r.get("Actual_concentration")) and _num(r.get("Predicted_concentration"))]
        if actual and pred:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(actual, pred, alpha=0.65)
            lo = min(min(actual), min(pred))
            hi = max(max(actual), max(pred))
            ax.plot([lo, hi], [lo, hi], linestyle="-")
            ax.plot([lo, hi], [2 * lo, 2 * hi], linestyle=":")
            ax.plot([lo, hi], [lo / 2, hi / 2], linestyle=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Actual standard concentration")
            ax.set_ylabel("Fold-external predicted concentration")
            ax.set_title(f"Negative-ESI framework ({result.selected_method})")
            save(fig, "03_actual_vs_predicted.png")
    except Exception as exc:
        warnings.append(f"Actual-vs-predicted plot skipped: {exc}")

    try:
        selected = [r for r in feature_rows if str(r.get("Status")) == "Selected"]
        selected.sort(key=lambda r: float(_num(r.get("Full_fit_importance")) or 0.0), reverse=True)
        selected = selected[:20]
        if selected:
            names = [str(r.get("Feature")) for r in selected][::-1]
            vals = [float(_num(r.get("Full_fit_importance")) or 0.0) for r in selected][::-1]
            fig, ax = plt.subplots(figsize=(9, max(4.5, 0.3 * len(names))))
            ax.barh(names, vals)
            ax.set_xlabel("Random-forest feature importance")
            ax.set_title("Negative-ESI literature-aligned feature importance")
            save(fig, "04_negative_feature_importance.png")
    except Exception as exc:
        warnings.append(f"Feature-importance plot skipped: {exc}")

    return paths, warnings
