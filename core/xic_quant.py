
from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fisher_reader import extract_averaged_spectrum_from_raw
from .chemistry import (
    MASS_MH_TO_FORMATE,
    NEG_ADDUCTS,
    POS_ADDUCTS,
    calc_mz,
    compute_neutral_mass_from_formula,
    guess_polarity_from_filter_string,
    normalize_polarity,
)
from .spectrum_processing import pick_peaks_by_mz_bins
from .negative_ion_channels import (
    CHANNEL_BY_ID,
    PUBLIC_CHANNEL_IDS,
    IonChannelDefinition,
    NegativeIonPanel,
    PanelChannel,
    channel_mz,
    default_discovery_panel,
    is_channel_eligible,
    load_panel,
    normalise_action,
)
from .target_matching import MatchRow, match_targets_to_spectrum
from .targets_csv import TargetSpec
from .utils import ensure_dir, safe_stem_from_raw_path

# ==============
# Small helpers
# ==============


def _safe_filename(s: str, max_len: int = 120) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_\-\.\(\)\[\]]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "item"
    return s[:max_len]


def _extract_combo_from_raw_row(raw_row: Optional[Dict[str, str]]) -> str:

    if not raw_row:
        return ""
    for k in (
        "combo",
        "abc_combo",
        "source",
        "origin",
        "from",
        "components",
        'Combo',
        'Source',
    ):
        v = raw_row.get(k, "")
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _clear_png_files(d: Path) -> None:

    try:
        d = Path(d)
        if not d.exists() or not d.is_dir():
            return
        for p in d.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass

    # NOTE:
    # This helper function should ONLY clear old PNGs.
    # A previous refactor mistakenly left plotting-related variables here
    # (mz/inten/min_rel), which can raise: NameError: name 'mz' is not defined.
    return


def _ppm_error(mz_obs: float, mz_ref: float) -> float:
    try:
        mz_obs = float(mz_obs)
        mz_ref = float(mz_ref)
        if not np.isfinite(mz_obs) or not np.isfinite(mz_ref) or mz_ref == 0:
            return float("nan")
        return (mz_obs - mz_ref) / mz_ref * 1e6
    except Exception:
        return float("nan")


def _get_chromatogram_massrange(raw, *, mz: float, ppm: float, ms_filter: str) -> Tuple[np.ndarray, np.ndarray, str]:
    from fisher_py.data.business import TraceType

    ms_filter = (ms_filter or "ms").lower().strip()
    candidates = [ms_filter] if ms_filter in ("ms", "ms1", "ms2") else ["ms", "ms1", "ms2"]

    last_err: Optional[Exception] = None
    for f in candidates:
        try:
            
            rt, inten = raw.get_chromatogram(float(mz), float(ppm), TraceType.MassRange, ms_filter=f)
            rt = np.asarray(rt, dtype=float)
            inten = np.asarray(inten, dtype=float)
            return rt, inten, f
        except TypeError as e:
            
            last_err = e
            rt, inten = raw.get_chromatogram(float(mz), float(ppm), TraceType.MassRange)
            return np.asarray(rt, dtype=float), np.asarray(inten, dtype=float), ""
        except Exception as e:
            last_err = e
            continue

    
    _ = last_err
    return np.asarray([], dtype=float), np.asarray([], dtype=float), ""


@dataclass
class PeakResult:
    apex_rt: float
    area: float
    rt_start: float
    rt_end: float
    peak_height: float
    baseline: float
    raw_apex: float




def _robust_noise_sigma(intensity: np.ndarray) -> float:
    """Estimate chromatographic high-frequency noise with a robust first-difference MAD.

    The value is only a diagnostic/SNR proxy; it is not claimed to be an instrument S/N.
    First differences reduce sensitivity to a slowly changing baseline, which is important
    for complex 1000-component samples where the background can be higher than in standards.
    """
    y = np.asarray(intensity, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return 0.0
    d = np.diff(y)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return 0.0
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sigma = mad / 0.6744897501960817 / np.sqrt(2.0) if mad > 0 else 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        q = float(np.quantile(y, 0.20))
        lower = y[y <= float(np.quantile(y, 0.60))]
        if lower.size >= 3:
            mad2 = float(np.median(np.abs(lower - float(np.median(lower)))))
            sigma = mad2 / 0.6744897501960817 if mad2 > 0 else 0.0
    return float(sigma if np.isfinite(sigma) and sigma > 0 else 0.0)


def _select_internal_standard_peak(
    rt: np.ndarray,
    intensity: np.ndarray,
    *,
    global_min_peak_height: float,
    edge_ratio: float,
    baseline_quantile: float,
    adaptive: bool,
    min_snr_proxy: float,
    min_fraction_of_global_height: float,
    expected_rt: Optional[float] = None,
    expected_rt_tolerance_min: float = 0.30,
) -> Tuple[Optional[PeakResult], Optional[PeakResult], Dict[str, object]]:
    """Select an internal-standard peak and retain a diagnostic candidate even if rejected.

    Normal targets still use the global minimum peak height.  For the internal standard,
    the software can additionally accept a clearly visible peak in a higher-background
    sample when its robust S/N proxy is adequate.  This avoids silently losing the same
    internal standard in the complex sample while retaining a transparent audit trail.
    """
    rt_all = np.asarray(rt, dtype=float)
    y_all = np.asarray(intensity, dtype=float)
    diag: Dict[str, object] = {
        "IS_Diagnostic_Status": "NO_TRACE",
        "IS_Accepted_By": "",
        "IS_Candidate_Apex_RT": "",
        "IS_Candidate_RT_Start": "",
        "IS_Candidate_RT_End": "",
        "IS_Candidate_Area": "",
        "IS_Candidate_Raw_Apex": "",
        "IS_Candidate_Baseline": "",
        "IS_Candidate_Net_Height": "",
        "IS_Noise_Sigma_Proxy": "",
        "IS_SNR_Proxy": "",
        "IS_Global_Min_Peak_Height": float(global_min_peak_height),
        "IS_Adaptive_Min_Height": "",
        "IS_Expected_RT": expected_rt if expected_rt is not None else "",
        "IS_Expected_RT_Tolerance_Min": float(expected_rt_tolerance_min),
        "IS_Diagnostic_Note": "empty chromatogram",
    }
    if rt_all.size == 0 or y_all.size == 0 or rt_all.size != y_all.size:
        return None, None, diag

    mask = np.isfinite(rt_all) & np.isfinite(y_all)
    rt_all = rt_all[mask]
    y_all = y_all[mask]
    if rt_all.size < 2:
        diag["IS_Diagnostic_Note"] = "chromatogram has fewer than two finite points"
        return None, None, diag

    # If a reference RT is supplied, choose within that window.  This is useful because
    # the same internal standard is present in both the 100-standard runs and the
    # 1000-component sample.  Otherwise use the strongest chromatographic component.
    rt_use = rt_all
    y_use = y_all
    expected_used = False
    try:
        er = float(expected_rt) if expected_rt is not None and str(expected_rt).strip() != "" else float("nan")
    except Exception:
        er = float("nan")
    if np.isfinite(er):
        tol = max(0.001, float(expected_rt_tolerance_min))
        m = (rt_all >= er - tol) & (rt_all <= er + tol)
        if int(np.count_nonzero(m)) >= 2:
            rt_use = rt_all[m]
            y_use = y_all[m]
            expected_used = True
        else:
            diag["IS_Diagnostic_Note"] = "no chromatogram points in expected-RT window; inspected full trace"

    candidate = integrate_largest_peak(
        rt_use,
        y_use,
        min_peak_height=0.0,
        edge_ratio=float(edge_ratio),
        baseline_quantile=float(baseline_quantile),
    )
    if candidate is None:
        diag["IS_Diagnostic_Status"] = "NO_CANDIDATE"
        diag["IS_Diagnostic_Note"] = "no integrable chromatographic component"
        return None, None, diag

    noise = _robust_noise_sigma(y_use)
    snr = float(candidate.peak_height / noise) if noise > 0 else (float("inf") if candidate.peak_height > 0 else 0.0)
    min_fraction = max(0.0, min(1.0, float(min_fraction_of_global_height)))
    adaptive_floor = max(float(global_min_peak_height) * min_fraction, float(min_snr_proxy) * noise)
    absolute_ok = bool(candidate.peak_height >= float(global_min_peak_height))
    adaptive_ok = bool(
        adaptive
        and candidate.peak_height > 0
        and candidate.peak_height >= adaptive_floor
        and snr >= float(min_snr_proxy)
    )

    diag.update({
        "IS_Candidate_Apex_RT": float(candidate.apex_rt),
        "IS_Candidate_RT_Start": float(candidate.rt_start),
        "IS_Candidate_RT_End": float(candidate.rt_end),
        "IS_Candidate_Area": float(candidate.area),
        "IS_Candidate_Raw_Apex": float(candidate.raw_apex),
        "IS_Candidate_Baseline": float(candidate.baseline),
        "IS_Candidate_Net_Height": float(candidate.peak_height),
        "IS_Noise_Sigma_Proxy": float(noise),
        "IS_SNR_Proxy": float(snr) if np.isfinite(snr) else "inf",
        "IS_Adaptive_Min_Height": float(adaptive_floor),
    })
    rt_note = "expected RT window" if expected_used else "full trace"
    if absolute_ok:
        diag["IS_Diagnostic_Status"] = "ACCEPTED"
        diag["IS_Accepted_By"] = "global_min_peak_height"
        diag["IS_Diagnostic_Note"] = f"accepted in {rt_note}; passed global minimum peak height"
        return candidate, candidate, diag
    if adaptive_ok:
        diag["IS_Diagnostic_Status"] = "ACCEPTED_ADAPTIVE"
        diag["IS_Accepted_By"] = "adaptive_background_SNR"
        diag["IS_Diagnostic_Note"] = (
            f"accepted in {rt_note}; below global height but passed adaptive background/SNR rule"
        )
        return candidate, candidate, diag

    diag["IS_Diagnostic_Status"] = "REJECTED"
    diag["IS_Accepted_By"] = ""
    diag["IS_Diagnostic_Note"] = (
        f"candidate retained for review ({rt_note}) but rejected: net height "
        f"{candidate.peak_height:.6g} < global {float(global_min_peak_height):.6g} and "
        f"adaptive floor {adaptive_floor:.6g}; SNR proxy={snr:.3g}"
    )
    return None, candidate, diag


def _override_internal_standard_matches(
    spec,
    targets: Sequence[TargetSpec],
    match_rows: Sequence[MatchRow],
    *,
    ppm: float,
    global_adduct_mode: str,
    global_forced_adduct: str,
    internal_standard_adduct: str,
) -> List[MatchRow]:
    """Use a deterministic ion form for the shared internal standard.

    Previously the internal standard used the same spectrum-driven adduct search as all
    analytes.  In a dense 1000-component spectrum, an unrelated peak could make the same
    formula select a different ion form from the clean 100-standard run.  This helper
    keeps the internal-standard ion form stable across files.
    """
    out = list(match_rows)
    raw_pol = normalize_polarity(guess_polarity_from_filter_string(getattr(spec, "filter_string", "") or ""))
    mode = str(global_adduct_mode or "").strip().lower()
    requested = str(internal_standard_adduct or "auto_raw").strip()

    for idx, target in enumerate(targets):
        raw_meta = getattr(target, "raw_row", None) or {}
        if not _truthy_flag(raw_meta.get("is_internal_standard", "")):
            continue
        if not str(target.formula or "").strip() and target.theoretical_mz is None:
            continue

        adduct_token = requested
        low = requested.lower().replace(" ", "")
        if low in {"", "auto", "auto_raw", "auto_from_raw", "raw"}:
            if mode == "force" and str(global_forced_adduct or "").strip():
                adduct_token = str(global_forced_adduct).strip()
            else:
                pol = raw_pol
                if not pol:
                    if mode in {"neg_all", "negative", "neg"}:
                        pol = "negative"
                    elif mode in {"pos_all", "positive", "pos"}:
                        pol = "positive"
                    elif idx < len(out):
                        pol = normalize_polarity(getattr(out[idx], "polarity", "") or "")
                adduct_token = "M-H" if pol == "negative" else "M+H"

        adduct_upper = adduct_token.upper().replace(" ", "")
        pol = "negative" if ("M-H" in adduct_upper or "HCOO" in adduct_upper or "FA-H" in adduct_upper) else "positive"
        fixed_target = TargetSpec(
            file_key=target.file_key,
            name=target.name,
            formula=target.formula,
            adduct=adduct_token,
            polarity=pol,
            theoretical_mz=target.theoretical_mz,
            ppm=target.ppm,
            raw_row=dict(raw_meta),
        )
        fixed = match_targets_to_spectrum(
            spec,
            [fixed_target],
            default_ppm=float(ppm),
            default_polarity=pol,
            min_rel_for_found=0.0,
            adduct_mode="auto",
            forced_adduct="",
            prefer_proton=False,
        )[0]
        note = str(fixed.note or "").strip()
        fixed.note = (note + "; " if note else "") + f"internal-standard ion fixed to {fixed.adduct}"
        out[idx] = fixed
    return out


def _apply_internal_standard_diag(row: Dict[str, object], diag: Dict[str, object], candidate: Optional[PeakResult]) -> None:
    for key, value in diag.items():
        row[key] = value
    row["_is_candidate_peak"] = candidate


def save_internal_standard_diagnostic_png(
    rt: np.ndarray,
    intensity: np.ndarray,
    *,
    accepted_peak: Optional[PeakResult],
    candidate_peak: Optional[PeakResult],
    diag: Dict[str, object],
    out_png: Path,
    formula: str,
    adduct: str,
    mz: float,
    ppm: float,
) -> None:
    """Always save an internal-standard chromatogram, including rejected candidates."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rt = np.asarray(rt, dtype=float)
    y = np.asarray(intensity, dtype=float)
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    if rt.size and y.size and rt.size == y.size:
        ax.plot(rt, y, linewidth=1.0, label="Internal-standard XIC")
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Internal standard XIC: {formula or 'formula unavailable'}")

    baseline = _num_for_diag(diag.get("IS_Candidate_Baseline"))
    global_min = _num_for_diag(diag.get("IS_Global_Min_Peak_Height"))
    adaptive_min = _num_for_diag(diag.get("IS_Adaptive_Min_Height"))
    if baseline is not None:
        ax.axhline(baseline, linestyle=":", linewidth=0.9, label="Estimated baseline")
        if global_min is not None:
            ax.axhline(baseline + global_min, linestyle="--", linewidth=0.8, label="Global height threshold")
        if adaptive_min is not None:
            ax.axhline(baseline + adaptive_min, linestyle="-.", linewidth=0.8, label="Adaptive IS threshold")

    p = accepted_peak if accepted_peak is not None else candidate_peak
    if p is not None:
        ax.axvline(float(p.apex_rt), linestyle="--", linewidth=0.9, label="Candidate apex")
        ax.axvspan(
            float(p.rt_start), float(p.rt_end),
            alpha=(0.16 if accepted_peak is not None else 0.08),
            label=("Accepted integration window" if accepted_peak is not None else "Rejected candidate window"),
        )
        if rt.size and y.size:
            try:
                ai = int(np.argmin(np.abs(rt - float(p.apex_rt))))
                ax.scatter([float(rt[ai])], [float(y[ai])], s=24)
            except Exception:
                pass

    status = str(diag.get("IS_Diagnostic_Status", ""))
    note = str(diag.get("IS_Diagnostic_Note", ""))
    snr = diag.get("IS_SNR_Proxy", "")
    net = diag.get("IS_Candidate_Net_Height", "")
    raw_apex = diag.get("IS_Candidate_Raw_Apex", "")
    apex_rt = diag.get("IS_Candidate_Apex_RT", "")
    text = (
        f"Ion={adduct}; m/z={float(mz):.5f}; ppm=±{float(ppm):g}\n"
        f"Status={status}; accepted_by={diag.get('IS_Accepted_By', '')}\n"
        f"candidate RT={apex_rt}; raw apex={raw_apex}; baseline={diag.get('IS_Candidate_Baseline', '')}\n"
        f"net height={net}; noise sigma proxy={diag.get('IS_Noise_Sigma_Proxy', '')}; SNR proxy={snr}\n"
        f"{note}"
    )
    ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
    if rt.size and y.size:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
    else:
        ax.text(0.5, 0.45, "NO CHROMATOGRAM RETURNED", transform=ax.transAxes, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _num_for_diag(value: object) -> Optional[float]:
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except Exception:
        return None

@dataclass
class DualAdductPeakResult:
    """One chromatographic component represented by one or two negative-ion channels."""

    combined: PeakResult
    mh_peak: Optional[PeakResult]
    formate_peak: Optional[PeakResult]
    formate_accepted: bool
    formate_rt_delta_min: float = float("nan")
    formate_shape_correlation: float = float("nan")
    channels_used: str = "[M-H]-"
    formate_note: str = ""


def integrate_largest_peak(
    rt: np.ndarray,
    inten: np.ndarray,
    *,
    min_peak_height: float = 1e4,
    edge_ratio: float = 0.01,
    baseline_quantile: float = 0.10,
) -> Optional[PeakResult]:

    if rt.size == 0 or inten.size == 0:
        return None

    rt = rt.astype(float)
    y_raw = inten.astype(float)

    baseline = float(np.quantile(y_raw, float(baseline_quantile)))
    y = y_raw - baseline
    y[y < 0] = 0.0

    apex_idx = int(np.argmax(y))
    peak_height = float(y[apex_idx])
    raw_apex = float(y_raw[apex_idx])

    if peak_height < float(min_peak_height):
        return None

    edge_th = peak_height * float(edge_ratio)
    left = apex_idx
    while left > 0 and y[left] > edge_th:
        left -= 1
    right = apex_idx
    while right < y.size - 1 and y[right] > edge_th:
        right += 1

    rt_slice = rt[left : right + 1]
    y_slice = y[left : right + 1]
    if rt_slice.size < 2:
        return None

    area = float(np.trapz(y_slice, rt_slice))
    return PeakResult(
        apex_rt=float(rt[apex_idx]),
        area=area,
        rt_start=float(rt[left]),
        rt_end=float(rt[right]),
        peak_height=peak_height,
        baseline=baseline,
        raw_apex=raw_apex,
    )


def integrate_top_peaks(
    rt: np.ndarray,
    inten: np.ndarray,
    *,
    n_peaks: int,
    min_peak_height: float = 1e4,
    edge_ratio: float = 0.01,
    baseline_quantile: float = 0.10,
    # NOTE: defaults are tuned for "close but real" multiple peaks.
    # - Do NOT rely on a large minimum RT distance (it will incorrectly remove close peaks).
    # - Use a stricter valley merge (valley must be VERY shallow to merge),
    #   so real close peaks with a visible valley are preserved.
    min_peak_distance_min: float = 0.0,
    smooth_points: int = 3,
    min_prominence_ratio: float = 0.05,
    merge_valley_ratio: float = 0.92,
    min_rel_height: float = 0.005,
) -> List[PeakResult]:

    if rt.size == 0 or inten.size == 0 or int(n_peaks) <= 0:
        return []

    rt = rt.astype(float)
    y_raw = inten.astype(float)

    baseline = float(np.quantile(y_raw, float(baseline_quantile)))
    y = y_raw - baseline
    y[y < 0] = 0.0
    if y.size < 3:
        p = integrate_largest_peak(
            rt,
            inten,
            min_peak_height=float(min_peak_height),
            edge_ratio=float(edge_ratio),
            baseline_quantile=float(baseline_quantile),
        )
        return [p] if p is not None else []

    
    sp = int(max(1, smooth_points))
    if sp % 2 == 0:
        sp += 1
    if sp > 1:
        ker = np.ones(sp, dtype=float) / float(sp)
        y_s = np.convolve(y, ker, mode="same")
    else:
        y_s = y

    # candidate apex: y_s[i-1] < y_s[i] >= y_s[i+1]
    cand: List[int] = []
    for i in range(1, int(y_s.size) - 1):
        if y_s[i] >= y_s[i - 1] and y_s[i] > y_s[i + 1]:
            if float(y[i]) >= float(min_peak_height):
                cand.append(i)

    if not cand:
        
        p = integrate_largest_peak(
            rt,
            inten,
            min_peak_height=float(min_peak_height),
            edge_ratio=float(edge_ratio),
            baseline_quantile=float(baseline_quantile),
        )
        return [p] if p is not None else []

    # -------------------------
    # Peak selection (robust):
    # - add a simple prominence filter (avoid splitting one broad peak into many pseudo-peaks)
    # - enforce minimum RT distance
    # - merge adjacent peaks if the valley between them is too shallow
    # - integrate using non-overlapping valley boundaries
    # -------------------------

    def _simple_prominence(yv: np.ndarray, idx: int) -> float:
        """Approximate peak prominence without SciPy.

        We look for the minimum (valley) on left/right until a higher point appears.
        This suppresses small ripples riding on top of a big peak.
        """

        try:
            idx = int(idx)
        except Exception:
            return 0.0
        if idx <= 0 or idx >= int(yv.size) - 1:
            return 0.0
        peak_h = float(yv[idx])
        if not np.isfinite(peak_h):
            return 0.0

        # left base
        left_min = peak_h
        j = idx
        while j > 0:
            j -= 1
            v = float(yv[j])
            if np.isfinite(v):
                left_min = min(left_min, v)
            if float(yv[j]) > peak_h:
                break

        # right base
        right_min = peak_h
        j = idx
        while j < int(yv.size) - 1:
            j += 1
            v = float(yv[j])
            if np.isfinite(v):
                right_min = min(right_min, v)
            if float(yv[j]) > peak_h:
                break

        base = max(left_min, right_min)
        prom = peak_h - float(base)
        if not np.isfinite(prom):
            return 0.0
        return float(max(0.0, prom))

    # filter by prominence ratio
    prom_ratio = float(max(0.0, min_prominence_ratio))
    cand2: List[int] = []
    for idx in cand:
        h = float(y_s[idx])
        if h <= 0 or not np.isfinite(h):
            continue
        prom = _simple_prominence(y_s, int(idx))
        if prom_ratio > 0:
            if prom < prom_ratio * h:
                continue
        cand2.append(int(idx))

    if not cand2:
        # fallback: still try the max peak
        p = integrate_largest_peak(
            rt,
            inten,
            min_peak_height=float(min_peak_height),
            edge_ratio=float(edge_ratio),
            baseline_quantile=float(baseline_quantile),
        )
        return [p] if p is not None else []

    # sort by smoothed height desc, then distance filter
    cand2.sort(key=lambda idx: float(y_s[idx]), reverse=True)
    selected: List[int] = []
    min_dist = float(max(0.0, min_peak_distance_min))
    for idx in cand2:
        if selected and min_dist > 0:
            rt_i = float(rt[int(idx)])
            if any(abs(rt_i - float(rt[int(j)])) < min_dist for j in selected):
                continue
        selected.append(int(idx))
        # keep some headroom; we'll later pick top areas
        if len(selected) >= int(n_peaks) * 8:
            break

    if not selected:
        return []

    # sort by RT for merging
    selected.sort(key=lambda idx: float(rt[int(idx)]))

    # merge adjacent peaks if valley is too shallow
    mv_ratio = float(max(0.0, merge_valley_ratio))
    if mv_ratio > 0 and len(selected) >= 2:
        i = 0
        while i < len(selected) - 1:
            a = int(selected[i])
            b = int(selected[i + 1])
            if b <= a + 1:
                i += 1
                continue
            seg = y_s[a : b + 1]
            if seg.size < 3:
                i += 1
                continue
            valley_idx = int(np.argmin(seg)) + a
            valley_y = float(y_s[valley_idx])
            ha = float(y_s[a])
            hb = float(y_s[b])
            small_h = float(min(ha, hb))
            if small_h <= 0 or not np.isfinite(valley_y):
                # degenerate -> keep higher
                keep = a if ha >= hb else b
                drop = b if keep == a else a
                # remove drop
                if drop == a:
                    selected.pop(i)
                else:
                    selected.pop(i + 1)
                i = max(i - 1, 0)
                continue

            # If valley is too high relative to the smaller peak, they are likely one broad peak.
            if valley_y > mv_ratio * small_h:
                keep = a if ha >= hb else b
                drop = b if keep == a else a
                if drop == a:
                    selected.pop(i)
                else:
                    selected.pop(i + 1)
                i = max(i - 1, 0)
                continue
            i += 1

    if not selected:
        return []

    # build non-overlapping boundaries using valleys between adjacent peaks
    selected.sort(key=lambda idx: float(rt[int(idx)]))
    valleys: List[int] = []
    for i in range(len(selected) - 1):
        a = int(selected[i])
        b = int(selected[i + 1])
        if b <= a + 1:
            valleys.append(int(a))
            continue
        seg = y_s[a : b + 1]
        v = int(np.argmin(seg)) + a
        valleys.append(int(v))

    # integrate each selected peak within its valley-bounded region
    peaks: List[PeakResult] = []
    for i, idx in enumerate(selected):
        idx = int(idx)
        # We'll refine the apex using the RAW (baseline-corrected) signal within the peak region.
        # This fixes the issue where Apex_RT is sometimes not exactly at the peak top
        # (because the candidate idx is from the smoothed curve).

        # region bounds
        reg_left = 0 if i == 0 else int(valleys[i - 1])
        reg_right = int(y.size) - 1 if i == len(selected) - 1 else int(valleys[i])
        reg_left = max(0, min(reg_left, idx))
        reg_right = min(int(y.size) - 1, max(reg_right, idx))

        # refine bounds using edge threshold, but do not cross valley boundaries
        # First determine a provisional apex (idx), then compute edge threshold from the
        # true apex height inside the bounded region.
        # Region for searching the true apex
        sL = int(max(0, min(reg_left, idx)))
        sR = int(min(int(y.size) - 1, max(reg_right, idx)))
        if sR <= sL:
            continue
        apex_idx = int(np.argmax(y[sL : sR + 1])) + sL
        peak_height = float(y[apex_idx])
        if peak_height < float(min_peak_height):
            continue

        edge_th = peak_height * float(edge_ratio)
        left = int(apex_idx)
        while left > reg_left and float(y[left]) > edge_th:
            left -= 1
        right = int(apex_idx)
        while right < reg_right and float(y[right]) > edge_th:
            right += 1

        rt_slice = rt[left : right + 1]
        y_slice = y[left : right + 1]
        if rt_slice.size < 2:
            continue

        area = float(np.trapz(y_slice, rt_slice))
        peaks.append(
            PeakResult(
                apex_rt=float(rt[apex_idx]),
                area=float(area),
                rt_start=float(rt[left]),
                rt_end=float(rt[right]),
                peak_height=float(peak_height),
                baseline=float(baseline),
                raw_apex=float(y_raw[apex_idx]),
            )
        )

    if not peaks:
        return []

    # filter out tiny "extra" peaks to avoid "hard forcing" multiple peaks inside one broad peak
    try:
        mh = float(max(float(p.peak_height) for p in peaks))
    except Exception:
        mh = 0.0
    rel_h = float(max(0.0, min_rel_height))
    if mh > 0 and rel_h > 0 and len(peaks) > 1:
        keep: List[PeakResult] = []
        for p in peaks:
            if float(p.peak_height) >= mh * rel_h:
                keep.append(p)
        # always keep at least the strongest peak
        if not keep:
            peaks.sort(key=lambda p: float(p.peak_height), reverse=True)
            keep = [peaks[0]]
        peaks = keep

    # pick top n_peaks by area, then sort by RT (stable assignment for duplicates)
    peaks.sort(key=lambda p: float(p.area), reverse=True)
    peaks = peaks[: int(n_peaks)]
    peaks.sort(key=lambda p: float(p.apex_rt))
    return peaks


def _multipeak_params_for_mode(mode: str) -> Dict[str, float]:
    """Return parameter overrides for multi-peak detection.

    Why this exists:
    - There is a real trade-off between "don't split one peak" and "don't miss close real peaks".
    - Different datasets/instruments can prefer different settings.

    Modes:
    - conservative: harder to split (reduces false multi-peaks)
    - balanced: default
    - sensitive: easier to split (keeps close real peaks)
    """

    m = str(mode or "").strip().lower()
    if m in ('conservative', 'strict', '保守'):
        return {
            "smooth_points": 7,
            "min_prominence_ratio": 0.08,
            "merge_valley_ratio": 0.85,
            "min_rel_height": 0.02,
            "min_peak_distance_min": 0.02,  # ~1.2 sec
        }
    if m in ('sensitive', 'loose', '敏感'):
        return {
            "smooth_points": 1,
            "min_prominence_ratio": 0.02,
            "merge_valley_ratio": 0.98,
            "min_rel_height": 0.0,
            "min_peak_distance_min": 0.0,
        }
    # balanced
    return {
        "smooth_points": 3,
        "min_prominence_ratio": 0.05,
        "merge_valley_ratio": 0.92,
        "min_rel_height": 0.005,
        "min_peak_distance_min": 0.0,
    }



def _normalise_percent_fraction(value: float, *, default: float = 0.0) -> float:
    """Accept either a fraction (0.01) or a percentage (1 = 1%)."""

    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    # The UI passes percent values. Values >1 are unambiguously percentages;
    # value==1 is also treated as 1%, which is the intuitive UI behaviour.
    if v >= 1.0:
        v = v / 100.0
    return float(max(0.0, min(1.0, v)))


def _align_trace(reference_rt: np.ndarray, source_rt: np.ndarray, source_intensity: np.ndarray) -> np.ndarray:
    """Interpolate a chromatogram onto another RT grid, using zero outside the source range."""

    ref = np.asarray(reference_rt, dtype=float)
    src_rt = np.asarray(source_rt, dtype=float)
    src_y = np.asarray(source_intensity, dtype=float)
    if ref.size == 0:
        return np.asarray([], dtype=float)
    if src_rt.size == 0 or src_y.size == 0 or src_rt.size != src_y.size:
        return np.zeros_like(ref, dtype=float)
    mask = np.isfinite(src_rt) & np.isfinite(src_y)
    if int(np.sum(mask)) < 2:
        return np.zeros_like(ref, dtype=float)
    x = src_rt[mask]
    y = src_y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # np.interp requires increasing x; collapse duplicate RT values by keeping the max intensity.
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size < 2:
        return np.zeros_like(ref, dtype=float)
    uy = np.zeros(ux.size, dtype=float)
    for i, val in enumerate(y):
        j = int(inv[i])
        if float(val) > float(uy[j]):
            uy[j] = float(val)
    return np.interp(ref, ux, uy, left=0.0, right=0.0)


def _baseline_correct_trace(intensity: np.ndarray, baseline_quantile: float) -> Tuple[np.ndarray, float]:
    y = np.asarray(intensity, dtype=float)
    if y.size == 0:
        return np.asarray([], dtype=float), 0.0
    finite = y[np.isfinite(y)]
    baseline = float(np.quantile(finite, float(baseline_quantile))) if finite.size else 0.0
    out = np.nan_to_num(y - baseline, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0] = 0.0
    return out, baseline


def _peak_shape_correlation(
    rt_mh: np.ndarray,
    int_mh: np.ndarray,
    rt_formate: np.ndarray,
    int_formate: np.ndarray,
    *,
    rt_start: float,
    rt_end: float,
    baseline_quantile: float,
) -> float:
    """Pearson correlation of the two baseline-corrected XIC shapes in a local RT window."""

    try:
        lo = float(min(rt_start, rt_end))
        hi = float(max(rt_start, rt_end))
    except Exception:
        return float("nan")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    rt_mh = np.asarray(rt_mh, dtype=float)
    int_mh = np.asarray(int_mh, dtype=float)
    rt_formate = np.asarray(rt_formate, dtype=float)
    int_formate = np.asarray(int_formate, dtype=float)
    if rt_mh.size < 3 or rt_formate.size < 3:
        return float("nan")

    ref = rt_mh[(rt_mh >= lo) & (rt_mh <= hi)]
    if ref.size < 5:
        # Use a modest common grid when the scan density is low.
        ref = np.linspace(lo, hi, 21, dtype=float)
    mh_corr, _ = _baseline_correct_trace(int_mh, baseline_quantile)
    fa_corr, _ = _baseline_correct_trace(int_formate, baseline_quantile)
    a = _align_trace(ref, rt_mh, mh_corr)
    b = _align_trace(ref, rt_formate, fa_corr)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 5 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return float("nan")


def _combine_channel_peaks(
    mh_peak: Optional[PeakResult],
    formate_peak: Optional[PeakResult],
    *,
    rt_mh: np.ndarray,
    int_mh: np.ndarray,
    rt_formate: np.ndarray,
    int_formate: np.ndarray,
    baseline_quantile: float,
) -> PeakResult:
    """Build one audit-friendly combined peak while preserving component areas."""

    if mh_peak is None and formate_peak is None:
        raise ValueError("At least one channel peak is required")
    if mh_peak is None:
        p = formate_peak
        assert p is not None
        return PeakResult(**p.__dict__)
    if formate_peak is None:
        return PeakResult(**mh_peak.__dict__)

    start = float(min(mh_peak.rt_start, formate_peak.rt_start))
    end = float(max(mh_peak.rt_end, formate_peak.rt_end))
    ref = np.asarray(rt_mh, dtype=float)
    if ref.size < 2:
        ref = np.asarray(rt_formate, dtype=float)
    mask = (ref >= start) & (ref <= end)
    local_rt = ref[mask]
    if local_rt.size < 2:
        local_rt = np.linspace(start, end, 21, dtype=float)

    mh_bc, mh_baseline = _baseline_correct_trace(np.asarray(int_mh, dtype=float), baseline_quantile)
    fa_bc, fa_baseline = _baseline_correct_trace(np.asarray(int_formate, dtype=float), baseline_quantile)
    mh_y = _align_trace(local_rt, np.asarray(rt_mh, dtype=float), mh_bc)
    fa_y = _align_trace(local_rt, np.asarray(rt_formate, dtype=float), fa_bc)
    total = mh_y + fa_y
    if total.size:
        ai = int(np.argmax(total))
        apex_rt = float(local_rt[ai])
        peak_height = float(total[ai])
        raw_mh = _align_trace(local_rt, np.asarray(rt_mh, dtype=float), np.asarray(int_mh, dtype=float))
        raw_fa = _align_trace(local_rt, np.asarray(rt_formate, dtype=float), np.asarray(int_formate, dtype=float))
        raw_apex = float(raw_mh[ai] + raw_fa[ai])
    else:
        apex_rt = float(mh_peak.apex_rt if mh_peak.peak_height >= formate_peak.peak_height else formate_peak.apex_rt)
        peak_height = float(mh_peak.peak_height + formate_peak.peak_height)
        raw_apex = float(mh_peak.raw_apex + formate_peak.raw_apex)

    return PeakResult(
        apex_rt=apex_rt,
        area=float(mh_peak.area + formate_peak.area),
        rt_start=start,
        rt_end=end,
        peak_height=peak_height,
        baseline=float(mh_baseline + fa_baseline),
        raw_apex=raw_apex,
    )


def build_dual_adduct_peaks(
    mh_peaks: Sequence[PeakResult],
    formate_peaks: Sequence[PeakResult],
    *,
    need: int,
    rt_mh: np.ndarray,
    int_mh: np.ndarray,
    rt_formate: np.ndarray,
    int_formate: np.ndarray,
    rt_tolerance_min: float = 0.10,
    min_formate_rel_height_pct: float = 1.0,
    min_shape_correlation: float = 0.30,
    baseline_quantile: float = 0.10,
    allow_formate_only: bool = False,
) -> List[DualAdductPeakResult]:
    """Pair [M-H]- and [M+HCOO]- peaks one-to-one and conditionally sum their areas.

    A formate peak is accepted only when it independently passes the ordinary peak-height
    detector, coelutes with the selected [M-H]- peak, and (when calculable) has a compatible
    chromatographic shape. This prevents adding the full formate XIC as noise.
    """

    need = max(0, int(need))
    if need <= 0:
        return []
    tol = float(max(0.0, rt_tolerance_min))
    try:
        rel_fraction = float(min_formate_rel_height_pct) / 100.0
    except Exception:
        rel_fraction = 0.01
    if not np.isfinite(rel_fraction):
        rel_fraction = 0.01
    rel_fraction = float(max(0.0, min(1.0, rel_fraction)))
    try:
        min_corr = float(min_shape_correlation)
    except Exception:
        min_corr = 0.30
    min_corr = float(max(-1.0, min(1.0, min_corr)))

    mh_list = sorted(list(mh_peaks), key=lambda p: float(p.apex_rt))
    fa_list = sorted(list(formate_peaks), key=lambda p: float(p.apex_rt))
    unused = set(range(len(fa_list)))
    paired: List[DualAdductPeakResult] = []

    for mh in mh_list:
        accepted: List[Tuple[Tuple[float, float, float], int, float, float]] = []
        rejected_notes: List[Tuple[float, str]] = []
        for j in sorted(unused):
            fa = fa_list[j]
            delta = abs(float(fa.apex_rt) - float(mh.apex_rt))
            if delta > tol:
                rejected_notes.append((delta, f"nearest formate apex outside RT tolerance ({delta:.4g} min)"))
                continue
            if rel_fraction > 0 and float(fa.peak_height) < float(mh.peak_height) * rel_fraction:
                rejected_notes.append((delta, f"formate height below {rel_fraction * 100:.3g}% of [M-H]-"))
                continue
            lo = min(float(mh.rt_start), float(fa.rt_start))
            hi = max(float(mh.rt_end), float(fa.rt_end))
            corr = _peak_shape_correlation(
                rt_mh,
                int_mh,
                rt_formate,
                int_formate,
                rt_start=lo,
                rt_end=hi,
                baseline_quantile=float(baseline_quantile),
            )
            if min_corr > -1.0:
                if not np.isfinite(corr):
                    rejected_notes.append((delta, "formate shape correlation could not be calculated"))
                    continue
                if corr < min_corr:
                    rejected_notes.append((delta, f"formate shape correlation {corr:.3f} below {min_corr:.3f}"))
                    continue
            corr_score = float(corr) if np.isfinite(corr) else 0.0
            accepted.append(((delta, -corr_score, -float(fa.area)), j, delta, corr))

        if accepted:
            accepted.sort(key=lambda x: x[0])
            _, j, delta, corr = accepted[0]
            unused.discard(int(j))
            fa = fa_list[int(j)]
            combined = _combine_channel_peaks(
                mh,
                fa,
                rt_mh=rt_mh,
                int_mh=int_mh,
                rt_formate=rt_formate,
                int_formate=int_formate,
                baseline_quantile=float(baseline_quantile),
            )
            paired.append(
                DualAdductPeakResult(
                    combined=combined,
                    mh_peak=mh,
                    formate_peak=fa,
                    formate_accepted=True,
                    formate_rt_delta_min=float(delta),
                    formate_shape_correlation=float(corr),
                    channels_used="[M-H]- + [M+HCOO]-",
                    formate_note="accepted: coeluting formate peak",
                )
            )
        else:
            note = "no formate peak above the ordinary peak-height threshold"
            if rejected_notes:
                rejected_notes.sort(key=lambda x: x[0])
                note = rejected_notes[0][1]
            paired.append(
                DualAdductPeakResult(
                    combined=_combine_channel_peaks(
                        mh,
                        None,
                        rt_mh=rt_mh,
                        int_mh=int_mh,
                        rt_formate=rt_formate,
                        int_formate=int_formate,
                        baseline_quantile=float(baseline_quantile),
                    ),
                    mh_peak=mh,
                    formate_peak=None,
                    formate_accepted=False,
                    channels_used="[M-H]-",
                    formate_note=note,
                )
            )

    # Conservative default: formate-only peaks are not used unless explicitly enabled.
    if bool(allow_formate_only) and len(paired) < need:
        remaining = [fa_list[j] for j in unused]
        remaining.sort(key=lambda p: float(p.area), reverse=True)
        for fa in remaining[: max(0, need - len(paired))]:
            paired.append(
                DualAdductPeakResult(
                    combined=_combine_channel_peaks(
                        None,
                        fa,
                        rt_mh=rt_mh,
                        int_mh=int_mh,
                        rt_formate=rt_formate,
                        int_formate=int_formate,
                        baseline_quantile=float(baseline_quantile),
                    ),
                    mh_peak=None,
                    formate_peak=fa,
                    formate_accepted=True,
                    channels_used="[M+HCOO]- only",
                    formate_note="accepted as formate-only because [M-H]- was not detected",
                )
            )

    # Keep the strongest required components and then restore RT order for duplicate assignment.
    if len(paired) > need:
        paired.sort(key=lambda d: float(d.combined.area), reverse=True)
        paired = paired[:need]
    paired.sort(key=lambda d: float(d.combined.apex_rt))
    return paired


def save_dual_xic_png(
    rt_mh: np.ndarray,
    int_mh: np.ndarray,
    rt_formate: np.ndarray,
    int_formate: np.ndarray,
    dual: Optional[DualAdductPeakResult],
    *,
    out_png: Path,
    title: str,
    subtitle: str,
    ratio_text: Optional[str] = None,
) -> None:
    """Plot both negative-ion channels so the conditional area sum can be audited visually."""

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 3.6))
    ax = plt.gca()
    ax.plot(rt_mh, int_mh, linewidth=1.0, label="[M-H]-")
    ax.plot(rt_formate, int_formate, linewidth=1.0, label="[M+HCOO]-")
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    if dual is not None:
        p = dual.combined
        ax.axvspan(p.rt_start, p.rt_end, alpha=0.12)
        ax.axvline(p.apex_rt, linestyle="--", linewidth=0.8)
        msg = subtitle
        mh_area = float(dual.mh_peak.area) if dual.mh_peak is not None else 0.0
        fa_area = float(dual.formate_peak.area) if dual.formate_peak is not None else 0.0
        msg += f"\n[M-H]- area={mh_area:.3g}; formate area={fa_area:.3g}; total={p.area:.3g}"
        msg += f"\nChannels used: {dual.channels_used}"
        if np.isfinite(float(dual.formate_shape_correlation)):
            msg += f"; shape r={float(dual.formate_shape_correlation):.3f}"
        if ratio_text:
            msg += f"; {ratio_text}"
        if dual.formate_note:
            msg += f"\n{dual.formate_note}"
        ax.text(0.01, 0.98, msg, transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
    else:
        ax.text(0.01, 0.98, subtitle + "\nNOT FOUND", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()


def save_xic_png(
    rt: np.ndarray,
    inten: np.ndarray,
    peak: Optional[PeakResult],
    *,
    out_png: Path,
    title: str,
    subtitle: str,
    ratio_text: Optional[str] = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10.5, 3.2))
    plt.plot(rt, inten, linewidth=1.0)
    plt.xlabel("RT (min)")
    plt.ylabel("Intensity")
    plt.title(title)

    if peak is not None:
        
        plt.axvspan(peak.rt_start, peak.rt_end, alpha=0.15)
        plt.axvline(peak.apex_rt, linestyle="--", linewidth=0.8)
        # Use nearest point (not interpolation) to ensure the marker is exactly on the apex sample.
        # This avoids the visual issue where Apex_RT looks slightly off the peak top.
        if rt.size:
            try:
                ai = int(np.argmin(np.abs(rt - float(peak.apex_rt))))
                apex_y = float(inten[ai])
            except Exception:
                apex_y = float(np.interp(float(peak.apex_rt), rt, inten))
        else:
            apex_y = 0.0
        plt.scatter([peak.apex_rt], [apex_y], s=18)

        msg = subtitle
        msg += f"\nArea={peak.area:.3g}"
        if ratio_text:
            msg += f", {ratio_text}"
        plt.text(0.01, 0.98, msg, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9)
    else:
        plt.text(0.01, 0.98, subtitle + "\nNOT FOUND", transform=plt.gca().transAxes, va="top", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()

def _primary_peak_from_row(row: Dict[str, object]) -> Optional[PeakResult]:
    dual = row.get("_dual_peak")
    if isinstance(dual, DualAdductPeakResult) and isinstance(dual.mh_peak, PeakResult):
        return dual.mh_peak
    peak = row.get("_peak")
    if isinstance(peak, PeakResult) and bool(row.get("MH_Found", False)):
        return peak
    return None


def _detect_channel_peaks(
    rt: np.ndarray,
    inten: np.ndarray,
    *,
    desired: int,
    min_peak_height: float,
    edge_ratio: float,
    baseline_quantile: float,
) -> List[PeakResult]:
    """Detect a generous candidate list without fabricating extra peaks."""
    best: List[PeakResult] = []
    for mode in ("balanced", "sensitive"):
        kw = _multipeak_params_for_mode(mode)
        peaks = integrate_top_peaks(
            rt,
            inten,
            n_peaks=max(1, int(desired)),
            min_peak_height=float(min_peak_height),
            edge_ratio=float(edge_ratio),
            baseline_quantile=float(baseline_quantile),
            min_peak_distance_min=float(kw.get("min_peak_distance_min", 0.0)),
            smooth_points=int(kw.get("smooth_points", 3)),
            min_prominence_ratio=float(kw.get("min_prominence_ratio", 0.05)),
            merge_valley_ratio=float(kw.get("merge_valley_ratio", 0.92)),
            min_rel_height=0.0,
        )
        if len(peaks) > len(best):
            best = peaks
        if len(best) >= int(desired):
            break
    return sorted(best, key=lambda p: float(p.apex_rt))


def _match_support_peak(
    candidate: PeakResult,
    support_peaks: Sequence[PeakResult],
    *,
    rt_tolerance_min: float,
) -> Optional[PeakResult]:
    choices = [
        p for p in support_peaks
        if abs(float(p.apex_rt) - float(candidate.apex_rt)) <= float(rt_tolerance_min)
    ]
    if not choices:
        return None
    choices.sort(key=lambda p: (abs(float(p.apex_rt) - float(candidate.apex_rt)), -float(p.area)))
    return choices[0]


def apply_negative_ion_panel(
    raw,
    rows: List[Dict[str, object]],
    *,
    panel: NegativeIonPanel,
    ppm: float,
    ms_filter: str,
    min_peak_height: float,
    edge_ratio: float,
    baseline_quantile: float,
    include_formate_evidence: bool = True,
    include_for_internal_standard: bool = True,
    internal_standard_ppm: Optional[float] = None,
    internal_standard_min_peak_height: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Search panel channels around each accepted [M-H]- chromatographic peak.

    The function is intentionally conservative:
    - [M-H]- is the RT/shape anchor by default;
    - every alternate channel must independently pass the absolute peak-height rule;
    - RT coelution, relative height and chromatographic-shape correlation are checked;
    - fragment/dimer/diagnostic channels remain evidence-only even if a panel row says ``sum``.
    """
    evidence: List[Dict[str, object]] = []
    panel_map = panel.by_id()

    # Preserve the area before additional channels for auditing and fragility features.
    for row in rows:
        row["Quant_Area_Before_Panel"] = float(row.get("Area", 0.0) or 0.0)
        row["Accepted_Channels"] = str(row.get("Adduct_Channels_Used", "") or "")
        row["Accepted_Channel_Count"] = 0
        row["Summed_Channel_Count"] = 0
        row["Additional_Summed_Area"] = 0.0
        row["Adduct_Cluster_Area"] = 0.0
        row["Fragment_Diagnostic_Area"] = 0.0
        row["Ion_Form_Diversity_Count"] = 1 if bool(row.get("MH_Found", False)) else 0
        row["Observed_Fragility_Index"] = 0.0
        row["Adduct_Cluster_Proneness_Index"] = 0.0
        row["Primary_Ion_Fraction"] = 1.0 if bool(row.get("MH_Found", False)) else 0.0
        row["Br_Fragment_Fraction"] = 0.0
        row["Negative_Ion_Panel"] = panel.panel_name
        row["_additional_traces"] = []

    # Existing formate branch is converted into the same evidence format and its
    # summation action is made panel-aware.
    formate_panel = panel_map.get("FORMATE")
    if include_formate_evidence and formate_panel is not None and formate_panel.enabled:
        formate_def = CHANNEL_BY_ID["FORMATE"]
        formate_action = normalise_action(formate_panel.quant_action, formate_def)
        for row in rows:
            if not bool(row.get("Dual_Adduct_Enabled", False)):
                continue
            mh_area = float(row.get("MH_Area", 0.0) or 0.0)
            fa_area = float(row.get("Formate_Area", 0.0) or 0.0)
            accepted = bool(row.get("Formate_Accepted", False))
            allow_sum_for_row = bool(include_for_internal_standard or not bool(row.get("Is_internal_standard", False)))
            shape = row.get("Formate_Shape_Correlation", "")
            delta = row.get("Formate_RT_Delta_min", "")
            # Legacy dual-channel code may already have summed formate.  Enforce the panel action.
            current = float(row.get("Area", 0.0) or 0.0)
            if accepted and fa_area > 0:
                if formate_action == "sum" and allow_sum_for_row:
                    desired = mh_area + fa_area if mh_area > 0 else fa_area
                else:
                    desired = mh_area if mh_area > 0 else 0.0
                row["Area"] = desired
                row["Adduct_Channels_Used"] = (
                    "[M-H]- + [M+HCOO]-" if formate_action == "sum" and allow_sum_for_row else "[M-H]-"
                )
                row["Accepted_Channels"] = row["Adduct_Channels_Used"]
                row["Accepted_Channel_Count"] = int(row.get("Accepted_Channel_Count", 0)) + 1
                row["Ion_Form_Diversity_Count"] = int(row.get("Ion_Form_Diversity_Count", 0)) + 1
                row["Adduct_Cluster_Area"] = float(row.get("Adduct_Cluster_Area", 0.0)) + fa_area
                if formate_action == "sum" and allow_sum_for_row:
                    row["Summed_Channel_Count"] = int(row.get("Summed_Channel_Count", 0)) + 1
                    row["Additional_Summed_Area"] = float(row.get("Additional_Summed_Area", 0.0)) + fa_area
                traces = row.get("_additional_traces")
                fa_peak = row.get("_formate_peak")
                if not isinstance(fa_peak, PeakResult):
                    dual_peak = row.get("_dual_peak")
                    fa_peak = dual_peak.formate_peak if isinstance(dual_peak, DualAdductPeakResult) else None
                if isinstance(traces, list) and isinstance(fa_peak, PeakResult):
                    traces.append((formate_def.display, row.get("_rt_formate", []), row.get("_inten_formate", []), fa_peak))
            evidence.append({
                "No": row.get("No", ""), "Name": row.get("Name", ""),
                "Formula": row.get("Formula", ""), "Combo": row.get("Combo", ""),
                "Channel_ID": "FORMATE", "Channel": formate_def.display,
                "Category": formate_def.category, "Quant_Action": formate_action,
                "Eligible": True, "Accepted": accepted,
                "High_Confidence": accepted,
                "Theoretical_mz": row.get("Formate_Theoretical_mz", ""),
                "XIC_mz": row.get("Formate_XIC_mz", ""),
                "Apex_RT": row.get("Formate_Apex_RT", ""),
                "RT_Delta_Min": delta, "Shape_Correlation": shape,
                "Peak_Height": row.get("Formate_Peak_height", ""),
                "Area": fa_area,
                "Area_Fraction_vs_MH_Pct": (fa_area / mh_area * 100.0 if mh_area > 0 else ""),
                "Isotope_Support_OK": "", "Support_Area": "", "Support_Ratio": "",
                "Summed_Into_Quant_Area": bool(accepted and formate_action == "sum" and allow_sum_for_row),
                "Reason": row.get("Formate_Note", ""),
            })

    # Process each additional channel once per unique formula and assign peaks one-to-one by RT.
    formula_groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        formula = str(row.get("Formula", "") or "").strip()
        if formula:
            formula_groups.setdefault(formula, []).append(idx)

    for channel_id, cfg in panel_map.items():
        if channel_id == "FORMATE" or not cfg.enabled or channel_id not in CHANNEL_BY_ID:
            continue
        defn = CHANNEL_BY_ID[channel_id]
        if defn.category == "isotope_support":
            continue
        action = normalise_action(cfg.quant_action, defn)
        if action == "exclude":
            continue

        support_def = CHANNEL_BY_ID.get(defn.support_channel_id) if defn.support_channel_id else None
        for formula, row_indices in formula_groups.items():
            formula_is_internal = bool(row_indices) and all(bool(rows[ii].get("Is_internal_standard", False)) for ii in row_indices)
            channel_ppm = float(internal_standard_ppm) if formula_is_internal and internal_standard_ppm is not None else float(ppm)
            channel_min_height = (
                float(internal_standard_min_peak_height)
                if formula_is_internal and internal_standard_min_peak_height is not None
                else float(min_peak_height)
            )
            eligible = is_channel_eligible(defn, formula)
            try:
                _, neutral_mass = compute_neutral_mass_from_formula(formula)
            except Exception:
                neutral_mass = float("nan")
            cand_mz = channel_mz(defn, neutral_mass, formula) if eligible and np.isfinite(neutral_mass) else float("nan")
            support_mz = (
                channel_mz(support_def, neutral_mass, formula)
                if support_def is not None and eligible and np.isfinite(neutral_mass)
                else float("nan")
            )

            if eligible and np.isfinite(cand_mz):
                rt_c, int_c, _ = _get_chromatogram_massrange(raw, mz=float(cand_mz), ppm=float(channel_ppm), ms_filter=ms_filter)
                cand_peaks = _detect_channel_peaks(
                    rt_c, int_c,
                    desired=max(8, len(row_indices) * 4),
                    min_peak_height=float(channel_min_height),
                    edge_ratio=float(edge_ratio),
                    baseline_quantile=float(baseline_quantile),
                )
            else:
                rt_c = np.asarray([], dtype=float); int_c = np.asarray([], dtype=float); cand_peaks = []

            if support_def is not None and np.isfinite(support_mz):
                rt_s, int_s, _ = _get_chromatogram_massrange(raw, mz=float(support_mz), ppm=float(channel_ppm), ms_filter=ms_filter)
                support_peaks = _detect_channel_peaks(
                    rt_s, int_s,
                    desired=max(8, len(row_indices) * 4),
                    min_peak_height=max(1.0, float(channel_min_height) * 0.05),
                    edge_ratio=float(edge_ratio),
                    baseline_quantile=float(baseline_quantile),
                )
            else:
                rt_s = np.asarray([], dtype=float); int_s = np.asarray([], dtype=float); support_peaks = []

            unused = set(range(len(cand_peaks)))
            ordered_rows = sorted(row_indices, key=lambda i: float(rows[i].get("MH_Apex_RT", float("inf")) or float("inf")))
            for row_idx in ordered_rows:
                row = rows[row_idx]
                primary = _primary_peak_from_row(row)
                mh_area = float(row.get("MH_Area", 0.0) or 0.0)
                reason = ""
                accepted = False
                high_conf = False
                selected: Optional[PeakResult] = None
                selected_j: Optional[int] = None
                shape_corr = float("nan")
                rt_delta = float("nan")
                support_peak: Optional[PeakResult] = None
                support_ok: object = ""
                support_ratio: object = ""

                if not eligible:
                    reason = f"requires element {defn.formula_element_required} in formula"
                elif not np.isfinite(cand_mz):
                    reason = "candidate m/z could not be calculated"
                elif primary is None and panel.require_primary_mh:
                    reason = "no accepted [M-H]- anchor peak"
                else:
                    anchor_rt = float(primary.apex_rt) if primary is not None else float(row.get("Apex_RT", float("nan")))
                    options = []
                    for j in sorted(unused):
                        p = cand_peaks[j]
                        d_rt = abs(float(p.apex_rt) - anchor_rt) if np.isfinite(anchor_rt) else 0.0
                        if d_rt > float(cfg.rt_tolerance_min):
                            continue
                        rel = (float(p.peak_height) / float(primary.peak_height) * 100.0) if primary is not None and primary.peak_height > 0 else 100.0
                        if rel < float(cfg.min_rel_height_pct):
                            continue
                        corr = _peak_shape_correlation(
                            np.asarray(row.get("_rt_mh", row.get("_rt", [])), dtype=float),
                            np.asarray(row.get("_inten_mh", row.get("_inten", [])), dtype=float),
                            rt_c, int_c,
                            rt_start=min(float(primary.rt_start), float(p.rt_start)) if primary is not None else float(p.rt_start),
                            rt_end=max(float(primary.rt_end), float(p.rt_end)) if primary is not None else float(p.rt_end),
                            baseline_quantile=float(baseline_quantile),
                        ) if primary is not None else float("nan")
                        if primary is not None and float(cfg.min_shape_correlation) > -1:
                            if not np.isfinite(corr) or corr < float(cfg.min_shape_correlation):
                                continue
                        options.append(((d_rt, -(corr if np.isfinite(corr) else 0.0), -float(p.area)), j, p, corr))
                    if options:
                        options.sort(key=lambda x: x[0])
                        _, selected_j, selected, shape_corr = options[0]
                        rt_delta = abs(float(selected.apex_rt) - anchor_rt) if np.isfinite(anchor_rt) else float("nan")
                        accepted = True
                        if support_def is not None:
                            support_peak = _match_support_peak(selected, support_peaks, rt_tolerance_min=float(cfg.rt_tolerance_min))
                            if support_peak is not None and support_peak.area > 0:
                                ratio = float(selected.area) / float(support_peak.area)
                                support_ratio = ratio
                                low = float(defn.support_ratio_low or 0.0)
                                high = float(defn.support_ratio_high or float("inf"))
                                support_ok = bool(low <= ratio <= high)
                            else:
                                support_ok = False
                            high_conf = bool(accepted and support_ok)
                        else:
                            high_conf = accepted
                        unused.discard(int(selected_j))
                    else:
                        reason = "no coeluting candidate passed height/RT/shape thresholds"

                area = float(selected.area) if selected is not None else 0.0
                support_area = float(support_peak.area) if support_peak is not None else 0.0
                # The isotope-support trace is diagnostic only.  Do not add 37Cl/81Br
                # support area to the quantitative channel area, because the primary
                # [M-H]- channel is also represented by its selected monoisotopic XIC.
                # Summing support isotopes here would bias chloride/bromide channels.
                area_for_channel = area
                can_sum = bool(
                    accepted and action == "sum" and defn.quantifiable and
                    (support_def is None or bool(support_ok)) and
                    (include_for_internal_standard or not bool(row.get("Is_internal_standard", False)))
                )
                if can_sum:
                    row["Area"] = float(row.get("Area", 0.0) or 0.0) + area_for_channel
                    row["Additional_Summed_Area"] = float(row.get("Additional_Summed_Area", 0.0)) + area_for_channel
                    row["Summed_Channel_Count"] = int(row.get("Summed_Channel_Count", 0)) + 1
                    used = str(row.get("Adduct_Channels_Used", "") or "")
                    row["Adduct_Channels_Used"] = (used + " + " + defn.display).strip(" +")
                if accepted:
                    row["Accepted_Channel_Count"] = int(row.get("Accepted_Channel_Count", 0)) + 1
                    row["Ion_Form_Diversity_Count"] = int(row.get("Ion_Form_Diversity_Count", 0)) + 1
                    accepted_names = str(row.get("Accepted_Channels", "") or "")
                    row["Accepted_Channels"] = (accepted_names + " | " + defn.display).strip(" |")
                    if defn.category in {"adduct", "cluster", "substitution", "dimer"}:
                        row["Adduct_Cluster_Area"] = float(row.get("Adduct_Cluster_Area", 0.0)) + area_for_channel
                    if defn.category in {"fragment", "transformation", "diagnostic"}:
                        row["Fragment_Diagnostic_Area"] = float(row.get("Fragment_Diagnostic_Area", 0.0)) + area_for_channel
                    if defn.channel_id in {"BR_LOSS_FROM_MH", "BR_ANION79"}:
                        row["Br_Fragment_Area"] = float(row.get("Br_Fragment_Area", 0.0) or 0.0) + area_for_channel
                    traces = row.get("_additional_traces")
                    if isinstance(traces, list):
                        traces.append((defn.display, rt_c, int_c, selected))

                evidence.append({
                    "No": row.get("No", ""), "Name": row.get("Name", ""),
                    "Formula": formula, "Combo": row.get("Combo", ""),
                    "Channel_ID": defn.channel_id, "Channel": defn.display,
                    "Category": defn.category, "Quant_Action": action,
                    "Eligible": eligible, "Accepted": accepted,
                    "High_Confidence": high_conf,
                    "Theoretical_mz": float(cand_mz) if np.isfinite(cand_mz) else "",
                    "XIC_mz": float(cand_mz) if np.isfinite(cand_mz) else "",
                    "Apex_RT": float(selected.apex_rt) if selected is not None else "",
                    "RT_Delta_Min": rt_delta if np.isfinite(rt_delta) else "",
                    "Shape_Correlation": shape_corr if np.isfinite(shape_corr) else "",
                    "Peak_Height": float(selected.peak_height) if selected is not None else 0.0,
                    "Area": area_for_channel,
                    "Area_Fraction_vs_MH_Pct": (area_for_channel / mh_area * 100.0 if mh_area > 0 else ""),
                    "Isotope_Support_OK": support_ok,
                    "Support_Area": support_area if support_peak is not None else "",
                    "Support_Ratio": support_ratio,
                    "Summed_Into_Quant_Area": can_sum,
                    "Reason": reason or ("accepted" if accepted else "not detected"),
                })

    for row in rows:
        mh = float(row.get("MH_Area", 0.0) or 0.0)
        adduct_cluster = float(row.get("Adduct_Cluster_Area", 0.0) or 0.0)
        fragment = float(row.get("Fragment_Diagnostic_Area", 0.0) or 0.0)
        br_fragment = float(row.get("Br_Fragment_Area", 0.0) or 0.0)
        total_evidence = mh + adduct_cluster + fragment
        row["Observed_Fragility_Index"] = fragment / total_evidence if total_evidence > 0 else 0.0
        row["Adduct_Cluster_Proneness_Index"] = adduct_cluster / total_evidence if total_evidence > 0 else 0.0
        row["Primary_Ion_Fraction"] = mh / total_evidence if total_evidence > 0 else 0.0
        row["Br_Fragment_Fraction"] = br_fragment / total_evidence if total_evidence > 0 else 0.0
        row["Quant_Area_After_Panel"] = float(row.get("Area", 0.0) or 0.0)
    return evidence


def save_multi_channel_xic_png(
    row: Dict[str, object],
    *,
    out_png: Path,
    title: str,
    subtitle: str,
    ratio_text: str = "",
) -> None:
    """Plot primary XIC plus accepted additional ion channels (top area channels only)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    rt = np.asarray(row.get("_rt_mh", row.get("_rt", [])), dtype=float)
    yy = np.asarray(row.get("_inten_mh", row.get("_inten", [])), dtype=float)
    if rt.size and yy.size:
        ax.plot(rt, yy, linewidth=1.2, label="[M-H]-")
    traces = row.get("_additional_traces")
    items = list(traces) if isinstance(traces, list) else []
    items.sort(key=lambda x: float(x[3].area) if len(x) > 3 and isinstance(x[3], PeakResult) else 0.0, reverse=True)
    for label, trt, tint, peak in items[:7]:
        ax.plot(np.asarray(trt, dtype=float), np.asarray(tint, dtype=float), linewidth=0.9, label=str(label))
    p = row.get("_peak")
    if isinstance(p, PeakResult):
        ax.axvspan(p.rt_start, p.rt_end, alpha=0.10)
        ax.axvline(p.apex_rt, linestyle="--", linewidth=0.8)
    ax.set_xlabel("RT (min)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    if items or rt.size:
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    msg = subtitle
    msg += f"\nQuant area={float(row.get('Area', 0.0) or 0.0):.4g}; channels={row.get('Adduct_Channels_Used', '')}"
    msg += f"\nFragility={float(row.get('Observed_Fragility_Index', 0.0) or 0.0):.3f}; ion forms={int(row.get('Ion_Form_Diversity_Count', 0) or 0)}"
    if ratio_text:
        msg += f"; {ratio_text}"
    ax.text(0.01, 0.98, msg, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)


# ==========================
# DDA MS2 export (optional)
# ==========================

_PRECURSOR_RE = re.compile(r"\bms2\s*\(?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _parse_ms2_precursor_mz(event_str: str) -> Optional[float]:
    if not event_str:
        return None
    m = _PRECURSOR_RE.search(event_str)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _get_ms2_scan_arrays(raw) -> Tuple[np.ndarray, np.ndarray]:
    scans = getattr(raw, "_ms2_scan_numbers", None)
    rts = getattr(raw, "_ms2_retention_times", None)

    if scans is not None and rts is not None:
        try:
            scans = np.asarray(scans, dtype=int)
            rts = np.asarray(rts, dtype=float)
            if scans.size == rts.size and scans.size > 0:
                return scans, rts
        except Exception:
            pass

    
    first = int(getattr(raw, "first_scan", 1))
    last = int(getattr(raw, "last_scan", first))
    scan_list: List[int] = []
    rt_list: List[float] = []

    for sn in range(first, last + 1):
        try:
            ev = raw.get_scan_event_str_from_scan_number(int(sn))
        except Exception:
            continue
        if "ms2" not in str(ev).lower():
            continue
        try:
            rt = float(raw.get_retention_time_from_scan_number(int(sn)))
        except Exception:
            rt = float("nan")
        scan_list.append(int(sn))
        rt_list.append(rt)

    return np.asarray(scan_list, dtype=int), np.asarray(rt_list, dtype=float)


@dataclass
class MS2Pick:
    scan: int
    rt: float
    precursor_mz: float
    precursor_ppm: float
    event_str: str


def find_best_ms2_for_target(
    raw,
    *,
    target_mz: float,
    ppm: float,
    rt_center: float,
    rt_start: Optional[float],
    rt_end: Optional[float],
) -> Optional[MS2Pick]:

    try:
        target_mz = float(target_mz)
    except Exception:
        return None

    if not np.isfinite(target_mz) or target_mz <= 0:
        return None

    scans, rts = _get_ms2_scan_arrays(raw)
    if scans.size == 0:
        return None

    # RT window
    def _finite(x) -> bool:
        try:
            return x is not None and np.isfinite(float(x))
        except Exception:
            return False

    if _finite(rt_start) and _finite(rt_end):
        lo = float(rt_start) - 0.05
        hi = float(rt_end) + 0.05
    elif _finite(rt_center):
        lo = float(rt_center) - 0.20
        hi = float(rt_center) + 0.20
    else:
        lo, hi = float("-inf"), float("inf")

    # candidates by RT
    try:
        mask = (rts >= lo) & (rts <= hi)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            idxs = np.arange(scans.size)
    except Exception:
        idxs = np.arange(scans.size)

    best: Optional[MS2Pick] = None
    best_score: Optional[Tuple[float, float]] = None  # (|rt-ctr|, |ppm|)
    for j in idxs.tolist():
        sn = int(scans[int(j)])
        rt = float(rts[int(j)]) if j < rts.size else float("nan")

        try:
            ev = raw.get_scan_event_str_from_scan_number(int(sn))
        except Exception:
            continue
        ev = str(ev or "")
        prec = _parse_ms2_precursor_mz(ev)
        if prec is None or not np.isfinite(float(prec)):
            continue

        err_ppm = abs(_ppm_error(float(prec), float(target_mz)))
        if not np.isfinite(err_ppm) or err_ppm > float(ppm):
            continue

        score = (abs(float(rt) - float(rt_center)) if np.isfinite(float(rt_center)) else 0.0, err_ppm)
        if best is None or best_score is None or score < best_score:
            best = MS2Pick(
                scan=sn,
                rt=float(rt),
                precursor_mz=float(prec),
                precursor_ppm=float(_ppm_error(float(prec), float(target_mz))),
                event_str=ev,
            )
            best_score = score

    return best


def save_ms2_png(
    mz: np.ndarray,
    inten: np.ndarray,
    *,
    out_png: Path,
    title: str,
    subtitle: str,
    label_top_n: int = 0,
    min_rel: float = 0.0,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    mz = np.asarray(mz, dtype=float)
    inten = np.asarray(inten, dtype=float)
    if mz.size == 0 or inten.size == 0:
        return

    # ensure sorted by m/z (some RAW may return unsorted arrays)
    try:
        order = np.argsort(mz)
        mz = mz[order]
        inten = inten[order]
    except Exception:
        pass
    # optional threshold filtering (relative to base peak)
    mz_plot = mz
    inten_plot = inten
    try:
        min_rel_f = float(min_rel) if min_rel is not None else 0.0
    except Exception:
        min_rel_f = 0.0
    if min_rel_f > 0:
        try:
            base = float(np.max(inten))
            th = base * (min_rel_f / 100.0)
            mask = inten >= th
            if int(np.count_nonzero(mask)) >= 1:
                mz_plot = mz[mask]
                inten_plot = inten[mask]
        except Exception:
            mz_plot = mz
            inten_plot = inten

    plt.figure(figsize=(10.5, 3.2))
    # stick spectrum
    plt.vlines(mz_plot, 0, inten_plot, linewidth=0.8)
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(title)

    plt.text(0.01, 0.98, subtitle, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9)

    # ========================
    # Peak labeling strategy
    # ========================
    # User requirement: for MS2 plots, label the strongest peak in EACH 50 Da bin.
    # - Keep previous behavior for compatibility: if label_top_n>0, label global top N;
    #   otherwise, label by 50 Da bins.
    ax = plt.gca()
    x_min_data, x_max_data = float(np.min(mz)), float(np.max(mz))
    y_max_data = float(np.max(inten)) if float(np.max(inten)) > 0 else 1.0

    # add headroom so labels stay inside the frame and don't touch the top border
    ax.set_xlim(x_min_data, x_max_data + 50.0)
    ax.set_ylim(0.0, y_max_data * 1.12)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # margins inside axes (avoid covering the border line)
    mx = (x_max - x_min) * 0.015
    my = (y_max - y_min) * 0.06
    dx = (x_max - x_min) * 0.008
    dy = (y_max - y_min) * 0.04

    peaks_to_label: List[Tuple[float, float]] = []
    if int(label_top_n) > 0:
        n = int(label_top_n)
        n = max(1, min(n, int(mz.size)))
        # apply the same relative threshold for labeling (optional)
        if min_rel_f > 0:
            try:
                base = float(np.max(inten))
                th = base * (min_rel_f / 100.0)
                idx_all = np.where(inten >= th)[0]
                if idx_all.size == 0:
                    idx_all = np.arange(int(mz.size))
                idx = idx_all[np.argsort(inten[idx_all])[::-1]][:n]
            except Exception:
                idx = np.argsort(inten)[::-1][:n]
        else:
            idx = np.argsort(inten)[::-1][:n]
        for k in idx.tolist():
            peaks_to_label.append((float(mz[int(k)]), float(inten[int(k)])))
    else:
        # bin labeling: each 50 Da window choose the highest intensity peak
        try:
            bin_peaks = pick_peaks_by_mz_bins(mz, inten, bin_width_da=50.0, min_rel=float(min_rel_f))
            for p in bin_peaks:
                peaks_to_label.append((float(p.mz), float(p.intensity)))
        except Exception:
            # fallback: if bin picking fails, label global top 10
            idx = np.argsort(inten)[::-1][: min(10, int(mz.size))]
            for k in idx.tolist():
                peaks_to_label.append((float(mz[int(k)]), float(inten[int(k)])))

    # place labels slightly to the side, inside axes; keep away from borders
    for x, y in peaks_to_label:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if y <= 0:
            continue

        # choose side
        x_text = x + dx
        ha = "left"
        if x_text > x_max - mx:
            x_text = x - dx
            ha = "right"
        if x_text < x_min + mx:
            x_text = x + dx
            ha = "left"
        # clamp within safe region
        x_text = min(max(x_text, x_min + mx), x_max - mx)

        y_text = y + dy
        y_text = min(max(y_text, y_min + my), y_max - my)

        ann = ax.annotate(
            f"{x:.4f}",
            xy=(x, y),
            xytext=(x_text, y_text),
            textcoords="data",
            ha=ha,
            va="bottom",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", linewidth=0.6),
            annotation_clip=True,
            clip_on=True,
        )
        try:
            ann.set_clip_on(True)
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()


def save_ms2_placeholder_png(
    *,
    out_png: Path,
    title: str,
    subtitle: str,
    message: str,
) -> None:

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10.5, 3.2))
    ax = plt.gca()
    ax.axis("off")

    ax.text(0.5, 0.78, title, ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.55, subtitle, ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.28, message, ha="center", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()


# =====================
# Main quant function
# =====================


def _truthy_flag(value: object) -> bool:
    return str(value or "").strip().lower() in {'1', 'true', 'yes', 'y', 'Yes', '是'}


def quant_xic_for_raw(
    raw_path: Path,
    targets: Sequence[TargetSpec],
    out_dir: Path,
    *,
    ppm: float = 5.0,
    ms_filter: str = "ms",
    avg_scans: int = 8,
    bin_decimals: int = 4,
    min_peak_height: float = 1e4,
    edge_ratio: float = 0.01,
    baseline_quantile: float = 0.10,
    use_observed_mz: bool = False,
    use_last_as_internal_standard: bool = False,
    # Dedicated shared-internal-standard controls.  These are intentionally
    # independent from analyte auto-adduct matching and the global peak-height rule.
    internal_standard_adduct: str = "auto_raw",
    internal_standard_ppm: Optional[float] = None,
    internal_standard_adaptive_detection: bool = True,
    internal_standard_min_snr_proxy: float = 5.0,
    internal_standard_min_height_fraction: float = 0.10,
    internal_standard_expected_rt: Optional[float] = None,
    internal_standard_rt_tolerance_min: float = 0.30,
    export_ms2_if_dda: bool = False,
    ms2_label_top_n: int = 0,
    ms2_min_rel: float = 1.0,
    adduct_mode: str = "auto",  # auto | force
    forced_adduct: str = "",  # e.g. M+H / M-H
    # Multi-peak detection mode for duplicated targets (same m/z).
    # auto: try balanced first, if not enough peaks then try sensitive.
    multi_peak_mode: str = "auto",  # auto | balanced | conservative | sensitive
    # When duplicated targets need multiple peaks, whether to relax the min_peak_height threshold
    # to try to "fill" enough peaks. Default False = STRICT: all selected peaks must pass min_peak_height.
    relax_min_peak_height_for_duplicates: bool = False,
    # For duplicated targets (same m/z): relative peak height threshold vs the strongest peak in the same XIC.
    # Peaks with height < (max_peak_height * dup_min_rel_height) will be ignored when selecting multiple peaks.
    # Set 0 to disable.
    # Note: if you input >1, it will be treated as percent (e.g. 10 -> 0.10).
    dup_min_rel_height: float = 0.10,
    # Negative-ESI dual-channel quantification. When enabled, [M-H]- is the primary
    # channel and a coeluting [M+HCOO]- peak is conditionally added to the area.
    combine_negative_formate: bool = False,
    formate_rt_tolerance_min: float = 0.10,
    formate_min_rel_height_pct: float = 1.0,
    formate_min_shape_correlation: float = 0.30,
    allow_formate_only: bool = False,
    include_formate_for_internal_standard: bool = True,
    # Extended negative-ion channel workflow.
    # legacy: original [M-H]- / optional formate behavior
    # discover: search all registered channels but do not sum non-primary channels
    # panel: load a saved experiment panel and sum only channels marked "sum"
    negative_channel_mode: str = "legacy",
    negative_channel_panel_file: str = "",
    negative_channel_rt_tolerance_min: float = 0.10,
    negative_channel_min_rel_height_pct: float = 1.0,
    negative_channel_min_shape_correlation: float = 0.30,
) -> Tuple[Path, Path]:

    raw_path = Path(raw_path)
    out_dir = ensure_dir(Path(out_dir))

    channel_mode = str(negative_channel_mode or "legacy").strip().lower()
    if channel_mode not in {"legacy", "discover", "panel"}:
        channel_mode = "legacy"
    channel_panel: Optional[NegativeIonPanel] = None
    if channel_mode == "discover":
        channel_panel = default_discovery_panel(
            rt_tolerance_min=float(negative_channel_rt_tolerance_min),
            min_rel_height_pct=float(negative_channel_min_rel_height_pct),
            min_shape_correlation=float(negative_channel_min_shape_correlation),
        )
        # Discovery must not change the quantitative area.  It only reports evidence.
        for pc in channel_panel.channels:
            defn = CHANNEL_BY_ID.get(pc.channel_id)
            pc.quant_action = "evidence" if defn is not None and not defn.quantifiable else "search"
        channel_panel.panel_name = "discovery_all_candidates"
    elif channel_mode == "panel":
        if not str(negative_channel_panel_file or "").strip():
            raise ValueError("negative_channel_mode='panel' requires a panel CSV/JSON file")
        channel_panel = load_panel(Path(str(negative_channel_panel_file)))
    panel_map = channel_panel.by_id() if channel_panel is not None else {}
    panel_formate_action = (
        normalise_action(panel_map["FORMATE"].quant_action, CHANNEL_BY_ID["FORMATE"])
        if "FORMATE" in panel_map and panel_map["FORMATE"].enabled else "exclude"
    )
    if "FORMATE" in panel_map and panel_map["FORMATE"].enabled:
        # The fixed panel owns the formate evidence thresholds as well as its sum/search action.
        formate_rt_tolerance_min = float(panel_map["FORMATE"].rt_tolerance_min)
        formate_min_rel_height_pct = float(panel_map["FORMATE"].min_rel_height_pct)
        formate_min_shape_correlation = float(panel_map["FORMATE"].min_shape_correlation)

    try:
        is_ppm = float(internal_standard_ppm) if internal_standard_ppm is not None and str(internal_standard_ppm).strip() != "" else float(ppm)
    except Exception:
        is_ppm = float(ppm)
    if not np.isfinite(is_ppm) or is_ppm <= 0:
        is_ppm = float(ppm)
    is_ppm = max(0.1, float(is_ppm))

    sample = safe_stem_from_raw_path(raw_path)
    plots_dir = out_dir / f"{sample}__XIC_plots"
    ensure_dir(plots_dir)
    _clear_png_files(plots_dir)

    
    spec = extract_averaged_spectrum_from_raw(
        raw_path,
        avg_scans=int(avg_scans),
        bin_decimals=int(bin_decimals),
        ms_filter="ms",  
    )

    target_list: List[TargetSpec] = []
    source_targets = list(targets)
    for target_index, target in enumerate(source_targets):
        raw_meta = dict(getattr(target, "raw_row", None) or {})
        is_last_requested = bool(use_last_as_internal_standard) and target_index == len(source_targets) - 1
        if is_last_requested and not _truthy_flag(raw_meta.get("is_internal_standard", "")):
            raw_meta["is_internal_standard"] = "1"
            target = TargetSpec(
                file_key=target.file_key,
                name=target.name or "Internal Standard",
                formula=target.formula,
                adduct=target.adduct,
                polarity=target.polarity,
                theoretical_mz=target.theoretical_mz,
                ppm=target.ppm,
                raw_row=raw_meta,
            )
        target_list.append(target)

    match_rows: List[MatchRow] = match_targets_to_spectrum(
        spec,
        target_list,
        default_ppm=float(ppm),
        default_polarity="positive",
        min_rel_for_found=0.0,
        adduct_mode=str(adduct_mode or "auto"),
        forced_adduct=str(forced_adduct or ""),
        prefer_proton=True,
    )
    # The shared internal standard must keep the same ion form in the clean
    # standards and the dense unknown sample.  Do not let chance peaks in the
    # average spectrum switch it from [M-H]- to [M+H]+ (or another adduct).
    match_rows = _override_internal_standard_matches(
        spec,
        target_list,
        match_rows,
        ppm=float(is_ppm),
        global_adduct_mode=str(adduct_mode or "auto"),
        global_forced_adduct=str(forced_adduct or ""),
        internal_standard_adduct=str(internal_standard_adduct or "auto_raw"),
    )

    
    try:
        from fisher_py import RawFile
    except Exception as e:
        raise RuntimeError(
            f'Could not import fisher_py for XIC extraction.\nInstall it with: python -m pip install fisher-py==1.0.22\nOriginal error: {e}'
        )
    raw = RawFile(str(raw_path))

    is_dda = "dda" in str(raw_path.name).lower()
    export_ms2 = bool(export_ms2_if_dda) and bool(is_dda)
    ms2_dir = out_dir / f"{sample}__MS2_plots"
    if export_ms2:
        ensure_dir(ms2_dir)
        _clear_png_files(ms2_dir)

    
    
    rows: List[Dict[str, object]] = []

    # 3.1) Build base rows while preserving target order.
    for i, (t, m) in enumerate(zip(target_list, match_rows), start=1):
        name = (m.name or "").strip()
        formula = (m.formula_hill or m.formula_input or "").strip()
        matched_adduct = (m.adduct or "").strip()
        polarity = str(m.polarity or "").strip().lower()
        theo_mz = float(m.theoretical_mz) if m.theoretical_mz is not None else float("nan")

        mz0 = theo_mz
        if use_observed_mz and m.observed_mz is not None and np.isfinite(float(m.observed_mz)):
            mz0 = float(m.observed_mz)

        raw_meta = getattr(t, "raw_row", None) or {}
        combo = _extract_combo_from_raw_row(raw_meta)
        is_internal_standard = _truthy_flag(raw_meta.get("is_internal_standard", "")) or (
            bool(use_last_as_internal_standard) and i == len(target_list)
        )

        negative_identity = (
            polarity.startswith("neg")
            or matched_adduct.endswith("-")
            or "M-H" in matched_adduct.upper()
            or "HCOO" in matched_adduct.upper()
            or "FA-H" in matched_adduct.upper()
        )
        multi_channel_requested = channel_panel is not None and bool(negative_identity)
        dual_requested = bool(combine_negative_formate or multi_channel_requested) and bool(negative_identity)
        formate_sum_enabled = bool(combine_negative_formate)
        if channel_panel is not None:
            formate_sum_enabled = bool(panel_formate_action == "sum")
        if is_internal_standard and not bool(include_formate_for_internal_standard):
            formate_sum_enabled = False

        mh_theoretical = float("nan")
        formate_theoretical = float("nan")
        dual_note = ""
        if dual_requested:
            # Prefer an explicit neutral formula. If only a theoretical m/z is available,
            # derive the other channel from the exact neutral formic-acid mass difference.
            try:
                if formula:
                    _, neutral_mass = compute_neutral_mass_from_formula(formula)
                    mh_theoretical = calc_mz(float(neutral_mass), NEG_ADDUCTS["H"])
                    formate_theoretical = calc_mz(float(neutral_mass), NEG_ADDUCTS["FA"])
                elif np.isfinite(theo_mz):
                    ad_u = matched_adduct.upper()
                    if "HCOO" in ad_u or "FA-H" in ad_u:
                        formate_theoretical = float(theo_mz)
                        mh_theoretical = float(theo_mz) - float(MASS_MH_TO_FORMATE)
                    else:
                        mh_theoretical = float(theo_mz)
                        formate_theoretical = float(theo_mz) + float(MASS_MH_TO_FORMATE)
            except Exception as e:
                dual_note = f"dual-adduct m/z calculation failed: {e}"

            if not np.isfinite(mh_theoretical) or not np.isfinite(formate_theoretical):
                dual_requested = False
                if not dual_note:
                    dual_note = "dual-adduct mode disabled for this row because neutral formula/mass was unavailable"

        mh_extract_mz = mh_theoretical
        formate_extract_mz = formate_theoretical
        if dual_requested and use_observed_mz and m.observed_mz is not None and np.isfinite(float(m.observed_mz)):
            obs = float(m.observed_mz)
            ad_u = matched_adduct.upper()
            if "HCOO" in ad_u or "FA-H" in ad_u:
                formate_extract_mz = obs
                mh_extract_mz = obs - float(MASS_MH_TO_FORMATE)
            elif "M-H" in ad_u or matched_adduct == "[M-H]-":
                mh_extract_mz = obs
                formate_extract_mz = obs + float(MASS_MH_TO_FORMATE)

        if dual_requested:
            # [M-H]- remains the primary XIC coordinate. The accepted formate area is added later.
            mz0 = float(mh_extract_mz)
            adduct_display = "[M-H]- + optional [M+HCOO]-"
            key_val = float(mh_theoretical) if np.isfinite(mh_theoretical) else float(mz0)
        else:
            adduct_display = matched_adduct
            key_val = float(theo_mz) if np.isfinite(float(theo_mz)) else float(mz0)

        mz_key = f"{key_val:.5f}" if np.isfinite(key_val) else f"nan_{i}"
        if is_internal_standard:
            mz_key = f"IS::{mz_key}"

        rows.append(
            {
                "No": i,
                "Name": name,
                "Formula": formula,
                "Combo": combo,
                "Is_internal_standard": bool(is_internal_standard),
                "Polarity": polarity,
                "Matched_Adduct": matched_adduct,
                "Adduct": adduct_display,
                "Theoretical_mz": theo_mz,
                "XIC_mz": float(mz0) if np.isfinite(float(mz0)) else float("nan"),
                "MH_Theoretical_mz": float(mh_theoretical) if np.isfinite(mh_theoretical) else "",
                "MH_XIC_mz": float(mh_extract_mz) if np.isfinite(mh_extract_mz) else "",
                "Formate_Theoretical_mz": float(formate_theoretical) if np.isfinite(formate_theoretical) else "",
                "Formate_XIC_mz": float(formate_extract_mz) if np.isfinite(formate_extract_mz) else "",
                "Dual_Adduct_Enabled": bool(dual_requested),
                "Formate_Sum_Enabled": bool(formate_sum_enabled),
                "Negative_Channel_Mode": channel_mode,
                "Negative_Channel_Panel": channel_panel.panel_name if channel_panel is not None else "",
                "Dual_Adduct_Note": dual_note,
                "PPM": float(is_ppm if is_internal_standard else ppm),
                "IS_PPM": float(is_ppm) if is_internal_standard else "",
                "MS_filter": "",
                "Apex_RT": float("nan"),
                "RT_start": float("nan"),
                "RT_end": float("nan"),
                "Peak_height": 0.0,
                "Area": 0.0,
                "Found": False,
                "MH_Found": False,
                "MH_Area": 0.0,
                "MH_Peak_height": 0.0,
                "MH_Apex_RT": float("nan"),
                "Formate_Found": False,
                "Formate_Accepted": False,
                "Formate_Area": 0.0,
                "Formate_Peak_height": 0.0,
                "Formate_Apex_RT": float("nan"),
                "Formate_RT_Delta_min": float("nan"),
                "Formate_Shape_Correlation": float("nan"),
                "Formate_Area_Fraction_%": 0.0,
                "Adduct_Channels_Used": matched_adduct,
                "Formate_Note": "",
                "IS_Dedicated_Adduct": matched_adduct if is_internal_standard else "",
                "IS_Diagnostic_Status": "",
                "IS_Accepted_By": "",
                "IS_Candidate_Apex_RT": "",
                "IS_Candidate_RT_Start": "",
                "IS_Candidate_RT_End": "",
                "IS_Candidate_Area": "",
                "IS_Candidate_Raw_Apex": "",
                "IS_Candidate_Baseline": "",
                "IS_Candidate_Net_Height": "",
                "IS_Noise_Sigma_Proxy": "",
                "IS_SNR_Proxy": "",
                "IS_Global_Min_Peak_Height": float(min_peak_height) if is_internal_standard else "",
                "IS_Adaptive_Min_Height": "",
                "IS_Expected_RT": internal_standard_expected_rt if is_internal_standard and internal_standard_expected_rt is not None else "",
                "IS_Expected_RT_Tolerance_Min": float(internal_standard_rt_tolerance_min) if is_internal_standard else "",
                "IS_Diagnostic_Note": "",
                "IS_Diagnostic_PNG": "",
                "IS_XIC_Plot_In_Plots_Folder": "",
                "IS_Diagnostic_CSV": "",
                "IS_Trace_CSV": "",
                "IS_All_Channels_PNG": "",
                "PeakIndex": 1,
                "PeakCount": 1,
                "_mz_key": mz_key,
                "_rt": np.asarray([], dtype=float),
                "_inten": np.asarray([], dtype=float),
                "_peak": None,
                "_dual_peak": None,
                "_rt_mh": np.asarray([], dtype=float),
                "_inten_mh": np.asarray([], dtype=float),
                "_rt_formate": np.asarray([], dtype=float),
                "_inten_formate": np.asarray([], dtype=float),
                "MS2_Found": "",
                "MS2_Scan": "",
                "MS2_RT": "",
                "MS2_Precursor_mz": "",
                "MS2_Precursor_ppm": "",
                "MS2_Target_Adduct": "",
                "MS2_PNG": "",
            }
        )

    
    groups: Dict[str, List[int]] = {}
    for idx, r in enumerate(rows):
        k = str(r.get("_mz_key") or "")
        groups.setdefault(k, []).append(int(idx))

    for mz_key, idx_list in groups.items():
        if not idx_list:
            continue

        need = int(len(idx_list))
        try:
            dup_rel = float(dup_min_rel_height)
        except Exception:
            dup_rel = 0.0
        if not np.isfinite(dup_rel):
            dup_rel = 0.0
        if dup_rel > 1.0:
            dup_rel = dup_rel / 100.0
        dup_rel = float(max(0.0, min(1.0, dup_rel)))

        mode = str(multi_peak_mode or "auto").strip().lower()
        if mode in ('auto', 'adaptive', 'recommended', '', '推荐'):
            modes = ["balanced", "sensitive"]
        elif mode in ('balanced', '平衡'):
            modes = ["balanced"]
        elif mode in ('conservative', 'strict', '保守'):
            modes = ["conservative"]
        elif mode in ('sensitive', 'loose', '敏感'):
            modes = ["sensitive"]
        else:
            modes = ["balanced", "sensitive"]

        def _try_modes_on_trace(
            rt_trace: np.ndarray,
            int_trace: np.ndarray,
            *,
            min_h: float,
            desired: int,
            relative_threshold: float,
        ) -> List[PeakResult]:
            best: List[PeakResult] = []
            for mm in modes:
                kw = _multipeak_params_for_mode(mm)
                ps = integrate_top_peaks(
                    rt_trace,
                    int_trace,
                    n_peaks=int(max(1, desired)),
                    min_peak_height=float(min_h),
                    edge_ratio=float(edge_ratio),
                    baseline_quantile=float(baseline_quantile),
                    min_peak_distance_min=float(kw.get("min_peak_distance_min", 0.0)),
                    smooth_points=int(kw.get("smooth_points", 3)),
                    min_prominence_ratio=float(kw.get("min_prominence_ratio", 0.05)),
                    merge_valley_ratio=float(kw.get("merge_valley_ratio", 0.92)),
                    min_rel_height=float(relative_threshold),
                )
                if len(ps) > len(best):
                    best = ps
                if len(best) >= int(desired):
                    break
            return best

        dual_group = any(bool(rows[ii].get("Dual_Adduct_Enabled", False)) for ii in idx_list)
        is_group = any(bool(rows[ii].get("Is_internal_standard", False)) for ii in idx_list)
        group_ppm = float(is_ppm if is_group else ppm)

        if dual_group:
            mh_vals = []
            fa_vals = []
            for ii in idx_list:
                try:
                    mhv = float(rows[ii].get("MH_XIC_mz", float("nan")))
                except Exception:
                    mhv = float("nan")
                try:
                    fav = float(rows[ii].get("Formate_XIC_mz", float("nan")))
                except Exception:
                    fav = float("nan")
                if np.isfinite(mhv):
                    mh_vals.append(mhv)
                if np.isfinite(fav):
                    fa_vals.append(fav)
            mh_mz = float(np.median(np.asarray(mh_vals, dtype=float))) if mh_vals else float("nan")
            fa_mz = float(np.median(np.asarray(fa_vals, dtype=float))) if fa_vals else float("nan")
            if not np.isfinite(mh_mz) or not np.isfinite(fa_mz):
                continue

            rt_mh, int_mh, ms_used_mh = _get_chromatogram_massrange(
                raw, mz=mh_mz, ppm=float(group_ppm), ms_filter=ms_filter
            )
            rt_fa, int_fa, ms_used_fa = _get_chromatogram_massrange(
                raw, mz=fa_mz, ppm=float(group_ppm), ms_filter=ms_filter
            )

            if is_group:
                accepted_is_peak, candidate_is_peak, is_diag = _select_internal_standard_peak(
                    rt_mh,
                    int_mh,
                    global_min_peak_height=float(min_peak_height),
                    edge_ratio=float(edge_ratio),
                    baseline_quantile=float(baseline_quantile),
                    adaptive=bool(internal_standard_adaptive_detection),
                    min_snr_proxy=float(internal_standard_min_snr_proxy),
                    min_fraction_of_global_height=float(internal_standard_min_height_fraction),
                    expected_rt=internal_standard_expected_rt,
                    expected_rt_tolerance_min=float(internal_standard_rt_tolerance_min),
                )
                mh_peaks = [accepted_is_peak] if isinstance(accepted_is_peak, PeakResult) else []
                for ii in idx_list:
                    if bool(rows[ii].get("Is_internal_standard", False)):
                        _apply_internal_standard_diag(rows[ii], is_diag, candidate_is_peak)
            else:
                mh_peaks = _try_modes_on_trace(
                    rt_mh,
                    int_mh,
                    min_h=float(min_peak_height),
                    desired=need,
                    relative_threshold=float(dup_rel if need > 1 else 0.0),
                )
                if bool(relax_min_peak_height_for_duplicates) and need > 1 and len(mh_peaks) < need:
                    mh2 = _try_modes_on_trace(
                        rt_mh,
                        int_mh,
                        min_h=float(max(0.0, float(min_peak_height) * 0.20)),
                        desired=need,
                        relative_threshold=float(dup_rel),
                    )
                    if len(mh2) > len(mh_peaks):
                        mh_peaks = mh2

            # Retrieve extra formate candidates because unrelated ions at the same m/z can exist;
            # only one-to-one coeluting candidates will be accepted below.
            fa_candidate_n = int(max(8, need * 4))
            formate_peaks = _try_modes_on_trace(
                rt_fa,
                int_fa,
                min_h=float(min_peak_height),
                desired=fa_candidate_n,
                relative_threshold=0.0,
            )

            dual_peaks = build_dual_adduct_peaks(
                mh_peaks,
                formate_peaks,
                need=need,
                rt_mh=rt_mh,
                int_mh=int_mh,
                rt_formate=rt_fa,
                int_formate=int_fa,
                rt_tolerance_min=float(formate_rt_tolerance_min),
                min_formate_rel_height_pct=float(formate_min_rel_height_pct),
                min_shape_correlation=float(formate_min_shape_correlation),
                baseline_quantile=float(baseline_quantile),
                allow_formate_only=bool(allow_formate_only),
            )

            for j, row_idx in enumerate(idx_list):
                dual = dual_peaks[j] if j < len(dual_peaks) else None
                r = rows[row_idx]
                r["MS_filter"] = ms_used_mh or ms_used_fa or (ms_filter or "")
                r["PeakIndex"] = j + 1
                r["PeakCount"] = need
                r["_rt_mh"] = rt_mh
                r["_inten_mh"] = int_mh
                r["_rt_formate"] = rt_fa
                r["_inten_formate"] = int_fa
                r["_dual_peak"] = dual

                if dual is None:
                    r["Found"] = False
                    r["Area"] = 0.0
                    r["Peak_height"] = 0.0
                    r["Apex_RT"] = float("nan")
                    r["RT_start"] = float("nan")
                    r["RT_end"] = float("nan")
                    r["Formate_Note"] = "no accepted [M-H]-/formate chromatographic component"
                    r["_rt"] = rt_mh
                    r["_inten"] = int_mh
                    r["_peak"] = None
                    continue

                mhp = dual.mh_peak
                fap = dual.formate_peak
                sum_formate_here = bool(r.get("Formate_Sum_Enabled", False))
                if mhp is not None:
                    p = dual.combined if (sum_formate_here and fap is not None and dual.formate_accepted) else mhp
                elif fap is not None and sum_formate_here and bool(allow_formate_only):
                    p = fap
                else:
                    r["Found"] = False
                    r["Area"] = 0.0
                    r["Peak_height"] = 0.0
                    r["Apex_RT"] = float("nan")
                    r["RT_start"] = float("nan")
                    r["RT_end"] = float("nan")
                    r["Formate_Note"] = "formate evidence found but panel/legacy settings require an [M-H]- anchor"
                    r["_rt"] = rt_mh
                    r["_inten"] = int_mh
                    r["_peak"] = None
                    continue

                r["Found"] = True
                r["Area"] = float(p.area)
                r["Peak_height"] = float(p.peak_height)
                r["Apex_RT"] = float(p.apex_rt)
                r["RT_start"] = float(p.rt_start)
                r["RT_end"] = float(p.rt_end)
                r["_peak"] = p
                r["MH_Found"] = bool(mhp is not None)
                r["MH_Area"] = float(mhp.area) if mhp is not None else 0.0
                r["MH_Peak_height"] = float(mhp.peak_height) if mhp is not None else 0.0
                r["MH_Apex_RT"] = float(mhp.apex_rt) if mhp is not None else float("nan")
                r["Formate_Found"] = bool(fap is not None)
                r["Formate_Accepted"] = bool(dual.formate_accepted and fap is not None)
                r["Formate_Area"] = float(fap.area) if fap is not None else 0.0
                r["Formate_Peak_height"] = float(fap.peak_height) if fap is not None else 0.0
                r["Formate_Apex_RT"] = float(fap.apex_rt) if fap is not None else float("nan")
                r["Formate_RT_Delta_min"] = float(dual.formate_rt_delta_min)
                r["Formate_Shape_Correlation"] = float(dual.formate_shape_correlation)
                r["Adduct_Channels_Used"] = (
                    dual.channels_used if sum_formate_here else ("[M-H]-" if mhp is not None else dual.channels_used)
                )
                r["Formate_Note"] = dual.formate_note
                r["Formate_Area_Fraction_%"] = (
                    float(fap.area) / float(p.area) * 100.0
                    if fap is not None and float(p.area) > 0
                    else 0.0
                )

                if sum_formate_here and dual.formate_accepted and fap is not None:
                    if rt_mh.size:
                        r["_rt"] = rt_mh
                        r["_inten"] = np.asarray(int_mh, dtype=float) + _align_trace(rt_mh, rt_fa, int_fa)
                    else:
                        r["_rt"] = rt_fa
                        r["_inten"] = int_fa
                else:
                    r["_rt"] = rt_mh if rt_mh.size else rt_fa
                    r["_inten"] = int_mh if rt_mh.size else int_fa

        else:
            mz_vals: List[float] = []
            for ii in idx_list:
                try:
                    v = float(rows[ii].get("XIC_mz", float("nan")))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    mz_vals.append(float(v))
            mz0 = float(np.median(np.asarray(mz_vals, dtype=float))) if mz_vals else float("nan")
            if not np.isfinite(mz0):
                continue

            rt, inten, ms_used = _get_chromatogram_massrange(
                raw, mz=mz0, ppm=float(group_ppm), ms_filter=ms_filter
            )
            if is_group:
                accepted_is_peak, candidate_is_peak, is_diag = _select_internal_standard_peak(
                    rt,
                    inten,
                    global_min_peak_height=float(min_peak_height),
                    edge_ratio=float(edge_ratio),
                    baseline_quantile=float(baseline_quantile),
                    adaptive=bool(internal_standard_adaptive_detection),
                    min_snr_proxy=float(internal_standard_min_snr_proxy),
                    min_fraction_of_global_height=float(internal_standard_min_height_fraction),
                    expected_rt=internal_standard_expected_rt,
                    expected_rt_tolerance_min=float(internal_standard_rt_tolerance_min),
                )
                peaks = [accepted_is_peak] if isinstance(accepted_is_peak, PeakResult) else []
                for ii in idx_list:
                    if bool(rows[ii].get("Is_internal_standard", False)):
                        _apply_internal_standard_diag(rows[ii], is_diag, candidate_is_peak)
            else:
                peaks = _try_modes_on_trace(
                    rt,
                    inten,
                    min_h=float(min_peak_height),
                    desired=need,
                    relative_threshold=float(dup_rel if need > 1 else 0.0),
                )
                if bool(relax_min_peak_height_for_duplicates) and need > 1 and len(peaks) < need:
                    peaks2 = _try_modes_on_trace(
                        rt,
                        inten,
                        min_h=float(max(0.0, float(min_peak_height) * 0.20)),
                        desired=need,
                        relative_threshold=float(dup_rel),
                    )
                    if len(peaks2) > len(peaks):
                        peaks = peaks2

            for j, row_idx in enumerate(idx_list):
                peak = peaks[j] if j < len(peaks) else None
                r = rows[row_idx]
                r["MS_filter"] = ms_used or (ms_filter or "")
                r["PeakIndex"] = j + 1
                r["PeakCount"] = need
                r["_rt"] = rt
                r["_inten"] = inten
                r["_peak"] = peak

                if peak is None:
                    r["Found"] = False
                    r["Area"] = 0.0
                    r["Peak_height"] = 0.0
                    r["Apex_RT"] = float("nan")
                    r["RT_start"] = float("nan")
                    r["RT_end"] = float("nan")
                else:
                    r["Found"] = True
                    r["Area"] = float(peak.area)
                    r["Peak_height"] = float(peak.peak_height)
                    r["Apex_RT"] = float(peak.apex_rt)
                    r["RT_start"] = float(peak.rt_start)
                    r["RT_end"] = float(peak.rt_end)
                    matched = str(r.get("Matched_Adduct", ""))
                    if matched == "[M-H]-":
                        r["MH_Found"] = True
                        r["MH_Area"] = float(peak.area)
                        r["MH_Peak_height"] = float(peak.peak_height)
                        r["MH_Apex_RT"] = float(peak.apex_rt)
                        r["Adduct_Channels_Used"] = "[M-H]-"
                    elif "HCOO" in matched.upper() or "FA-H" in matched.upper():
                        r["Formate_Found"] = True
                        r["Formate_Accepted"] = True
                        r["Formate_Area"] = float(peak.area)
                        r["Formate_Peak_height"] = float(peak.peak_height)
                        r["Formate_Apex_RT"] = float(peak.apex_rt)
                        r["Formate_Area_Fraction_%"] = 100.0
                        r["Adduct_Channels_Used"] = "[M+HCOO]-"
                        r["Formate_Note"] = "single-adduct formate extraction"

    # 3.4) Extended negative-ion channel search / fixed-panel summation.
    channel_evidence: List[Dict[str, object]] = []
    if channel_panel is not None:
        channel_evidence = apply_negative_ion_panel(
            raw,
            rows,
            panel=channel_panel,
            ppm=float(ppm),
            ms_filter=ms_filter,
            min_peak_height=float(min_peak_height),
            edge_ratio=float(edge_ratio),
            baseline_quantile=float(baseline_quantile),
            include_formate_evidence=True,
            include_for_internal_standard=bool(include_formate_for_internal_standard),
            internal_standard_ppm=float(is_ppm),
            internal_standard_min_peak_height=float(
                max(0.0, float(min_peak_height) * max(0.0, min(1.0, float(internal_standard_min_height_fraction))))
            ),
        )

    
    ratio_mode = "percent_of_total_area"
    internal_name = ""
    internal_area = 0.0
    internal_found = False

    if use_last_as_internal_standard and rows:
        ratio_mode = "vs_internal_standard"
        explicit_is_rows = [r for r in rows if bool(r.get("Is_internal_standard", False))]
        internal_row = explicit_is_rows[-1] if explicit_is_rows else rows[-1]
        internal_name = str(internal_row.get("Name") or internal_row.get("Formula") or "")
        try:
            internal_area = float(internal_row.get("Area", 0.0) or 0.0)
        except Exception:
            internal_area = 0.0
        internal_found = bool(internal_row.get("Found", False)) and internal_area > 0

        for r in rows:
            a = float(r.get("Area", 0.0) or 0.0)
            if internal_found:
                r["Ratio_%"] = a / internal_area * 100.0
            else:
                
                r["Ratio_%"] = ""
            r["Ratio_mode"] = ratio_mode
            r["Internal_standard"] = internal_name
            r["Internal_area"] = internal_area if internal_found else ""
            r["Internal_standard_found"] = bool(internal_found)
    else:
        total_area = float(sum(float(r.get("Area", 0.0) or 0.0) for r in rows))
        for r in rows:
            a = float(r.get("Area", 0.0) or 0.0)
            r["Ratio_%"] = a / total_area * 100.0 if total_area > 0 else 0.0
            r["Ratio_mode"] = ratio_mode
            r["Internal_standard"] = ""
            r["Internal_area"] = ""
            r["Internal_standard_found"] = ""

    
    if export_ms2:
        for r in rows:
            
            label_main = str(r.get("Name") or "") or str(r.get("Formula") or "")
            if not label_main:
                label_main = f"Target_{int(r.get('No', 0))}"

            out_png = ms2_dir / f"{int(r.get('No', 0)):03d}__{_safe_filename(label_main)}__MS2.png"
            title = f"MS2: {label_main}"

            
            r["MS2_Found"] = False
            r["MS2_PNG"] = str(out_png)

            peak = r.get("_peak")
            mz0 = float(r.get("XIC_mz", float("nan")))

            if not isinstance(peak, PeakResult) or (not np.isfinite(mz0)):
                subtitle = f"precursor(target)={mz0:.4f}  ppm=\u00B1{float(ppm):g}" if np.isfinite(mz0) else ""
                save_ms2_placeholder_png(
                    out_png=out_png,
                    title=title,
                    subtitle=subtitle,
                    message="NO XIC PEAK / NO MS2",
                )
                continue

            candidate_mzs: List[Tuple[float, str]] = [(mz0, str(r.get("Matched_Adduct") or r.get("Adduct") or "target"))]
            if bool(r.get("Dual_Adduct_Enabled", False)):
                candidate_mzs = []
                try:
                    mh_mz = float(r.get("MH_XIC_mz", float("nan")))
                except Exception:
                    mh_mz = float("nan")
                try:
                    fa_mz = float(r.get("Formate_XIC_mz", float("nan")))
                except Exception:
                    fa_mz = float("nan")
                if np.isfinite(mh_mz) and bool(r.get("MH_Found", False)):
                    candidate_mzs.append((mh_mz, "[M-H]-"))
                if np.isfinite(fa_mz) and bool(r.get("Formate_Accepted", False)):
                    candidate_mzs.append((fa_mz, "[M+HCOO]-"))
                if not candidate_mzs and np.isfinite(mh_mz):
                    candidate_mzs.append((mh_mz, "[M-H]-"))

            pick = None
            pick_adduct = ""
            pick_score = None
            for cand_mz, cand_adduct in candidate_mzs:
                cand_pick = find_best_ms2_for_target(
                    raw,
                    target_mz=float(cand_mz),
                    ppm=float(ppm),
                    rt_center=float(peak.apex_rt),
                    rt_start=float(peak.rt_start),
                    rt_end=float(peak.rt_end),
                )
                if cand_pick is None:
                    continue
                score = (abs(float(cand_pick.precursor_ppm)), abs(float(cand_pick.rt) - float(peak.apex_rt)))
                if pick is None or pick_score is None or score < pick_score:
                    pick = cand_pick
                    pick_adduct = cand_adduct
                    pick_score = score

            if pick is None:
                subtitle = f"RT~{float(peak.apex_rt):.2f} min  precursor(target)={mz0:.4f}\nNO MS2 MATCH (\u00B1{float(ppm):g} ppm)"
                save_ms2_placeholder_png(
                    out_png=out_png,
                    title=title,
                    subtitle=subtitle,
                    message="MS2 NOT FOUND",
                )
                continue

            
            try:
                mz, inten, charges, event_str = raw.get_scan_from_scan_number(int(pick.scan))
                mz = np.asarray(mz, dtype=float)
                inten = np.asarray(inten, dtype=float)
            except Exception:
                subtitle = f"scan={pick.scan}  RT={pick.rt:.2f} min"
                save_ms2_placeholder_png(
                    out_png=out_png,
                    title=title,
                    subtitle=subtitle,
                    message="FAILED TO READ MS2",
                )
                continue

            if mz.size == 0 or inten.size == 0:
                subtitle = f"scan={pick.scan}  RT={pick.rt:.2f} min"
                save_ms2_placeholder_png(
                    out_png=out_png,
                    title=title,
                    subtitle=subtitle,
                    message="EMPTY MS2",
                )
                continue

            subtitle = f"scan={pick.scan}  RT={pick.rt:.2f} min\nprecursor={pick.precursor_mz:.4f}  err={pick.precursor_ppm:+.2f} ppm  {pick_adduct}"
            save_ms2_png(
                mz,
                inten,
                out_png=out_png,
                title=title,
                subtitle=subtitle,
                label_top_n=int(ms2_label_top_n),
                min_rel=float(ms2_min_rel),
            )

            r["MS2_Found"] = True
            r["MS2_Scan"] = int(pick.scan)
            r["MS2_RT"] = float(pick.rt)
            # keep 4 decimals for precursor m/z in CSV output (for readability)
            r["MS2_Precursor_mz"] = f"{float(pick.precursor_mz):.4f}"
            r["MS2_Precursor_ppm"] = float(pick.precursor_ppm)
            r["MS2_Target_Adduct"] = pick_adduct

    # 5) XIC plots.  The internal standard is always written with a clear,
    # dedicated filename and a diagnostic plot, even when the peak is rejected.
    is_diag_rows: List[Dict[str, object]] = []
    is_diag_csv = out_dir / f"{sample}__Internal_Standard_diagnostic.csv"
    for r in rows:
        name = str(r.get("Name", "") or "")
        formula = str(r.get("Formula", "") or "")
        adduct = str(r.get("Adduct", "") or "")
        mz0 = float(r.get("XIC_mz", float("nan")))
        rt = r.get("_rt")
        inten = r.get("_inten")
        peak = r.get("_peak")
        dual = r.get("_dual_peak")
        is_internal = bool(r.get("Is_internal_standard", False))

        if not isinstance(rt, np.ndarray):
            rt = np.asarray([], dtype=float)
        if not isinstance(inten, np.ndarray):
            inten = np.asarray([], dtype=float)

        label_main = "Internal_Standard" if is_internal else (name if name else formula)
        if not label_main:
            label_main = f"Target_{int(r.get('No', 0))}"

        if is_internal:
            # Put this file at the sheet output root so it is not buried among 1000 plots.
            out_png = out_dir / f"{sample}__Internal_Standard_XIC.png"
            diag = {k: r.get(k, "") for k in (
                "IS_Diagnostic_Status", "IS_Accepted_By", "IS_Candidate_Apex_RT",
                "IS_Candidate_RT_Start", "IS_Candidate_RT_End", "IS_Candidate_Area",
                "IS_Candidate_Raw_Apex", "IS_Candidate_Baseline",
                "IS_Candidate_Net_Height", "IS_Noise_Sigma_Proxy", "IS_SNR_Proxy",
                "IS_Global_Min_Peak_Height", "IS_Adaptive_Min_Height",
                "IS_Expected_RT", "IS_Expected_RT_Tolerance_Min", "IS_Diagnostic_Note",
            )}
            if not str(diag.get("IS_Diagnostic_Status") or "").strip():
                diag["IS_Diagnostic_Status"] = "NO_TRACE" if rt.size == 0 else "NOT_EVALUATED"
                diag["IS_Diagnostic_Note"] = str(diag.get("IS_Diagnostic_Note") or "internal-standard diagnostic was not evaluated")
            primary_rt = rt
            primary_inten = inten
            accepted_peak = peak if isinstance(peak, PeakResult) and bool(r.get("Found", False)) else None
            if bool(r.get("Dual_Adduct_Enabled", False)):
                rt_mh_diag = r.get("_rt_mh")
                int_mh_diag = r.get("_inten_mh")
                if isinstance(rt_mh_diag, np.ndarray):
                    primary_rt = rt_mh_diag
                if isinstance(int_mh_diag, np.ndarray):
                    primary_inten = int_mh_diag
                if isinstance(dual, DualAdductPeakResult) and isinstance(dual.mh_peak, PeakResult):
                    accepted_peak = dual.mh_peak
            candidate_peak = r.get("_is_candidate_peak")
            if not isinstance(candidate_peak, PeakResult):
                candidate_peak = accepted_peak
            save_internal_standard_diagnostic_png(
                primary_rt,
                primary_inten,
                accepted_peak=accepted_peak,
                candidate_peak=candidate_peak if isinstance(candidate_peak, PeakResult) else None,
                diag=diag,
                out_png=out_png,
                formula=formula,
                adduct=str(r.get("Matched_Adduct") or adduct),
                mz=mz0,
                ppm=float(r.get("IS_PPM") or is_ppm),
            )
            # Also place a clearly named copy inside the ordinary XIC plot folder.
            # Older versions used a Chinese internal-standard label that was sanitised to
            # ``item``; users therefore could not recognise the last plot as the IS trace.
            plot_copy = plots_dir / f"{int(r.get('No', 0)):03d}__Internal_Standard__mz{mz0:.5f}.png"
            try:
                shutil.copy2(str(out_png), str(plot_copy))
            except Exception:
                plot_copy = Path("")
            trace_csv = out_dir / f"{sample}__Internal_Standard_XIC_trace.csv"
            try:
                base_val = _num_for_diag(diag.get("IS_Candidate_Baseline"))
                base_val = float(base_val or 0.0)
                formate_rt_trace = r.get("_rt_formate")
                formate_int_trace = r.get("_inten_formate")
                if not isinstance(formate_rt_trace, np.ndarray):
                    formate_rt_trace = np.asarray([], dtype=float)
                if not isinstance(formate_int_trace, np.ndarray):
                    formate_int_trace = np.asarray([], dtype=float)
                formate_aligned = (
                    _align_trace(primary_rt, formate_rt_trace, formate_int_trace)
                    if primary_rt.size and formate_rt_trace.size else np.zeros_like(primary_rt, dtype=float)
                )
                with trace_csv.open("w", encoding="utf-8-sig", newline="") as tf:
                    tw = csv.writer(tf)
                    tw.writerow(["RT_min", "Primary_intensity", "Primary_baseline_corrected", "Formate_intensity_aligned"])
                    for rt_v, raw_v, fa_v in zip(primary_rt.tolist(), primary_inten.tolist(), formate_aligned.tolist()):
                        tw.writerow([float(rt_v), float(raw_v), max(0.0, float(raw_v) - base_val), float(fa_v)])
            except Exception:
                trace_csv = Path("")

            r["XIC_PNG"] = str(plot_copy) if str(plot_copy) else str(out_png)
            r["IS_Diagnostic_PNG"] = str(out_png)
            r["IS_XIC_Plot_In_Plots_Folder"] = str(plot_copy) if str(plot_copy) else ""
            r["IS_Diagnostic_CSV"] = str(is_diag_csv)
            r["IS_Trace_CSV"] = str(trace_csv) if str(trace_csv) else ""
            if channel_panel is not None:
                all_channels_png = out_dir / f"{sample}__Internal_Standard_All_Channels.png"
                save_multi_channel_xic_png(
                    r,
                    out_png=all_channels_png,
                    title=f"Internal standard - all accepted ion channels: {formula}",
                    subtitle=(
                        f"Negative-ion panel={channel_panel.panel_name}; primary ppm=±{float(is_ppm):g}; "
                        f"status={diag.get('IS_Diagnostic_Status', '')}"
                    ),
                    ratio_text="",
                )
                r["IS_All_Channels_PNG"] = str(all_channels_png)
            is_diag_rows.append(r)
            continue

        out_png = plots_dir / f"{int(r.get('No', 0)):03d}__{_safe_filename(label_main)}__mz{mz0:.5f}.png"
        title = f"XIC: {label_main}"
        peak_idx = int(r.get("PeakIndex", 1) or 1)
        peak_cnt = int(r.get("PeakCount", 1) or 1)

        if bool(r.get("Dual_Adduct_Enabled", False)):
            try:
                mh_mz_text = f"{float(r.get('MH_XIC_mz')):.5f}"
            except Exception:
                mh_mz_text = "-"
            try:
                fa_mz_text = f"{float(r.get('Formate_XIC_mz')):.5f}"
            except Exception:
                fa_mz_text = "-"
            subtitle = f"[M-H]- m/z={mh_mz_text}; [M+HCOO]- m/z={fa_mz_text}; ppm=\u00B1{float(ppm):g}"
            if peak_cnt > 1:
                subtitle += f"  (peak {peak_idx}/{peak_cnt})"
        else:
            if peak_cnt > 1:
                subtitle = f"mz={mz0:.5f}  ppm=\u00B1{float(ppm):g}  {adduct}  (peak {peak_idx}/{peak_cnt})".strip()
            else:
                subtitle = f"mz={mz0:.5f}  ppm=\u00B1{float(ppm):g}  {adduct}".strip()

        ratio_mode = str(r.get("Ratio_mode") or "")
        try:
            ratio = float(r.get("Ratio_%", 0.0) or 0.0)
            ratio_text = (
                f"Ratio(vs IS)={ratio:.2f}%"
                if ratio_mode in {"vs_last_internal_standard", "vs_internal_standard"}
                else f"Ratio={ratio:.2f}%"
            )
        except Exception:
            ratio_text = "Ratio=NA"

        if channel_panel is not None:
            panel_subtitle = (
                f"Negative-ion panel={channel_panel.panel_name}; ppm=±{float(ppm):g}; "
                f"mode={channel_mode}"
            )
            save_multi_channel_xic_png(
                r,
                out_png=out_png,
                title=title,
                subtitle=panel_subtitle,
                ratio_text=ratio_text,
            )
        elif bool(r.get("Dual_Adduct_Enabled", False)):
            rt_mh = r.get("_rt_mh")
            int_mh = r.get("_inten_mh")
            rt_fa = r.get("_rt_formate")
            int_fa = r.get("_inten_formate")
            if not isinstance(rt_mh, np.ndarray):
                rt_mh = np.asarray([], dtype=float)
            if not isinstance(int_mh, np.ndarray):
                int_mh = np.asarray([], dtype=float)
            if not isinstance(rt_fa, np.ndarray):
                rt_fa = np.asarray([], dtype=float)
            if not isinstance(int_fa, np.ndarray):
                int_fa = np.asarray([], dtype=float)
            save_dual_xic_png(
                rt_mh,
                int_mh,
                rt_fa,
                int_fa,
                dual if isinstance(dual, DualAdductPeakResult) else None,
                out_png=out_png,
                title=title,
                subtitle=subtitle,
                ratio_text=ratio_text,
            )
        else:
            save_xic_png(
                rt,
                inten,
                peak if isinstance(peak, PeakResult) else None,
                out_png=out_png,
                title=title,
                subtitle=subtitle,
                ratio_text=ratio_text,
            )

        r["XIC_PNG"] = str(out_png)

    if is_diag_rows:
        is_diag_fields = [
            "No", "Name", "Formula", "Polarity", "Matched_Adduct", "Adduct",
            "Theoretical_mz", "XIC_mz", "Found", "Apex_RT", "RT_start", "RT_end",
            "Peak_height", "Area", "IS_Dedicated_Adduct", "IS_PPM", "IS_Diagnostic_Status",
            "IS_Accepted_By", "IS_Candidate_Apex_RT", "IS_Candidate_RT_Start",
            "IS_Candidate_RT_End", "IS_Candidate_Area", "IS_Candidate_Raw_Apex",
            "IS_Candidate_Baseline", "IS_Candidate_Net_Height", "IS_Noise_Sigma_Proxy",
            "IS_SNR_Proxy", "IS_Global_Min_Peak_Height", "IS_Adaptive_Min_Height",
            "IS_Expected_RT", "IS_Expected_RT_Tolerance_Min", "IS_Diagnostic_Note",
            "MH_Found", "MH_Area", "Formate_Found", "Formate_Accepted", "Formate_Area",
            "Adduct_Channels_Used", "XIC_PNG", "IS_Diagnostic_PNG",
            "IS_XIC_Plot_In_Plots_Folder", "IS_Trace_CSV", "IS_All_Channels_PNG",
        ]
        with is_diag_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=is_diag_fields, extrasaction="ignore")
            w.writeheader()
            for rr in is_diag_rows:
                w.writerow({k: rr.get(k, "") for k in is_diag_fields})

    
    out_csv = out_dir / f"{sample}__XIC_quant.csv"
    fieldnames = [
        "No",
        "Name",
        "Formula",
        "Combo",
        "Is_internal_standard",
        "IS_Dedicated_Adduct",
        "IS_PPM",
        "IS_Diagnostic_Status",
        "IS_Accepted_By",
        "IS_Candidate_Apex_RT",
        "IS_Candidate_RT_Start",
        "IS_Candidate_RT_End",
        "IS_Candidate_Area",
        "IS_Candidate_Raw_Apex",
        "IS_Candidate_Baseline",
        "IS_Candidate_Net_Height",
        "IS_Noise_Sigma_Proxy",
        "IS_SNR_Proxy",
        "IS_Global_Min_Peak_Height",
        "IS_Adaptive_Min_Height",
        "IS_Expected_RT",
        "IS_Expected_RT_Tolerance_Min",
        "IS_Diagnostic_Note",
        "IS_Diagnostic_PNG",
        "IS_XIC_Plot_In_Plots_Folder",
        "IS_Diagnostic_CSV",
        "IS_Trace_CSV",
        "IS_All_Channels_PNG",
        "Polarity",
        "Matched_Adduct",
        "Adduct",
        "Theoretical_mz",
        "XIC_mz",
        "MH_Theoretical_mz",
        "MH_XIC_mz",
        "MH_Found",
        "MH_Area",
        "MH_Peak_height",
        "MH_Apex_RT",
        "Formate_Theoretical_mz",
        "Formate_XIC_mz",
        "Formate_Found",
        "Formate_Accepted",
        "Formate_Area",
        "Formate_Peak_height",
        "Formate_Apex_RT",
        "Formate_RT_Delta_min",
        "Formate_Shape_Correlation",
        "Formate_Area_Fraction_%",
        "Adduct_Channels_Used",
        "Formate_Note",
        "Dual_Adduct_Enabled",
        "Dual_Adduct_Note",
        "Negative_Channel_Mode",
        "Negative_Channel_Panel",
        "Quant_Area_Before_Panel",
        "Quant_Area_After_Panel",
        "Accepted_Channels",
        "Accepted_Channel_Count",
        "Summed_Channel_Count",
        "Additional_Summed_Area",
        "Adduct_Cluster_Area",
        "Fragment_Diagnostic_Area",
        "Ion_Form_Diversity_Count",
        "Observed_Fragility_Index",
        "Adduct_Cluster_Proneness_Index",
        "Primary_Ion_Fraction",
        "Br_Fragment_Fraction",
        "Negative_Ion_Panel",
        "PPM",
        "MS_filter",
        "PeakIndex",
        "PeakCount",
        "Apex_RT",
        "RT_start",
        "RT_end",
        "Peak_height",
        "Area",
        "Ratio_%",
        "Ratio_mode",
        "Internal_standard",
        "Internal_area",
        "Internal_standard_found",
        "Found",
        "XIC_PNG",
        "MS2_Found",
        "MS2_Scan",
        "MS2_RT",
        "MS2_Precursor_mz",
        "MS2_Precursor_ppm",
        "MS2_Target_Adduct",
        "MS2_PNG",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row_out)

    # Long-form evidence table for discovery/panel audit.
    if channel_panel is not None:
        evidence_csv = out_dir / f"{sample}__negative_ion_channels.csv"
        evidence_fields = [
            "No", "Name", "Formula", "Combo", "Channel_ID", "Channel", "Category",
            "Quant_Action", "Eligible", "Accepted", "High_Confidence", "Theoretical_mz",
            "XIC_mz", "Apex_RT", "RT_Delta_Min", "Shape_Correlation", "Peak_Height",
            "Area", "Area_Fraction_vs_MH_Pct", "Isotope_Support_OK", "Support_Area",
            "Support_Ratio", "Summed_Into_Quant_Area", "Reason",
        ]
        with evidence_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=evidence_fields, extrasaction="ignore")
            w.writeheader()
            for ev in channel_evidence:
                w.writerow({k: ev.get(k, "") for k in evidence_fields})

    return out_csv, plots_dir
