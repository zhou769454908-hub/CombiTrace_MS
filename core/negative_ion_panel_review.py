"""Integrated review and cached application of negative-ion discovery results.

This module supports the streamlined triplicate workflow:

1. Run triplicate XIC in ``discover`` mode once.
2. Aggregate the generated ``*__negative_ion_channels.csv`` evidence files.
3. Review the channel-level evidence and save a fixed experiment panel.
4. Apply the selected panel to the already generated discovery CSV files without
   reading Thermo RAW files again.

Cached application can only use chromatographic peaks that were already accepted
and written by the discovery run.  It is therefore suitable for changing channel
inclusion/actions and for making thresholds stricter.  A more permissive peak-
detection threshold still requires a RAW re-run because rejected traces/peaks are
not fully represented in the cached CSV files.
"""
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .excel_sheet_xic import (
    RawResolution,
    expected_raw_stems,
    load_excel_sheet_plans,
    normalize_internal_standard_formula,
    read_quant_csv,
    safe_sheet_folder_name,
    write_parallel_summary_xlsx,
)
from .negative_ion_channels import (
    CHANNEL_BY_ID,
    NegativeIonPanel,
    PanelChannel,
    load_panel,
    normalise_action,
    write_panel_csv,
)
from .negative_ion_discovery import aggregate_channel_evidence


Progress = Optional[Callable[[str], None]]


@dataclass
class CachedPanelApplyReport:
    panel_file: Path
    output_root: Path
    summary_xlsx: Path
    average_csv: Path
    audit_csv: Path
    updated_quant_files: List[Path] = field(default_factory=list)
    updated_evidence_files: List[Path] = field(default_factory=list)
    missing_quant_files: List[str] = field(default_factory=list)
    missing_evidence_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def processed_raws(self) -> int:
        return len(self.updated_quant_files)


def _log(progress: Progress, message: str) -> None:
    if progress is not None:
        progress(str(message))


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "accepted", "on"}


def _float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _optional_float(value: object) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _normalised_stem(text: object) -> str:
    s = str(text or "").strip().lower()
    for suffix in (".raw", "__xic_quant", "__negative_ion_channels"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    out: List[str] = []
    dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            dash = False
        else:
            if not dash:
                out.append("-")
                dash = True
    return "".join(out).strip("-")


def _sample_from_quant_path(path: Path) -> str:
    name = Path(path).name
    marker = "__XIC_quant"
    pos = name.lower().find(marker.lower())
    return name[:pos] if pos >= 0 else Path(path).stem


def _sample_from_evidence_path(path: Path) -> str:
    name = Path(path).name
    marker = "__negative_ion_channels"
    pos = name.lower().find(marker.lower())
    return name[:pos] if pos >= 0 else Path(path).stem


def collect_discovery_evidence_files(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    result: List[Path] = []
    for p in root.rglob("*__negative_ion_channels.csv"):
        low_parts = {x.lower() for x in p.parts}
        if "selected_panel_results" in low_parts or "panel_applied_results" in low_parts:
            continue
        result.append(p)
    return sorted(result, key=lambda x: str(x).lower())


def collect_discovery_quant_files(root: Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    result: List[Path] = []
    for p in root.rglob("*__XIC_quant.csv"):
        low_parts = {x.lower() for x in p.parts}
        if "selected_panel_results" in low_parts or "panel_applied_results" in low_parts:
            continue
        result.append(p)
    return sorted(result, key=lambda x: str(x).lower())


def refresh_discovery_summary(
    discovery_root: Path,
    *,
    prevalence_threshold_pct: float = 20.0,
    min_isotope_support_pct: float = 50.0,
    min_median_area_fraction_pct: float = 0.5,
    summary_xlsx: Optional[Path] = None,
    suggested_panel_csv: Optional[Path] = None,
) -> Tuple[Path, Path, List[Dict[str, object]], List[Path]]:
    """Aggregate all discovery evidence below *discovery_root*.

    The suggested panel is deliberately separate from the user-selected panel so
    refreshing the summary never overwrites prior manual choices.
    """
    root = Path(discovery_root)
    evidence = collect_discovery_evidence_files(root)
    if not evidence:
        raise FileNotFoundError(
            "No '*__negative_ion_channels.csv' files were found. Run the triplicate XIC "
            "once with 'Discover all negative-ion candidates' first."
        )
    summary_xlsx = Path(summary_xlsx or (root / "negative_ion_discovery_summary.xlsx"))
    suggested_panel_csv = Path(suggested_panel_csv or (root / "negative_ion_suggested_panel.csv"))
    out_xlsx, out_panel, summary = aggregate_channel_evidence(
        evidence,
        out_xlsx=summary_xlsx,
        out_panel_csv=suggested_panel_csv,
        prevalence_threshold_pct=float(prevalence_threshold_pct),
        min_isotope_support_pct=float(min_isotope_support_pct),
        min_median_area_fraction_pct=float(min_median_area_fraction_pct),
        panel_name="selected_negative_ion_experiment_panel",
    )
    return out_xlsx, out_panel, summary, evidence


def _read_csv_with_fields(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(x) for x in (reader.fieldnames or [])]
        rows = [{str(k): ("" if v is None else str(v)) for k, v in row.items()} for row in reader]
    return fields, rows


def _write_csv_with_fields(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(dict.fromkeys(str(x) for x in fields))
    for row in rows:
        for key in row.keys():
            if str(key).startswith("_"):
                continue
            if str(key) not in ordered:
                ordered.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered})
    return path


def load_panel_review_rows(panel_csv: Path) -> List[Dict[str, str]]:
    _, rows = _read_csv_with_fields(Path(panel_csv))
    return rows


def save_panel_review_rows(
    rows: Sequence[Mapping[str, object]],
    out_panel_csv: Path,
    *,
    panel_name: str = "selected_negative_ion_experiment_panel",
) -> Path:
    """Save edited review rows as a valid fixed-panel CSV while retaining summary columns."""
    channels: List[PanelChannel] = []
    summary: Dict[str, Dict[str, object]] = {}
    for item in rows:
        cid = str(item.get("Channel_ID", "") or "").strip().upper()
        defn = CHANNEL_BY_ID.get(cid)
        if defn is None or cid == "MH" or defn.category == "isotope_support":
            continue
        action = normalise_action(item.get("Quant_Action", ""), defn)
        enabled_text = str(item.get("Enabled", "true") or "true").strip().lower()
        enabled = enabled_text not in {"0", "false", "no", "n", "off", "exclude"} and action != "exclude"
        channels.append(PanelChannel(
            channel_id=cid,
            enabled=enabled,
            quant_action=action,
            min_rel_height_pct=_float(item.get("Min_Rel_Height_Pct", 1.0), 1.0),
            min_shape_correlation=_float(item.get("Min_Shape_Correlation", 0.30), 0.30),
            rt_tolerance_min=_float(item.get("RT_Tolerance_Min", 0.10), 0.10),
            note=str(item.get("Panel_Note", item.get("Note", "")) or ""),
        ))
        summary[cid] = {
            "Eligible_Count": item.get("Eligible_Count", ""),
            "Detected_Count": item.get("Detected_Count", ""),
            "Detection_Rate_Pct": item.get("Detection_Rate_Pct", ""),
            "Median_Area_Fraction_vs_MH_Pct": item.get("Median_Area_Fraction_vs_MH_Pct", ""),
            "Median_RT_Delta_Min": item.get("Median_RT_Delta_Min", ""),
            "Median_Shape_Correlation": item.get("Median_Shape_Correlation", ""),
            "Isotope_Support_Rate_Pct": item.get("Isotope_Support_Rate_Pct", ""),
            "Suggested_Status": item.get("Suggested_Status", ""),
        }
    panel = NegativeIonPanel(channels, panel_name=panel_name, require_primary_mh=True, version="1.1")
    return write_panel_csv(panel, Path(out_panel_csv), summary=summary)


def merge_existing_actions(
    suggested_rows: Sequence[Mapping[str, object]],
    existing_panel_csv: Optional[Path],
) -> List[Dict[str, str]]:
    """Overlay previously saved actions/thresholds on a refreshed discovery summary."""
    previous: Dict[str, Dict[str, str]] = {}
    if existing_panel_csv and Path(existing_panel_csv).exists():
        for row in load_panel_review_rows(Path(existing_panel_csv)):
            cid = str(row.get("Channel_ID", "") or "").strip().upper()
            if cid:
                previous[cid] = row
    merged: List[Dict[str, str]] = []
    for raw in suggested_rows:
        row = {str(k): ("" if v is None else str(v)) for k, v in raw.items()}
        cid = str(row.get("Channel_ID", "") or "").strip().upper()
        old = previous.get(cid)
        if old:
            for key in (
                "Enabled", "Quant_Action", "Min_Rel_Height_Pct",
                "Min_Shape_Correlation", "RT_Tolerance_Min", "Panel_Note",
            ):
                if key in old and str(old.get(key, "")).strip() != "":
                    row[key] = old[key]
        row.setdefault("Auto_Suggested_Action", str(raw.get("Quant_Action", "") or ""))
        merged.append(row)
    return merged


def _evidence_key(row: Mapping[str, object]) -> str:
    no = str(row.get("No", "") or "").strip()
    try:
        return str(int(float(no)))
    except Exception:
        # Fallback is only used for malformed/legacy files.
        return "|".join([
            str(row.get("Formula", "") or "").strip(),
            str(row.get("Combo", "") or "").strip(),
            str(row.get("Name", "") or "").strip(),
        ])


def _cached_acceptance(
    ev: Mapping[str, object],
    quant_row: Mapping[str, object],
    cfg: PanelChannel,
) -> Tuple[bool, str]:
    if not _truthy(ev.get("Eligible", "")):
        return False, "not eligible"
    if not _truthy(ev.get("Accepted", "")):
        return False, "not accepted in discovery cache"

    rt_delta = _optional_float(ev.get("RT_Delta_Min", ""))
    if rt_delta is not None and abs(rt_delta) > float(cfg.rt_tolerance_min):
        return False, f"RT delta {rt_delta:.4g} > {cfg.rt_tolerance_min:.4g} min"

    mh_height = _float(quant_row.get("MH_Peak_height", quant_row.get("Peak_height", 0.0)), 0.0)
    peak_height = _float(ev.get("Peak_Height", 0.0), 0.0)
    if mh_height > 0:
        rel = peak_height / mh_height * 100.0
        if rel < float(cfg.min_rel_height_pct):
            return False, f"relative height {rel:.3g}% < {cfg.min_rel_height_pct:.3g}%"

    shape = _optional_float(ev.get("Shape_Correlation", ""))
    if float(cfg.min_shape_correlation) > -1 and shape is not None and shape < float(cfg.min_shape_correlation):
        return False, f"shape correlation {shape:.3g} < {cfg.min_shape_correlation:.3g}"

    return True, "accepted from discovery cache"


def apply_panel_to_quant_rows(
    quant_rows: Sequence[Mapping[str, object]],
    evidence_rows: Sequence[Mapping[str, object]],
    *,
    panel: NegativeIonPanel,
    include_for_internal_standard: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Recalculate quantitative areas from cached discovery evidence.

    Only channel peaks already accepted in the discovery file can be used.  The
    function does not read RAW data and does not regenerate chromatogram PNGs.
    """
    panel_map = panel.by_id()
    ev_by_key: Dict[str, List[Mapping[str, object]]] = {}
    for ev in evidence_rows:
        ev_by_key.setdefault(_evidence_key(ev), []).append(ev)

    out: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    for original in quant_rows:
        row: Dict[str, object] = dict(original)
        key = _evidence_key(row)
        mh_area = _float(row.get("MH_Area", row.get("Quant_Area_Before_Panel", row.get("Area", 0.0))), 0.0)
        mh_height = _float(row.get("MH_Peak_height", row.get("Peak_height", 0.0)), 0.0)
        mh_found = _truthy(row.get("MH_Found", "")) or (mh_area > 0 and _truthy(row.get("Found", "")))
        is_internal = _truthy(row.get("Is_internal_standard", ""))

        quant_area = mh_area if mh_found else 0.0
        accepted_channels: List[str] = ["[M-H]-"] if mh_found else []
        summed_channels: List[str] = []
        accepted_count = 0
        summed_count = 0
        adduct_cluster_area = 0.0
        fragment_area = 0.0
        br_fragment_area = 0.0

        for ev in ev_by_key.get(key, []):
            cid = str(ev.get("Channel_ID", "") or "").strip().upper()
            cfg = panel_map.get(cid)
            defn = CHANNEL_BY_ID.get(cid)
            if cfg is None or defn is None or not cfg.enabled:
                continue
            action = normalise_action(cfg.quant_action, defn)
            if action == "exclude":
                continue
            accepted, accept_note = _cached_acceptance(ev, row, cfg)
            support_required = bool(defn.support_channel_id)
            support_ok = (not support_required) or _truthy(ev.get("Isotope_Support_OK", ""))
            area = _float(ev.get("Area", 0.0), 0.0) if accepted else 0.0
            allow_sum = bool(include_for_internal_standard or not is_internal)
            can_sum = bool(
                accepted and support_ok and action == "sum" and defn.quantifiable and allow_sum
            )

            if accepted:
                accepted_count += 1
                accepted_channels.append(defn.display)
                if defn.category in {"adduct", "cluster", "substitution", "dimer"}:
                    adduct_cluster_area += area
                if defn.category in {"fragment", "transformation", "diagnostic"}:
                    fragment_area += area
                if cid in {"BR_LOSS_FROM_MH", "BR_ANION79"}:
                    br_fragment_area += area
            if can_sum:
                quant_area += area
                summed_count += 1
                summed_channels.append(defn.display)

            audit.append({
                "No": row.get("No", ""),
                "Name": row.get("Name", ""),
                "Formula": row.get("Formula", ""),
                "Combo": row.get("Combo", ""),
                "Is_Internal_Standard": is_internal,
                "Channel_ID": cid,
                "Channel": defn.display,
                "Category": defn.category,
                "Selected_Action": action,
                "Discovery_Accepted": _truthy(ev.get("Accepted", "")),
                "Accepted_After_Selected_Panel": accepted,
                "Isotope_Support_OK": ev.get("Isotope_Support_OK", ""),
                "Summed_Into_Area": can_sum,
                "Channel_Area": area,
                "Cached_Application_Note": accept_note if accepted else accept_note,
            })

        total_evidence = mh_area + adduct_cluster_area + fragment_area
        row["Found"] = bool(mh_found)
        row["Area"] = float(quant_area)
        row["Quant_Area_Before_Panel"] = float(mh_area)
        row["Quant_Area_After_Panel"] = float(quant_area)
        row["Additional_Summed_Area"] = float(max(0.0, quant_area - mh_area))
        row["Accepted_Channels"] = " | ".join(accepted_channels)
        row["Accepted_Channel_Count"] = int(accepted_count)
        row["Summed_Channel_Count"] = int(summed_count)
        row["Adduct_Cluster_Area"] = float(adduct_cluster_area)
        row["Fragment_Diagnostic_Area"] = float(fragment_area)
        row["Ion_Form_Diversity_Count"] = int((1 if mh_found else 0) + accepted_count)
        row["Observed_Fragility_Index"] = float(fragment_area / total_evidence) if total_evidence > 0 else 0.0
        row["Adduct_Cluster_Proneness_Index"] = float(adduct_cluster_area / total_evidence) if total_evidence > 0 else 0.0
        row["Primary_Ion_Fraction"] = float(mh_area / total_evidence) if total_evidence > 0 else 0.0
        row["Br_Fragment_Fraction"] = float(br_fragment_area / total_evidence) if total_evidence > 0 else 0.0
        row["Negative_Channel_Mode"] = "panel_cached"
        row["Negative_Channel_Panel"] = panel.panel_name
        row["Negative_Ion_Panel"] = panel.panel_name
        row["Cached_Panel_Applied"] = True
        row["Cached_Panel_Source"] = "discover_all_evidence"
        row["Adduct_Channels_Used"] = "[M-H]-" + (
            " + " + " + ".join(summed_channels) if summed_channels else ""
        ) if mh_found else ""
        if mh_found:
            row["Peak_height"] = mh_height
            if _optional_float(row.get("MH_Apex_RT", "")) is not None:
                row["Apex_RT"] = row.get("MH_Apex_RT", row.get("Apex_RT", ""))
        out.append(row)

    # Recalculate target/IS or total-area ratios after the selected channels have
    # changed every row's quantitative area.
    internal_rows = [r for r in out if _truthy(r.get("Is_internal_standard", ""))]
    if internal_rows:
        internal = internal_rows[-1]
        internal_area = _float(internal.get("Area", 0.0), 0.0)
        internal_found = _truthy(internal.get("Found", "")) and internal_area > 0
        internal_name = str(internal.get("Name") or internal.get("Formula") or "")
        for row in out:
            area = _float(row.get("Area", 0.0), 0.0)
            row["Ratio_%"] = (area / internal_area * 100.0) if internal_found else ""
            row["Ratio_mode"] = "vs_internal_standard"
            row["Internal_standard"] = internal_name
            row["Internal_area"] = internal_area if internal_found else ""
            row["Internal_standard_found"] = bool(internal_found)
    else:
        total_area = sum(_float(r.get("Area", 0.0), 0.0) for r in out)
        for row in out:
            area = _float(row.get("Area", 0.0), 0.0)
            row["Ratio_%"] = area / total_area * 100.0 if total_area > 0 else 0.0
            row["Ratio_mode"] = "percent_of_total_area"
            row["Internal_standard"] = ""
            row["Internal_area"] = ""
            row["Internal_standard_found"] = ""

    return out, audit


def apply_panel_to_quant_csv(
    quant_csv: Path,
    evidence_csv: Path,
    *,
    panel: NegativeIonPanel,
    output_quant_csv: Path,
    output_evidence_audit_csv: Optional[Path] = None,
    include_for_internal_standard: bool = True,
) -> Tuple[Path, Optional[Path], List[Dict[str, object]]]:
    q_fields, q_rows = _read_csv_with_fields(Path(quant_csv))
    _, evidence_rows = _read_csv_with_fields(Path(evidence_csv))
    updated, audit = apply_panel_to_quant_rows(
        q_rows,
        evidence_rows,
        panel=panel,
        include_for_internal_standard=bool(include_for_internal_standard),
    )
    out_q = _write_csv_with_fields(Path(output_quant_csv), q_fields, updated)
    out_e: Optional[Path] = None
    if output_evidence_audit_csv is not None:
        audit_fields = [
            "No", "Name", "Formula", "Combo", "Is_Internal_Standard",
            "Channel_ID", "Channel", "Category", "Selected_Action",
            "Discovery_Accepted", "Accepted_After_Selected_Panel",
            "Isotope_Support_OK", "Summed_Into_Area", "Channel_Area",
            "Cached_Application_Note",
        ]
        out_e = _write_csv_with_fields(Path(output_evidence_audit_csv), audit_fields, audit)
    return out_q, out_e, audit


def _match_file_by_expected(
    expected_stem: str,
    files: Sequence[Path],
    *,
    kind: str,
    preferred_parent_name: str = "",
) -> Optional[Path]:
    getter = _sample_from_quant_path if kind == "quant" else _sample_from_evidence_path
    expected_lower = str(expected_stem).strip().lower()
    expected_norm = _normalised_stem(expected_stem)
    candidates: List[Tuple[int, str, Path]] = []
    for p in files:
        sample = getter(p)
        sample_lower = sample.lower()
        sample_norm = _normalised_stem(sample)
        score: Optional[int] = None
        if sample_lower == expected_lower:
            score = 0
        elif sample_norm == expected_norm:
            score = 1
        elif sample_norm.endswith("-" + expected_norm) or expected_norm.endswith("-" + sample_norm):
            score = 2
        if score is None:
            continue
        if preferred_parent_name and p.parent.name == preferred_parent_name:
            score -= 1
        candidates.append((score, str(p).lower(), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_score = candidates[0][0]
    best = [x[2] for x in candidates if x[0] == best_score]
    return best[0] if len(best) == 1 else best[0]


def _infer_internal_standard_formula(run_records: Sequence[Mapping[str, object]]) -> str:
    for rec in run_records:
        quant_rows = rec.get("quant_rows", {})
        if not isinstance(quant_rows, Mapping):
            continue
        for rows in quant_rows.values():
            if not isinstance(rows, Sequence):
                continue
            for row in rows:
                if isinstance(row, Mapping) and _truthy(row.get("Is_internal_standard", "")):
                    formula = str(row.get("Formula", "") or "").strip()
                    if formula:
                        return formula
    return ""


def apply_panel_to_existing_triplicate(
    *,
    workbook_path: Path,
    discovery_root: Path,
    selected_panel_file: Path,
    output_root: Path,
    suffixes: Sequence[str] = ("-1", "-2", "-3"),
    layout: str = "auto",
    only_formula: bool = False,
    internal_standard_formula: str = "",
    include_for_internal_standard: bool = True,
    progress: Progress = None,
) -> CachedPanelApplyReport:
    """Apply a selected panel to an existing triplicate discover-all result folder.

    This operation does not open RAW files.  It produces new per-RAW quantitative
    CSVs and a new triplicate summary workbook under *output_root*.
    """
    workbook_path = Path(workbook_path)
    discovery_root = Path(discovery_root)
    selected_panel_file = Path(selected_panel_file)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    panel = load_panel(selected_panel_file)
    quant_files = collect_discovery_quant_files(discovery_root)
    evidence_files = collect_discovery_evidence_files(discovery_root)
    if not quant_files:
        raise FileNotFoundError("No cached '*__XIC_quant.csv' files were found in the discovery output folder.")
    if not evidence_files:
        raise FileNotFoundError("No cached '*__negative_ion_channels.csv' files were found in the discovery output folder.")

    plans, warnings = load_excel_sheet_plans(
        workbook_path,
        formula_columns=(1, 3, 5),
        header_row=1,
        data_start_row=2,
        only_formula=bool(only_formula),
        layout=str(layout or "auto"),
    )
    if not plans:
        raise RuntimeError("No valid A/B/C worksheet plans could be reconstructed from the workbook.")

    report = CachedPanelApplyReport(
        panel_file=selected_panel_file,
        output_root=output_root,
        summary_xlsx=output_root / "00_triplicate_selected_panel_summary.xlsx",
        average_csv=output_root / "00_triplicate_selected_panel_average.csv",
        audit_csv=output_root / "00_selected_panel_application_audit.csv",
        warnings=list(warnings),
    )
    run_records: List[Dict[str, object]] = []
    combined_audit: List[Dict[str, object]] = []

    for plan_idx, plan in enumerate(plans, start=1):
        _log(progress, f"[{plan_idx}/{len(plans)}] Apply selected panel to sheet: {plan.sheet_name}")
        quant_rows_by_rep: Dict[str, List[Dict[str, str]]] = {}
        resolutions: List[RawResolution] = []
        notes: List[str] = []
        parent_name = safe_sheet_folder_name(plan.sheet_name)
        out_sheet = output_root / parent_name
        out_sheet.mkdir(parents=True, exist_ok=True)

        for label, expected in expected_raw_stems(plan.sheet_name, suffixes):
            qsrc = _match_file_by_expected(expected, quant_files, kind="quant", preferred_parent_name=parent_name)
            esrc = _match_file_by_expected(expected, evidence_files, kind="evidence", preferred_parent_name=parent_name)
            actual_sample = _sample_from_quant_path(qsrc) if qsrc is not None else expected
            if qsrc is None:
                report.missing_quant_files.append(expected)
                notes.append(f"{label}: cached quant missing")
                resolutions.append(RawResolution(label, expected, None, "missing", "cached quant CSV missing"))
                _log(progress, f"  Replicate {label}: missing quant CSV for {expected}")
                continue
            if esrc is None:
                report.missing_evidence_files.append(expected)
                notes.append(f"{label}: discovery evidence missing")
                resolutions.append(RawResolution(label, expected, None, "missing", "discovery evidence CSV missing"))
                _log(progress, f"  Replicate {label}: missing discovery evidence for {expected}")
                continue

            out_q = out_sheet / f"{actual_sample}__XIC_quant_selected.csv"
            out_e = out_sheet / f"{actual_sample}__negative_ion_channels_selected.csv"
            out_q, out_e_written, audit = apply_panel_to_quant_csv(
                qsrc,
                esrc,
                panel=panel,
                output_quant_csv=out_q,
                output_evidence_audit_csv=out_e,
                include_for_internal_standard=bool(include_for_internal_standard),
            )
            report.updated_quant_files.append(out_q)
            if out_e_written is not None:
                report.updated_evidence_files.append(out_e_written)
            for row in audit:
                row["Sheet"] = plan.sheet_name
                row["Replicate"] = label
                row["Source_Quant_CSV"] = str(qsrc)
                row["Source_Evidence_CSV"] = str(esrc)
                combined_audit.append(row)
            quant_rows_by_rep[label] = read_quant_csv(out_q)
            resolutions.append(
                RawResolution(label, expected, Path(actual_sample + ".RAW"), "cached_panel", str(qsrc))
            )
            _log(progress, f"  Replicate {label}: updated {actual_sample}")

        run_records.append({
            "plan": plan,
            "resolutions": resolutions,
            "quant_rows": quant_rows_by_rep,
            "note": "; ".join(notes),
        })

    is_formula = str(internal_standard_formula or "").strip()
    if not is_formula:
        is_formula = _infer_internal_standard_formula(run_records)
    if is_formula:
        try:
            is_formula, _ = normalize_internal_standard_formula(is_formula)
        except Exception as exc:
            report.warnings.append(f"Internal-standard formula audit failed: {exc}")

    write_parallel_summary_xlsx(
        report.summary_xlsx,
        run_records,
        internal_standard_formula=is_formula,
        average_csv_path=report.average_csv,
    )
    audit_fields = [
        "Sheet", "Replicate", "No", "Name", "Formula", "Combo",
        "Is_Internal_Standard", "Channel_ID", "Channel", "Category",
        "Selected_Action", "Discovery_Accepted", "Accepted_After_Selected_Panel",
        "Isotope_Support_OK", "Summed_Into_Area", "Channel_Area",
        "Cached_Application_Note", "Source_Quant_CSV", "Source_Evidence_CSV",
    ]
    _write_csv_with_fields(report.audit_csv, audit_fields, combined_audit)
    shutil.copy2(selected_panel_file, output_root / "selected_negative_ion_panel.csv")
    _log(progress, f"Cached panel application complete: {report.summary_xlsx}")
    return report
