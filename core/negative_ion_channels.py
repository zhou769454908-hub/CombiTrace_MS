"""Negative-ESI ion-channel registry, panel I/O, and exact m/z calculations.

The registry separates alternate 1:1 ion forms, clusters/dimers, and in-source
fragments.  Only channels explicitly marked ``sum`` in a fixed experiment panel
are added to the quantitative area.  Dimer/fragment/diagnostic channels are
kept as evidence by default.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .chemistry import (
    ELECTRON_MASS_U,
    MASS_H_PLUS,
    MASS_K_PLUS,
    MASS_NA_PLUS,
    MONO_MASS,
    parse_formula,
)

MASS_37CL = 36.965902602
MASS_81BR = 80.9162897
H = MONO_MASS["H"]
C = MONO_MASS["C"]
N = MONO_MASS["N"]
O = MONO_MASS["O"]
P = MONO_MASS["P"]
S = MONO_MASS["S"]
CL35 = MONO_MASS["Cl"]
BR79 = MONO_MASS["Br"]

MASS_H2O = 2.0 * H + O
MASS_FORMIC_ACID = C + 2.0 * H + 2.0 * O
MASS_ACETIC_ACID = 2.0 * C + 4.0 * H + 2.0 * O
MASS_CO2 = C + 2.0 * O
MASS_SO3 = S + 3.0 * O
MASS_H3PO3 = 3.0 * H + P + 3.0 * O
MASS_HNO3 = H + N + 3.0 * O
MASS_H35CL = H + CL35
MASS_H37CL = H + MASS_37CL
MASS_H3PO4 = 3.0 * H + P + 4.0 * O
MASS_NA_REPLACE_H = MASS_NA_PLUS - MASS_H_PLUS
MASS_K_REPLACE_H = MASS_K_PLUS - MASS_H_PLUS
MASS_FORMATE_2H2O = MASS_FORMIC_ACID + 2.0 * MASS_H2O
MASS_CH2 = C + 2.0 * H
MASS_C2H4 = 2.0 * C + 4.0 * H
MASS_H79BR = H + BR79
MASS_H81BR = H + MASS_81BR
MASS_BR79_ANION = BR79 + ELECTRON_MASS_U
MASS_BR81_ANION = MASS_81BR + ELECTRON_MASS_U


@dataclass(frozen=True)
class IonChannelDefinition:
    channel_id: str
    display: str
    category: str
    mz_kind: str
    exact_delta_from_mh: Optional[float]
    source: str
    priority_stars: int
    default_action: str = "search"  # primary | sum | search | evidence | exclude
    quantifiable: bool = True
    formula_element_required: str = ""
    support_channel_id: str = ""
    expected_support_ratio: Optional[float] = None
    support_ratio_low: Optional[float] = None
    support_ratio_high: Optional[float] = None
    note: str = ""


CHANNELS: Tuple[IonChannelDefinition, ...] = (
    IonChannelDefinition("MH", "[M-H]-", "primary", "mh", 0.0,
                         "deprotonated molecule", 5, "primary", True),
    IonChannelDefinition("FORMATE", "[M+HCOO]-", "adduct", "delta_mh", MASS_FORMIC_ACID,
                         "formic acid / formate in mobile phase or matrix", 5, "sum", True),
    IonChannelDefinition("ACETATE", "[M+CH3COO]-", "adduct", "delta_mh", MASS_ACETIC_ACID,
                         "acetic acid / acetate buffer or contamination", 5, "search", True),
    IonChannelDefinition("CO2_CLUSTER", "[M+CO2-H]-", "cluster", "delta_mh", MASS_CO2,
                         "CO2 / carbonate background", 3, "search", True,
                         note="Exact +43.989829 Da relative to [M-H]-; nominally +44."),
    IonChannelDefinition("SO3_CLUSTER", "[M+SO3-H]-", "cluster", "delta_mh", MASS_SO3,
                         "sulfite / sulfate-related residue", 3, "search", True,
                         note="Exact +79.956815 Da relative to [M-H]-; one interpretation of nominal +80."),
    IonChannelDefinition("PHOSPHITE", "[M+H2PO3]-", "adduct", "delta_mh", MASS_H3PO3,
                         "phosphite / phosphorus-containing residue", 3, "search", True,
                         note="Exact +81.981981 Da relative to [M-H]-; separated from SO3."),
    IonChannelDefinition("NITRATE", "[M+NO3]-", "adduct", "delta_mh", MASS_HNO3,
                         "nitric acid / nitrate residue", 3, "search", True),
    IonChannelDefinition("CHLORIDE35", "[M+35Cl]-", "adduct", "delta_mh", MASS_H35CL,
                         "chloride / hydrochloric acid / salts", 4, "search", True,
                         support_channel_id="CHLORIDE37", expected_support_ratio=3.13,
                         support_ratio_low=1.2, support_ratio_high=7.0,
                         note="37Cl is used as isotopic support."),
    IonChannelDefinition("CHLORIDE37", "[M+37Cl]- support", "isotope_support", "delta_mh", MASS_H37CL,
                         "37Cl isotope support", 4, "evidence", False),
    IonChannelDefinition("PHOSPHATE", "[M+H2PO4]-", "adduct", "delta_mh", MASS_H3PO4,
                         "phosphate buffer / phosphate residue", 3, "search", True),
    IonChannelDefinition("NA_REPLACE", "[M+Na-2H]-", "substitution", "delta_mh", MASS_NA_REPLACE_H,
                         "sodium contamination / glassware / solvent", 3, "search", True),
    IonChannelDefinition("K_REPLACE", "[M+K-2H]-", "substitution", "delta_mh", MASS_K_REPLACE_H,
                         "potassium contamination / salts", 2, "search", True),
    IonChannelDefinition("FORMATE_2H2O", "[M+HCOO+2H2O]-", "cluster", "delta_mh", MASS_FORMATE_2H2O,
                         "water-rich mobile phase / low source temperature", 2, "search", True),
    IonChannelDefinition("DIMER", "[2M-H]-", "dimer", "dimer", None,
                         "high analyte concentration / low source temperature", 2, "evidence", False,
                         note="Evidence only; dimer response is not summed into monomer area by default."),
    IonChannelDefinition("NET_METHYL", "[M+CH2-H]- (net +14 hypothesis)", "transformation", "delta_mh", MASS_CH2,
                         "rare in-source methylation / line chemistry", 1, "evidence", False),
    IonChannelDefinition("NET_ETHYL", "[M+C2H4-H]- (net +28 hypothesis)", "transformation", "delta_mh", MASS_C2H4,
                         "rare in-source ethylation / line chemistry", 1, "evidence", False),
    IonChannelDefinition("BROMIDE_ADDUCT79", "[M+79Br]-", "adduct", "delta_mh", MASS_H79BR,
                         "bromide in matrix / salts", 2, "search", True,
                         support_channel_id="BROMIDE_ADDUCT81", expected_support_ratio=1.03,
                         support_ratio_low=0.45, support_ratio_high=2.2),
    IonChannelDefinition("BROMIDE_ADDUCT81", "[M+81Br]- support", "isotope_support", "delta_mh", MASS_H81BR,
                         "81Br isotope support", 2, "evidence", False),
    IonChannelDefinition("BR_LOSS_FROM_MH", "[M-H-Br]-", "fragment", "loss_br_from_mh", -BR79,
                         "in-source C-Br cleavage", 4, "evidence", False, formula_element_required="Br",
                         note="Target-specific debrominated fragment; not summed."),
    IonChannelDefinition("HBR_LOSS_FROM_MH", "[M-H-HBr]-", "fragment", "loss_hbr_from_mh", -MASS_H79BR,
                         "in-source neutral HBr loss", 3, "evidence", False, formula_element_required="Br",
                         note="Alternative bromine-loss hypothesis; evidence only and never summed by default."),
    IonChannelDefinition("BR_ANION79", "79Br- diagnostic", "diagnostic", "fixed", None,
                         "in-source fragmentation of organobromine compounds", 5, "evidence", False,
                         formula_element_required="Br", support_channel_id="BR_ANION81",
                         expected_support_ratio=1.03, support_ratio_low=0.45, support_ratio_high=2.2,
                         note="Fixed m/z 78.918886; coeluting 81Br- supports bromine-specific fragmentation."),
    IonChannelDefinition("BR_ANION81", "81Br- diagnostic support", "isotope_support", "fixed", None,
                         "81Br isotope support", 5, "evidence", False, formula_element_required="Br"),
)

CHANNEL_BY_ID: Dict[str, IonChannelDefinition] = {c.channel_id: c for c in CHANNELS}
PUBLIC_CHANNEL_IDS: Tuple[str, ...] = tuple(
    c.channel_id for c in CHANNELS if c.category != "isotope_support" and c.channel_id != "MH"
)


@dataclass
class PanelChannel:
    channel_id: str
    enabled: bool = True
    quant_action: str = "search"  # sum | search | evidence | exclude
    min_rel_height_pct: float = 1.0
    min_shape_correlation: float = 0.30
    rt_tolerance_min: float = 0.10
    note: str = ""


@dataclass
class NegativeIonPanel:
    channels: List[PanelChannel]
    panel_name: str = "negative_ion_panel"
    require_primary_mh: bool = True
    version: str = "1.0"

    def by_id(self) -> Dict[str, PanelChannel]:
        return {c.channel_id: c for c in self.channels}


def exact_delta_from_mh_text(defn: IonChannelDefinition) -> str:
    if defn.mz_kind == "dimer":
        return "depends on M"
    if defn.mz_kind == "fixed":
        return "fixed diagnostic m/z"
    if defn.exact_delta_from_mh is None:
        return ""
    return f"{defn.exact_delta_from_mh:+.6f}"


def is_channel_eligible(defn: IonChannelDefinition, formula: str) -> bool:
    if not defn.formula_element_required:
        return True
    try:
        counts = parse_formula(str(formula or ""))
    except Exception:
        return False
    return int(counts.get(defn.formula_element_required, 0)) > 0


def channel_mz(defn: IonChannelDefinition, neutral_mass: float, formula: str = "") -> float:
    m = float(neutral_mass)
    mh = m - MASS_H_PLUS
    if defn.mz_kind == "mh":
        return mh
    if defn.mz_kind == "delta_mh":
        return mh + float(defn.exact_delta_from_mh or 0.0)
    if defn.mz_kind == "dimer":
        return 2.0 * m - MASS_H_PLUS
    if defn.mz_kind == "fixed":
        if defn.channel_id == "BR_ANION79":
            return MASS_BR79_ANION
        if defn.channel_id == "BR_ANION81":
            return MASS_BR81_ANION
        return float("nan")
    if defn.mz_kind == "loss_br_from_mh":
        if not is_channel_eligible(defn, formula):
            return float("nan")
        return mh - BR79
    if defn.mz_kind == "loss_hbr_from_mh":
        if not is_channel_eligible(defn, formula):
            return float("nan")
        return mh - MASS_H79BR
    return float("nan")


def default_discovery_panel(*, rt_tolerance_min: float = 0.10,
                            min_rel_height_pct: float = 1.0,
                            min_shape_correlation: float = 0.30) -> NegativeIonPanel:
    rows: List[PanelChannel] = []
    for defn in CHANNELS:
        if defn.channel_id == "MH" or defn.category == "isotope_support":
            continue
        rows.append(PanelChannel(
            channel_id=defn.channel_id,
            enabled=True,
            quant_action=defn.default_action,
            min_rel_height_pct=float(min_rel_height_pct),
            min_shape_correlation=float(min_shape_correlation),
            rt_tolerance_min=float(rt_tolerance_min),
            note=defn.note,
        ))
    return NegativeIonPanel(rows, panel_name="all_negative_ion_candidates")


def normalise_action(action: object, defn: IonChannelDefinition) -> str:
    a = str(action or "").strip().lower()
    aliases = {
        "add": "sum", "quant": "sum", "quantify": "sum", "include": "sum",
        "search_only": "search", "audit": "search", "report": "evidence",
        "diagnostic": "evidence", "off": "exclude", "disabled": "exclude",
    }
    a = aliases.get(a, a)
    if a not in {"sum", "search", "evidence", "exclude"}:
        a = defn.default_action
    if a == "sum" and not defn.quantifiable:
        return "evidence"
    return a


def load_panel(path: Optional[Path]) -> NegativeIonPanel:
    if path is None or not str(path).strip():
        return default_discovery_panel()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Negative-ion panel not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        rows: List[PanelChannel] = []
        for item in list(data.get("channels", []) or []):
            cid = str(item.get("channel_id") or item.get("Channel_ID") or "").strip().upper()
            if cid not in CHANNEL_BY_ID or cid == "MH" or CHANNEL_BY_ID[cid].category == "isotope_support":
                continue
            d = CHANNEL_BY_ID[cid]
            rows.append(PanelChannel(
                channel_id=cid,
                enabled=bool(item.get("enabled", True)),
                quant_action=normalise_action(item.get("quant_action") or item.get("action"), d),
                min_rel_height_pct=float(item.get("min_rel_height_pct", 1.0)),
                min_shape_correlation=float(item.get("min_shape_correlation", 0.30)),
                rt_tolerance_min=float(item.get("rt_tolerance_min", 0.10)),
                note=str(item.get("note", "") or ""),
            ))
        return NegativeIonPanel(rows, str(data.get("panel_name", p.stem)),
                                bool(data.get("require_primary_mh", True)),
                                str(data.get("version", "1.0")))

    rows: List[PanelChannel] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            cid = str(item.get("Channel_ID") or item.get("channel_id") or "").strip().upper()
            if cid not in CHANNEL_BY_ID or cid == "MH" or CHANNEL_BY_ID[cid].category == "isotope_support":
                continue
            d = CHANNEL_BY_ID[cid]
            enabled_text = str(item.get("Enabled", item.get("enabled", "true"))).strip().lower()
            enabled = enabled_text not in {"0", "false", "no", "n", "off", "exclude"}
            def fv(name: str, default: float) -> float:
                try:
                    return float(item.get(name, default) or default)
                except Exception:
                    return float(default)
            rows.append(PanelChannel(
                channel_id=cid,
                enabled=enabled,
                quant_action=normalise_action(item.get("Quant_Action") or item.get("Action"), d),
                min_rel_height_pct=fv("Min_Rel_Height_Pct", 1.0),
                min_shape_correlation=fv("Min_Shape_Correlation", 0.30),
                rt_tolerance_min=fv("RT_Tolerance_Min", 0.10),
                note=str(item.get("Panel_Note", item.get("Note", "")) or ""),
            ))
    return NegativeIonPanel(rows, panel_name=p.stem)


def write_panel_csv(panel: NegativeIonPanel, path: Path,
                    *, summary: Optional[Mapping[str, Mapping[str, object]]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    fields = [
        "Channel_ID", "Display", "Category", "Enabled", "Quant_Action",
        "Exact_Delta_From_MH_Da", "Source_Hypothesis", "Priority_Stars",
        "Min_Rel_Height_Pct", "Min_Shape_Correlation", "RT_Tolerance_Min",
        "Eligible_Count", "Detected_Count", "Detection_Rate_Pct",
        "Median_Area_Fraction_vs_MH_Pct", "Median_RT_Delta_Min",
        "Median_Shape_Correlation", "Isotope_Support_Rate_Pct",
        "Suggested_Status", "Panel_Note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in panel.channels:
            d = CHANNEL_BY_ID.get(row.channel_id)
            if d is None:
                continue
            s = dict(summary.get(row.channel_id, {}) or {})
            w.writerow({
                "Channel_ID": d.channel_id,
                "Display": d.display,
                "Category": d.category,
                "Enabled": bool(row.enabled),
                "Quant_Action": normalise_action(row.quant_action, d),
                "Exact_Delta_From_MH_Da": exact_delta_from_mh_text(d),
                "Source_Hypothesis": d.source,
                "Priority_Stars": d.priority_stars,
                "Min_Rel_Height_Pct": row.min_rel_height_pct,
                "Min_Shape_Correlation": row.min_shape_correlation,
                "RT_Tolerance_Min": row.rt_tolerance_min,
                "Eligible_Count": s.get("Eligible_Count", ""),
                "Detected_Count": s.get("Detected_Count", ""),
                "Detection_Rate_Pct": s.get("Detection_Rate_Pct", ""),
                "Median_Area_Fraction_vs_MH_Pct": s.get("Median_Area_Fraction_vs_MH_Pct", ""),
                "Median_RT_Delta_Min": s.get("Median_RT_Delta_Min", ""),
                "Median_Shape_Correlation": s.get("Median_Shape_Correlation", ""),
                "Isotope_Support_Rate_Pct": s.get("Isotope_Support_Rate_Pct", ""),
                "Suggested_Status": s.get("Suggested_Status", ""),
                "Panel_Note": row.note or d.note,
            })
    return path


def write_panel_json(panel: NegativeIonPanel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": panel.version,
        "panel_name": panel.panel_name,
        "require_primary_mh": panel.require_primary_mh,
        "channels": [asdict(x) for x in panel.channels],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def registry_rows() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for d in CHANNELS:
        if d.channel_id == "MH" or d.category == "isotope_support":
            continue
        out.append({
            "Channel_ID": d.channel_id,
            "Display": d.display,
            "Category": d.category,
            "Exact_Delta_From_MH_Da": exact_delta_from_mh_text(d),
            "Source_Hypothesis": d.source,
            "Priority_Stars": d.priority_stars,
            "Default_Action": d.default_action,
            "Quantifiable": d.quantifiable,
            "Formula_Element_Required": d.formula_element_required,
            "Support_Channel_ID": d.support_channel_id,
            "Note": d.note,
        })
    return out

# Public convenience aliases used by the UI/discovery workflow.
def channel_registry_rows() -> List[Dict[str, object]]:
    return registry_rows()


def save_panel(panel: NegativeIonPanel, path: Path) -> Path:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return write_panel_json(panel, p)
    return write_panel_csv(panel, p)


def write_default_panel_template(path: Path) -> Path:
    return write_panel_csv(default_discovery_panel(), Path(path))


def write_channel_registry_csv(path: Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = registry_rows()
    fields = list(rows[0].keys()) if rows else ["Channel_ID"]
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p
