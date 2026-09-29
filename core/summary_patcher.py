"Patch existing Excel sheet/XIC summary workbooks without re-running RAW extraction.\nThis module is intended for the case where an old summary workbook already exists\nbeen updated from the legacy formula-only layout to the named-triplet layout:\n    A/B/C = reagent name / formula / mass\n    D/E/F = reagent name / formula / mass\n    G/H/I = reagent name / formula / mass\nThe patcher can update old combo strings such as\nto\n``A#8:P2-X5b:C2H7N | B#1:1k:C8H7N3O | C#1:Nu-2j:C6H7NO``\nby reading the reagent names from the source workbook. It also translates the\nsummary workbook's Chinese sheet names, headers and common text values to English."

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .chemistry import format_formula_hill, parse_formula

_SUBSCRIPT_TRANS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")


@dataclass
class PatchResult:
    output_path: Path
    combo_cells_seen: int = 0
    combo_cells_changed: int = 0
    reagent_labels_loaded: int = 0
    sheet_names_changed: int = 0
    header_cells_translated: int = 0
    value_cells_translated: int = 0
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def _safe_text(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip().translate(_SUBSCRIPT_TRANS)
    if s.startswith("'"):
        s = s[1:].strip()
    return s


def _normalize_sheet_key(name: str) -> str:
    return str(name or "").strip().lower()


def _safe_sheet_title(title: str, used: Iterable[str]) -> str:
    """Return an Excel-safe, unique sheet title (<=31 chars)."""

    base = re.sub(r"[\\/*?:\[\]]", "_", str(title or "Sheet")).strip() or "Sheet"
    base = base[:31]
    used_set = {str(x) for x in used}
    if base not in used_set:
        return base
    for i in range(2, 1000):
        suffix = f"_{i}"
        cand = (base[: 31 - len(suffix)] + suffix)[:31]
        if cand not in used_set:
            return cand
    return base[:28] + "_x"


# Existing summary sheet names -> English names.
SHEET_NAME_TRANSLATIONS = {'运行汇总': 'Run Summary', '三平行平均结果': 'Triplicate Average', '三平行XIC汇总': 'Triplicate XIC Summary', 'XIC长表': 'XIC Long Table', '内标汇总': 'Internal Standard'}

# Header translations. Keep common English headers unchanged.
HEADER_TRANSLATIONS = {'构件A数量': 'A Count', '构件B数量': 'B Count', '构件C数量': 'C Count', '组合数(保留重复)': 'Combination Count (Duplicates Kept)', '无效组合数': 'Invalid Combination Count', '平行1 RAW': 'Rep1 RAW', '平行1状态': 'Rep1 Status', '平行2 RAW': 'Rep2 RAW', '平行2状态': 'Rep2 Status', '平行3 RAW': 'Rep3 RAW', '平行3状态': 'Rep3 Status', '统一内标分子式': 'Common internal-standard formula', '备注': 'Note', '峰面积有效平行数': 'Valid_Area_Replicate_Count', '比例有效平行数': 'Valid_Ratio_Replicate_Count', '三平行完整': 'Triplicates_Complete', '平均峰面积': 'Mean_Peak_Area', '峰面积SD': 'Peak_Area_SD', '峰面积RSD_%': 'Peak_Area_RSD_%', '平均相对内标比例_%': 'Mean_Area_to_IS_Ratio_%', '平均总面积占比_%': 'Mean_Percent_of_Total_Area_%', '比例SD': 'Ratio_SD', '比例RSD_%': 'Ratio_RSD_%', '平均Apex_RT': 'Mean_Apex_RT', '最终三平行平均结果': 'Final_Triplicate_Mean_Result', '严格三平行平均结果(3/3)': 'Strict_Triplicate_Mean_Result_3of3', '结果类型': 'Result_Type', '结果有效平行数': 'Valid_Result_Replicate_Count', '平均规则': 'Averaging_Rule', 'A Count': 'A Count', 'B Count': 'B Count', 'C Count': 'C Count', 'Combination Count (Duplicates Kept)': 'Combination Count (Duplicates Kept)', 'Invalid Combination Count': 'Invalid Combination Count', 'Rep1 RAW': 'Rep1 RAW', 'Rep1 Status': 'Rep1 Status', 'Rep2 RAW': 'Rep2 RAW', 'Rep2 Status': 'Rep2 Status', 'Rep3 RAW': 'Rep3 RAW', 'Rep3 Status': 'Rep3 Status', 'Internal Standard Formula': 'Common internal-standard formula', 'Note': 'Note', 'Valid Replicates (Area)': 'Valid_Area_Replicate_Count', 'Valid Replicates (Ratio)': 'Valid_Ratio_Replicate_Count', 'Complete Triplicate': 'Triplicates_Complete', 'Mean Area': 'Mean_Peak_Area', 'Area SD': 'Peak_Area_SD', 'Area RSD (%)': 'Peak_Area_RSD_%', 'Mean Ratio vs IS (%)': 'Mean_Area_to_IS_Ratio_%', 'Mean Percent of Total Area (%)': 'Mean_Percent_of_Total_Area_%', 'Ratio SD': 'Ratio_SD', 'Ratio RSD (%)': 'Ratio_RSD_%', 'Mean Apex RT': 'Mean_Apex_RT', 'Final Triplicate Mean': 'Final_Triplicate_Mean_Result', 'Strict Triplicate Mean (3/3)': 'Strict_Triplicate_Mean_Result_3of3', 'Result Type': 'Result_Type', 'Valid Replicates (Result)': 'Valid_Result_Replicate_Count', 'Averaging Rule': 'Averaging_Rule'}

VALUE_TRANSLATIONS = {'是': 'Yes', '否': 'No', '未配置': 'Not configured', '统一内标': 'Internal Standard', '相对内标峰面积比(%)': 'Area / Internal Standard Area (%)', '峰面积': 'Peak Area', '有效平行均值只统计Found=True且数值有效的结果；严格三平行平均仅在3个平行均有效时填写，缺失值不按0计入': 'Valid-replicate mean uses only Found=True records with valid numeric values; strict triplicate mean is reported only when all three replicates are valid; missing values are not treated as zero.'}

# Minimal fragment replacement for notes/status values; kept conservative.
FRAGMENT_TRANSLATIONS: Tuple[Tuple[str, str], ...] = (
    ('Shared across sheets and replicates; Ratio=Area/IS x 100%', "Shared by all sheets/replicates; Ratio=Area/IS\u00D7100%"),
    ('Internal Standard', "Internal Standard"),
    ('Missing', "missing"),
    ('Failed', "failed"),
    ('Replicate', "Replicate"),
)

_COMPONENTS = {
    "A": (1, 2),  # name col, formula col
    "B": (4, 5),
    "C": (7, 8),
}


def _load_named_triplet_labels(source_workbook: Path) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], List[str], int]:
    """Load reagent labels from the named-triplet source workbook.

    Returns a nested mapping:
    ``mapping[sheet_key][letter]['R9']`` and ``mapping[sheet_key][letter]['#8']`` -> ``P2-X5b:C2H7N``.
    This supports both the old explicit Excel-row token (``[R9]``) and component-list index.
    """

    try:
        from openpyxl import load_workbook
    except Exception as e:
        raise RuntimeError('Reading the source workbook requires openpyxl. Install it with: python -m pip install openpyxl') from e

    source_workbook = Path(source_workbook)
    if not source_workbook.exists():
        raise FileNotFoundError(str(source_workbook))

    wb = load_workbook(filename=str(source_workbook), read_only=True, data_only=True)
    mapping: Dict[str, Dict[str, Dict[str, str]]] = {}
    warnings: List[str] = []
    n_loaded = 0
    try:
        for ws in wb.worksheets:
            if getattr(ws, "sheet_state", "visible") != "visible":
                continue
            skey = _normalize_sheet_key(ws.title)
            sheet_map: Dict[str, Dict[str, str]] = {"A": {}, "B": {}, "C": {}}
            for letter, (name_col, formula_col) in _COMPONENTS.items():
                idx = 0
                for r in range(2, int(ws.max_row or 2) + 1):
                    reagent = _safe_text(ws.cell(row=r, column=name_col).value)
                    formula_raw = _safe_text(ws.cell(row=r, column=formula_col).value)
                    if not reagent and not formula_raw:
                        continue
                    if not formula_raw:
                        continue
                    try:
                        formula = format_formula_hill(parse_formula(formula_raw))
                    except Exception:
                        # If formula parsing fails, still keep the raw text; better than losing traceability.
                        formula = formula_raw
                    idx += 1
                    label = f"{reagent}:{formula}" if reagent else f"R{r}:{formula}"
                    sheet_map[letter][f"R{r}"] = label
                    sheet_map[letter][f"#{idx}"] = label
                    n_loaded += 1
            mapping[skey] = sheet_map
    finally:
        wb.close()
    return mapping, warnings, n_loaded


_COMBO_PART_RE = re.compile(
    r"^(?P<letter>[ABC])#(?P<idx>\d+):(?P<body>.*?)(?:\[R(?P<row>\d+)\])?:(?P<formula>[A-Z][A-Za-z0-9]*)$"
)


def _looks_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def patch_combo_string(combo: str, *, sheet_name: str, label_mapping: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    """Patch one Combo text while preserving A#/B#/C# numbering."""

    text = str(combo or "")
    if "#" not in text or ":" not in text:
        return text

    sheet_key = _normalize_sheet_key(sheet_name)
    sheet_map = label_mapping.get(sheet_key, {})
    if not sheet_map:
        # Source workbook might use a slightly different sheet key; no safe patch.
        return text

    parts = [p.strip() for p in text.split("|")]
    changed = False
    out_parts: List[str] = []
    for part in parts:
        m = _COMBO_PART_RE.match(part)
        if not m:
            out_parts.append(part)
            continue
        letter = m.group("letter")
        idx = m.group("idx")
        row = m.group("row")
        old_body = m.group("body") or ""
        formula = m.group("formula")
        repl = None
        if row:
            repl = sheet_map.get(letter, {}).get(f"R{row}")
        if not repl:
            repl = sheet_map.get(letter, {}).get(f"#{idx}")
        if not repl:
            # At least strip Chinese category names if the old label is clearly legacy.
            if _looks_chinese(old_body):
                repl = formula
            else:
                out_parts.append(part)
                continue
        new_part = f"{letter}#{idx}:{repl}"
        out_parts.append(new_part)
        if new_part != part:
            changed = True
    return " | ".join(out_parts) if changed else text


def _translate_text_value(value: object) -> Tuple[object, bool]:
    if not isinstance(value, str):
        return value, False
    s = value.strip()
    if s in VALUE_TRANSLATIONS:
        return VALUE_TRANSLATIONS[s], True
    if s in HEADER_TRANSLATIONS:
        return HEADER_TRANSLATIONS[s], True
    if not _looks_chinese(s):
        return value, False
    new = s
    for zh, en in FRAGMENT_TRANSLATIONS:
        new = new.replace(zh, en)
    return new, new != value


def _translate_headers(ws) -> int:
    changed = 0
    if ws.max_row < 1:
        return 0
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=c)
        val = cell.value
        if isinstance(val, str) and val.strip() in HEADER_TRANSLATIONS:
            cell.value = HEADER_TRANSLATIONS[val.strip()]
            changed += 1
    return changed


def _find_header_columns(ws, header_names: Iterable[str]) -> Dict[str, int]:
    wanted = set(header_names)
    out: Dict[str, int] = {}
    if ws.max_row < 1:
        return out
    for c in range(1, ws.max_column + 1):
        val = ws.cell(row=1, column=c).value
        if isinstance(val, str):
            key = val.strip()
            if key in wanted:
                out[key] = c
    return out


def _style_and_widths(wb) -> None:
    try:
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except Exception:
        return

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for ws in wb.worksheets:
        if ws.max_row >= 1:
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.freeze_panes = "A2"
            try:
                ws.auto_filter.ref = ws.dimensions
            except Exception:
                pass
        for c in range(1, ws.max_column + 1):
            header = str(ws.cell(row=1, column=c).value or "")
            max_len = len(header)
            for r in range(2, min(ws.max_row, 250) + 1):
                v = ws.cell(row=r, column=c).value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            cap = 60 if header == "Combo" else 42 if "PNG" in header or "Path" in header else 28
            ws.column_dimensions[get_column_letter(c)].width = max(10, min(cap, max_len + 2))


def patch_summary_workbook(
    summary_xlsx: Path,
    source_workbook: Path,
    out_xlsx: Path,
    *,
    patch_combo: bool = True,
    translate_to_english: bool = True,
) -> PatchResult:
    'Patch an existing summary workbook in-place style and save as a new file.\n    Parameters\n    ----------\n    summary_xlsx:\n    source_workbook:\n        The component workbook in named-triplet layout. Used only to update combo labels.\n    out_xlsx:\n        Output workbook path. Existing file will be overwritten.\n    patch_combo:\n        Update old legacy Chinese combo labels to reagent-code labels.\n    translate_to_english:\n        Translate summary sheets, headers, and common values to English.'

    try:
        from openpyxl import load_workbook
    except Exception as e:
        raise RuntimeError('Updating the workbook requires openpyxl. Install it with: python -m pip install openpyxl') from e

    summary_xlsx = Path(summary_xlsx)
    source_workbook = Path(source_workbook)
    out_xlsx = Path(out_xlsx)
    if not summary_xlsx.exists():
        raise FileNotFoundError(str(summary_xlsx))
    if patch_combo and not source_workbook.exists():
        raise FileNotFoundError(str(source_workbook))

    label_mapping: Dict[str, Dict[str, Dict[str, str]]] = {}
    warnings: List[str] = []
    n_loaded = 0
    if patch_combo:
        label_mapping, warnings, n_loaded = _load_named_triplet_labels(source_workbook)

    wb = load_workbook(filename=str(summary_xlsx))
    result = PatchResult(output_path=out_xlsx, reagent_labels_loaded=n_loaded, warnings=warnings)

    # Translate sheet names first.
    if translate_to_english:
        used_titles = [ws.title for ws in wb.worksheets]
        for ws in wb.worksheets:
            if ws.title in SHEET_NAME_TRANSLATIONS:
                new_title = _safe_sheet_title(SHEET_NAME_TRANSLATIONS[ws.title], [t for t in used_titles if t != ws.title])
                used_titles = [new_title if t == ws.title else t for t in used_titles]
                if new_title != ws.title:
                    ws.title = new_title
                    result.sheet_names_changed += 1

    # Patch combo columns and translate headers/values.
    for ws in wb.worksheets:
        # Need to know sheet column before header translation, because original headers may be Chinese or English.
        hdrs = _find_header_columns(ws, {"Sheet", "Combo"})
        sheet_col = hdrs.get("Sheet")
        combo_col = hdrs.get("Combo")
        if patch_combo and combo_col:
            for r in range(2, ws.max_row + 1):
                cell = ws.cell(row=r, column=combo_col)
                if not isinstance(cell.value, str) or not cell.value.strip():
                    continue
                result.combo_cells_seen += 1
                sheet_name = ws.cell(row=r, column=sheet_col).value if sheet_col else ""
                new_combo = patch_combo_string(cell.value, sheet_name=str(sheet_name or ""), label_mapping=label_mapping)
                if new_combo != cell.value:
                    cell.value = new_combo
                    result.combo_cells_changed += 1

        if translate_to_english:
            result.header_cells_translated += _translate_headers(ws)
            # Translate common Chinese values; skip Combo cells after we just patched them.
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if combo_col and cell.column == combo_col:
                        continue
                    new_val, changed = _translate_text_value(cell.value)
                    if changed:
                        cell.value = new_val
                        result.value_cells_translated += 1

    # Bring default active sheet to Triplicate Average if present.
    for idx, ws in enumerate(wb.worksheets):
        if ws.title in {'Triplicate Average', 'Triplicate Average', '三平行平均结果', 'triplicate average', 'triplicate average'}:
            wb.active = idx
            break

    _style_and_widths(wb)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_xlsx))
    return result
