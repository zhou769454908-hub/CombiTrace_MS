"""Private/local structure features for CombiTrace-IE.

The user's exact structures and reaction rule remain on the local computer.
Supported structure workbook:

Structures sheet:
    Role | ID | Core_ID | Formula | Net_Formula | SMILES | Combo | Product_SMILES | Notes

ReactionRules sheet:
    Rule_ID | Core_ID | Reactant_Order | Reaction_SMARTS |
    A_Loss | B_Loss | C_Loss | Enabled | Notes

Current default stoichiometric check:
    Product = A + B + C + CORE contribution - N2 - H
where Net_Formula (when supplied) defines the stoichiometric contribution.
which represents B losing N2 and C losing H.  The fixed middle structure is
explicitly represented by a CORE row and linked through Core_ID.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chemistry import format_formula_hill, parse_formula
from .response_predictor import PreparedRow, _norm_header, parse_number, read_table

RDKIT_AVAILABLE = False
RDKIT_ERROR = ""
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.rdChemReactions import ReactionFromSmarts
    RDKIT_AVAILABLE = True
except Exception as e:  # optional dependency
    RDKIT_ERROR = str(e)


@dataclass
class StructureEntry:
    role: str
    entry_id: str
    core_id: str = ""
    formula: str = ""
    net_formula: str = ""
    smiles: str = ""
    combo: str = ""
    product_smiles: str = ""
    notes: str = ""
    numeric: Dict[str, float] = field(default_factory=dict)
    source_file: str = ""
    source_sheet: str = ""
    source_row: object = ""


@dataclass
class ReactionRule:
    rule_id: str
    core_id: str
    reactant_order: Tuple[str, ...] = ("CORE", "A", "B", "C")
    reaction_smarts: str = ""
    a_loss: str = ""
    b_loss: str = "N2"
    c_loss: str = "H"
    enabled: bool = True
    notes: str = ""


@dataclass
class StructureLibrary:
    entries: Dict[Tuple[str, str], StructureEntry] = field(default_factory=dict)
    cores: Dict[str, StructureEntry] = field(default_factory=dict)
    products: Dict[str, StructureEntry] = field(default_factory=dict)
    rules: Dict[str, ReactionRule] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    rdkit_available: bool = RDKIT_AVAILABLE


def _text(v: object) -> str:
    return "" if v is None else str(v).strip()


def _col(headers: Sequence[str], names: Sequence[str]) -> str:
    norm = {_norm_header(h): h for h in headers}
    for name in names:
        n = _norm_header(name)
        if n in norm:
            return norm[n]
    for h in headers:
        nh = _norm_header(h)
        for name in names:
            n = _norm_header(name)
            if n and n in nh:
                return h
    return ""


def _role(v: object) -> str:
    s = _text(v).upper().replace(" ", "")
    aliases = {
        "MIDDLE": "CORE", "CENTER": "CORE", "CENTRE": "CORE",
        "FIXEDCORE": "CORE", 'CORE': "CORE", 'Intermediate': "CORE",
        'Fixed_Intermediate': "CORE", 'Product': "PRODUCT",
    }
    aliases.update({"\u4e2d\u95f4\u4f53": "CORE", "\u56fa\u5b9a\u4e2d\u95f4\u4f53": "CORE", "\u4ea7\u7269": "PRODUCT", "\u6838\u5fc3": "CORE"})
    aliases = {k.upper().replace(" ", ""): v for k, v in aliases.items()}
    return aliases.get(s, s)


def _combo_key(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def _bool(v: object, default: bool = True) -> bool:
    s = _text(v).lower()
    if not s:
        return default
    if s in {'0', 'false', 'no', 'off', 'No', '否'}:
        return False
    if s in {'1', 'true', 'yes', 'on', 'Yes', '是'}:
        return True
    return default


def _sheet_exists(path: Path, name: str) -> bool:
    if not name or path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return False
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(path), read_only=True, data_only=True)
        try:
            return name in wb.sheetnames
        finally:
            wb.close()
    except Exception:
        return False


def _pick_sheet(path: Path, explicit: str, candidates: Sequence[str]) -> str:
    if explicit and _sheet_exists(path, explicit):
        return explicit
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            from openpyxl import load_workbook
            wb = load_workbook(str(path), read_only=True, data_only=True)
            try:
                mapping = {_norm_header(x): x for x in wb.sheetnames}
                for c in candidates:
                    if _norm_header(c) in mapping:
                        return mapping[_norm_header(c)]
            finally:
                wb.close()
        except Exception:
            pass
    return explicit


def load_structure_library(path: Optional[Path], structure_sheet: str = "", reaction_sheet: str = "") -> StructureLibrary:
    lib = StructureLibrary()
    if not path:
        lib.warnings.append('No structure library was supplied; using the Combo/formula/RT fallback.')
        return lib
    path = Path(path)
    if not path.exists():
        lib.warnings.append(f'Structure library not found: {path}')
        return lib

    try:
        sname = _pick_sheet(path, structure_sheet, ["Structures", "StructureLibrary", 'Structure_Library'])
        table = read_table(path, sheet_name=sname)
    except Exception as e:
        lib.warnings.append(f'Could not read structure library: {e}')
        return lib

    h = table.headers
    role_c = _col(h, ["Role", "Type", 'Role', 'Category'])
    id_c = _col(h, ["ID", "Fragment_ID", "Component_ID", 'ID', 'Component_ID'])
    core_c = _col(h, ["Core_ID", "Middle_ID", "Fixed_Core_ID", 'Intermediate_ID', 'Core_ID'])
    formula_c = _col(h, ["Formula", "Molecular Formula", 'Formula', "Full_Formula", 'Full_Formula'])
    net_formula_c = _col(h, [
        "Net_Formula", "Formula_Contribution", "Stoichiometric_Formula", "Net Formula",
        'Net_Formula', 'Stoichiometric_Formula', 'Element_Contribution',
    ])
    smiles_c = _col(h, ["SMILES", "Canonical_SMILES", 'Structure'])
    combo_c = _col(h, ["Combo", 'Combo', 'Source_Combination'])
    product_c = _col(h, ["Product_SMILES", "Product SMILES", 'Product_SMILES'])
    notes_c = _col(h, ["Notes", "Note", 'Note'])
    if not role_c or not id_c:
        lib.warnings.append('The structure library must contain Role and ID columns.')
        return lib
    metadata = {x for x in [role_c, id_c, core_c, formula_c, net_formula_c, smiles_c, combo_c, product_c, notes_c] if x}

    for raw in table.rows:
        role = _role(raw.get(role_c, ""))
        entry_id = _text(raw.get(id_c, ""))
        if not role or not entry_id:
            continue
        numeric: Dict[str, float] = {}
        for k, v in raw.items():
            if k in metadata or str(k).startswith("__source_"):
                continue
            x = parse_number(v)
            if x is not None:
                numeric[str(k)] = float(x)
        entry = StructureEntry(
            role=role,
            entry_id=entry_id,
            core_id=_text(raw.get(core_c, "")) if core_c else "",
            formula=_text(raw.get(formula_c, "")) if formula_c else "",
            net_formula=_text(raw.get(net_formula_c, "")) if net_formula_c else "",
            smiles=_text(raw.get(smiles_c, "")) if smiles_c else "",
            combo=_text(raw.get(combo_c, "")) if combo_c else "",
            product_smiles=_text(raw.get(product_c, "")) if product_c else "",
            notes=_text(raw.get(notes_c, "")) if notes_c else "",
            numeric=numeric,
            source_file=_text(raw.get("__source_file__", "")),
            source_sheet=_text(raw.get("__source_sheet__", "")),
            source_row=raw.get("__source_row__", ""),
        )
        lib.entries[(role, entry_id)] = entry
        if role == "CORE":
            lib.cores[entry_id] = entry
        if role == "PRODUCT" and entry.combo:
            lib.products[_combo_key(entry.combo)] = entry
        elif entry.combo and entry.product_smiles:
            lib.products[_combo_key(entry.combo)] = entry

    if len(lib.cores) == 1:
        only = next(iter(lib.cores))
        for entry in lib.entries.values():
            if entry.role in {"A", "B", "C"} and not entry.core_id:
                entry.core_id = only

    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        rname = _pick_sheet(path, reaction_sheet, ["ReactionRules", "Reaction Rules", 'ReactionRules'])
        if rname and _sheet_exists(path, rname):
            try:
                rt = read_table(path, sheet_name=rname)
                rh = rt.headers
                rid_c = _col(rh, ["Rule_ID", "Reaction_ID", 'Rule_ID'])
                rcore_c = _col(rh, ["Core_ID", "Middle_ID", 'Core_ID'])
                order_c = _col(rh, ["Reactant_Order", "Reactant Order", 'Reactant_Order'])
                smarts_c = _col(rh, ["Reaction_SMARTS", "SMARTS", 'Reaction_SMARTS'])
                aloss_c = _col(rh, ["A_Loss", "A Loss", 'A_Loss'])
                bloss_c = _col(rh, ["B_Loss", "B Loss", 'B_Loss'])
                closs_c = _col(rh, ["C_Loss", "C Loss", 'C_Loss'])
                enabled_c = _col(rh, ["Enabled", "Enable", 'Enabled'])
                rnotes_c = _col(rh, ["Notes", "Note", 'Note'])
                for raw in rt.rows:
                    core_id = _text(raw.get(rcore_c, "")) if rcore_c else ""
                    rid = _text(raw.get(rid_c, "")) if rid_c else (core_id or f"R{len(lib.rules)+1}")
                    order = _text(raw.get(order_c, "")) if order_c else "CORE,A,B,C"
                    rule = ReactionRule(
                        rule_id=rid,
                        core_id=core_id,
                        reactant_order=tuple(x.strip().upper() for x in re.split(r"[,;|>]", order) if x.strip()) or ("CORE", "A", "B", "C"),
                        reaction_smarts=_text(raw.get(smarts_c, "")) if smarts_c else "",
                        a_loss=_text(raw.get(aloss_c, "")) if aloss_c else "",
                        b_loss=_text(raw.get(bloss_c, "")) if bloss_c else "N2",
                        c_loss=_text(raw.get(closs_c, "")) if closs_c else "H",
                        enabled=_bool(raw.get(enabled_c, ""), True) if enabled_c else True,
                        notes=_text(raw.get(rnotes_c, "")) if rnotes_c else "",
                    )
                    if rule.enabled:
                        lib.rules[core_id] = rule
            except Exception as e:
                lib.warnings.append(f'Could not read ReactionRules: {e}')

    for core_id in lib.cores:
        if core_id not in lib.rules:
            lib.rules[core_id] = ReactionRule(f"Default_{core_id}", core_id)
    if not RDKIT_AVAILABLE:
        lib.warnings.append('RDKit is unavailable; SMILES descriptors and reaction SMARTS cannot be processed.' + (f" ({RDKIT_ERROR})" if RDKIT_ERROR else ""))
    return lib


def _counts(formula: str, warnings: List[str], label: str) -> Dict[str, int]:
    if not formula:
        warnings.append(f'{label} is missing Formula')
        return {}
    try:
        return {str(k): int(v) for k, v in parse_formula(formula).items()}
    except Exception as e:
        warnings.append(f'{label}: could not parse Formula {formula!r}: {e}')
        return {}


def _add(dst: Dict[str, int], src: Dict[str, int], sign: int = 1) -> None:
    for el, n in src.items():
        dst[el] = dst.get(el, 0) + sign * int(n)


def formula_equal(a: str, b: str) -> bool:
    try:
        return parse_formula(a) == parse_formula(b)
    except Exception:
        return False


def resolve_core_id(row: PreparedRow, lib: StructureLibrary) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    norm_map = {_norm_header(k): k for k in row.raw if not str(k).startswith("__source_")}
    for name in ["Core_ID", "Middle_ID", "Fixed_Core_ID", 'Intermediate_ID', 'Core_ID']:
        n = _norm_header(name)
        if n in norm_map:
            value = _text(row.raw.get(norm_map[n], ""))
            if value:
                return value, warnings
    ids = set()
    for role in ("A", "B", "C"):
        label = _text(row.combo_parts.get(role, {}).get("label", ""))
        e = lib.entries.get((role, label))
        if e and e.core_id:
            ids.add(e.core_id)
    if len(ids) == 1:
        return next(iter(ids)), warnings
    if len(ids) > 1:
        warnings.append(f'Components refer to different Core_ID values: {sorted(ids)}')
        return sorted(ids)[0], warnings
    if len(lib.cores) == 1:
        return next(iter(lib.cores)), warnings
    if not lib.cores:
        warnings.append('The structure library has no CORE record.')
    else:
        warnings.append('Multiple CORE records exist, but Core_ID is not specified.')
    return "", warnings


def calculate_product_formula(row: PreparedRow, lib: StructureLibrary, core_id: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    total: Dict[str, int] = {}
    for role in ("A", "B", "C"):
        part = row.combo_parts.get(role, {})
        label = _text(part.get("label", ""))
        entry = lib.entries.get((role, label))
        formula = (entry.net_formula or entry.formula) if entry else _text(part.get("formula", ""))
        if not formula:
            formula = _text(part.get("formula", ""))
        _add(total, _counts(formula, warnings, f"{role}/{label or '?'}"), +1)
    core = lib.cores.get(core_id)
    if core:
        core_formula = core.net_formula or core.formula
        _add(total, _counts(core_formula, warnings, f"CORE/{core_id}"), +1)
    else:
        warnings.append(f'CORE record not found: {core_id}')
    rule = lib.rules.get(core_id, ReactionRule("Default", core_id))
    for role, loss in (("A", rule.a_loss), ("B", rule.b_loss), ("C", rule.c_loss)):
        if loss:
            _add(total, _counts(loss, warnings, f"{role}_Loss"), -1)
    if any(n < 0 for n in total.values()):
        warnings.append(f'Negative element counts after stoichiometric correction: { {k: v for k, v in total.items() if v < 0}}')
        return "", warnings
    return format_formula_hill(total), warnings


def _mol(smiles: str):
    if not RDKIT_AVAILABLE or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _canonical(mol) -> str:
    if mol is None or not RDKIT_AVAILABLE:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _fp(mol, nbits: int = 512) -> Optional[np.ndarray]:
    if mol is None or not RDKIT_AVAILABLE:
        return None
    try:
        try:
            bv = AllChem.GetMorganGenerator(radius=2, fpSize=nbits).GetFingerprint(mol)
        except Exception:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr
    except Exception:
        return None



_SMARTS_CACHE: Dict[str, object] = {}


def _smarts_query(smarts: str):
    if not RDKIT_AVAILABLE or not smarts:
        return None
    if smarts not in _SMARTS_CACHE:
        try:
            _SMARTS_CACHE[smarts] = Chem.MolFromSmarts(smarts)
        except Exception:
            _SMARTS_CACHE[smarts] = None
    return _SMARTS_CACHE.get(smarts)


def _smarts_count(mol, smarts: str) -> float:
    q = _smarts_query(smarts)
    if mol is None or q is None:
        return 0.0
    try:
        return float(len(mol.GetSubstructMatches(q, uniquify=True)))
    except Exception:
        return 0.0


def _acid_base_proxy_descriptors(mol) -> Dict[str, float]:
    """Return transparent SMARTS-based ionisation proxies.

    These are *not* predicted pKa values.  They are deliberately named as
    proxies because RDKit does not provide a generally reliable pKa model.
    The counts are useful as physically interpretable features for small-data
    ESI response models and remain auditable in the output workbook.
    """
    if mol is None or not RDKIT_AVAILABLE:
        return {}

    # Acidic groups.  Overlapping subclasses are retained as separate features;
    # AcidicSiteCount_proxy uses a union-like broad pattern to reduce double counting.
    acid_patterns = {
        "CarboxylicAcidCount": "[CX3](=O)[OX2H1]",
        "SulfonicAcidCount": "[SX4](=O)(=O)[OX2H1]",
        "PhosphoricOHCount": "[PX4](=O)([OX2H1])",
        "PhenolCount": "[c][OX2H1]",
        "ThiolCount": "[SX2H1]",
        "ImideNHCount": "[NX3H1]([CX3](=O))[CX3](=O)",
    }
    # Basic groups.  The aliphatic amine pattern excludes the most common
    # resonance-deactivated nitrogens (amide/sulfonamide/phosphoramide).
    base_patterns = {
        "AliphaticAmineCount": "[NX3;H0,H1,H2;+0;!$(N[C,S,P]=[O,S,N]);!$(N-S(=O)=O);!$(N-P(=O))]",
        "AromaticBasicNCount": "[nH0;+0]",
        "AmidineGuanidineCount": "[NX3;H0,H1,H2][CX3](=[NX2])[NX3]",
        "ImineNCount": "[NX2;H0;+0]=[CX3]",
        "QuaternaryAmmoniumCount": "[N+;X4]",
    }
    ewg_patterns = {
        "CarbonylCount_proxy": "[CX3]=[OX1]",
        "NitroCount_proxy": "[NX3+](=O)[O-]",
        "NitrileCount_proxy": "[CX2]#N",
        "SulfoxideSulfoneCount_proxy": "[SX3,SX4](=O)",
        "HalogenCount_proxy": "[F,Cl,Br,I]",
    }
    out: Dict[str, float] = {}
    for name, smarts in acid_patterns.items():
        out[name] = _smarts_count(mol, smarts)
    for name, smarts in base_patterns.items():
        out[name] = _smarts_count(mol, smarts)
    for name, smarts in ewg_patterns.items():
        out[name] = _smarts_count(mol, smarts)
    out["ElectronWithdrawingGroupCount_proxy"] = float(sum(out[name] for name in ewg_patterns))

    # Broad, non-overlapping-enough proxies for model input.
    acidic = (
        out["CarboxylicAcidCount"] + out["SulfonicAcidCount"] +
        out["PhosphoricOHCount"] + out["PhenolCount"] +
        out["ThiolCount"] + out["ImideNHCount"]
    )
    basic = (
        out["AliphaticAmineCount"] + out["AromaticBasicNCount"] +
        out["AmidineGuanidineCount"] + out["ImineNCount"] +
        out["QuaternaryAmmoniumCount"]
    )
    out["AcidicSiteCount_proxy"] = float(acidic)
    out["BasicSiteCount_proxy"] = float(basic)
    out["IonizableSiteCount_proxy"] = float(acidic + basic)
    out["BasicMinusAcidic_proxy"] = float(basic - acidic)
    out["ZwitterionPotential_proxy"] = 1.0 if acidic > 0 and basic > 0 else 0.0

    pos_atoms = 0
    neg_atoms = 0
    abs_formal = 0.0
    for atom in mol.GetAtoms():
        q = int(atom.GetFormalCharge())
        if q > 0:
            pos_atoms += 1
        elif q < 0:
            neg_atoms += 1
        abs_formal += abs(q)
    out["PositiveFormalChargeAtomCount"] = float(pos_atoms)
    out["NegativeFormalChargeAtomCount"] = float(neg_atoms)
    out["AbsoluteFormalChargeSum"] = float(abs_formal)
    return out


def _gasteiger_charge_descriptors(mol) -> Dict[str, float]:
    if mol is None or not RDKIT_AVAILABLE:
        return {}
    try:
        m = Chem.AddHs(Chem.Mol(mol))
        AllChem.ComputeGasteigerCharges(m, nIter=12, throwOnParamFailure=False)
        charges: List[float] = []
        heavy_charges: List[float] = []
        for atom in m.GetAtoms():
            if not atom.HasProp("_GasteigerCharge"):
                continue
            try:
                q = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                continue
            if not math.isfinite(q):
                continue
            charges.append(q)
            if atom.GetAtomicNum() > 1:
                heavy_charges.append(q)
        vals = heavy_charges or charges
        if not vals:
            return {}
        positive = sum(q for q in vals if q > 0)
        negative = -sum(q for q in vals if q < 0)
        return {
            "GasteigerChargeMax": float(max(vals)),
            "GasteigerChargeMin": float(min(vals)),
            "GasteigerChargeRange": float(max(vals) - min(vals)),
            "GasteigerAbsChargeMean": float(sum(abs(q) for q in vals) / len(vals)),
            "GasteigerPositiveChargeSum": float(positive),
            "GasteigerNegativeChargeAbsSum": float(negative),
            "GasteigerChargeSeparation": float(positive + negative),
        }
    except Exception:
        return {}


def _extended_vsa_descriptors(mol) -> Dict[str, float]:
    """Selected charge/lipophilicity/refractivity surface-area descriptors."""
    if mol is None or not RDKIT_AVAILABLE:
        return {}
    out: Dict[str, float] = {}
    prefixes = ("PEOE_VSA", "SlogP_VSA", "SMR_VSA", "EState_VSA", "BCUT2D_")
    for name, fn in getattr(Descriptors, "_descList", []):
        if not str(name).startswith(prefixes):
            continue
        try:
            x = float(fn(mol))
            if math.isfinite(x):
                out[str(name)] = x
        except Exception:
            pass
    return out


def _desc2d(mol) -> Dict[str, float]:
    if mol is None or not RDKIT_AVAILABLE:
        return {}
    fns = {
        "MolWt": Descriptors.MolWt,
        "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
        "ExactMolWt": Descriptors.ExactMolWt,
        "MolLogP": Crippen.MolLogP,
        "MolMR": Crippen.MolMR,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "HBD": Lipinski.NumHDonors,
        "HBA": Lipinski.NumHAcceptors,
        "NHOHCount": Lipinski.NHOHCount,
        "NOCount": Lipinski.NOCount,
        "RotatableBonds": Lipinski.NumRotatableBonds,
        "RingCount": Lipinski.RingCount,
        "AromaticRingCount": Lipinski.NumAromaticRings,
        "AliphaticRingCount": Lipinski.NumAliphaticRings,
        "HeavyAtomCount": Lipinski.HeavyAtomCount,
        "HeteroAtomCount": Lipinski.NumHeteroatoms,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "FormalCharge": Chem.GetFormalCharge,
        "LabuteASA": rdMolDescriptors.CalcLabuteASA,
        "NumValenceElectrons": Descriptors.NumValenceElectrons,
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    }
    # Optional descriptors differ slightly across RDKit releases.
    optional = {
        "AmideBondCount": getattr(rdMolDescriptors, "CalcNumAmideBonds", None),
        "BridgeheadAtomCount": getattr(rdMolDescriptors, "CalcNumBridgeheadAtoms", None),
        "SpiroAtomCount": getattr(rdMolDescriptors, "CalcNumSpiroAtoms", None),
        "BertzCT": getattr(Descriptors, "BertzCT", None),
        "BalabanJ": getattr(Descriptors, "BalabanJ", None),
    }
    out: Dict[str, float] = {}
    for name, fn in {**fns, **{k: v for k, v in optional.items() if v is not None}}.items():
        try:
            x = float(fn(mol))
            if math.isfinite(x):
                out[name] = x
        except Exception:
            pass

    heavy = max(float(out.get("HeavyAtomCount", 0.0)), 1.0)
    atoms = max(float(mol.GetNumAtoms()), 1.0)
    aromatic_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    carbon_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6))
    out["AromaticAtomFraction"] = aromatic_atoms / heavy
    out["HeteroAtomFraction"] = float(out.get("HeteroAtomCount", 0.0)) / heavy
    out["CarbonAtomFraction"] = carbon_atoms / atoms
    out["TPSA_per_HeavyAtom"] = float(out.get("TPSA", 0.0)) / heavy
    out["MolLogP_per_HeavyAtom"] = float(out.get("MolLogP", 0.0)) / heavy
    asa = float(out.get("LabuteASA", 0.0))
    out["PolarSurfaceFraction_proxy"] = float(out.get("TPSA", 0.0)) / asa if asa > 0 else 0.0

    out.update(_acid_base_proxy_descriptors(mol))
    out.update(_gasteiger_charge_descriptors(mol))
    return out

def extended_descriptors_from_smiles(smiles: str) -> Dict[str, float]:
    """Return optional high-dimensional VSA/BCUT descriptors for export/benchmarking.

    These are intentionally kept separate from the default local-RF similarity
    so that a small calibration set is not silently dominated by dozens of
    correlated descriptors.
    """
    return _extended_vsa_descriptors(_mol(smiles))


def _desc3d(mol) -> Tuple[Dict[str, float], str]:
    if mol is None or not RDKIT_AVAILABLE:
        return {}, "rdkit_unavailable"
    try:
        m = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        code = AllChem.EmbedMolecule(m, params)
        if code != 0:
            return {}, f"embed_failed_{code}"
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        fns = {
            "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration,
            "Asphericity": rdMolDescriptors.CalcAsphericity,
            "Eccentricity": rdMolDescriptors.CalcEccentricity,
            "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex,
            "PMI1": rdMolDescriptors.CalcPMI1,
            "PMI2": rdMolDescriptors.CalcPMI2,
            "PMI3": rdMolDescriptors.CalcPMI3,
            "NPR1": rdMolDescriptors.CalcNPR1,
            "NPR2": rdMolDescriptors.CalcNPR2,
            "InertialShapeFactor": getattr(rdMolDescriptors, "CalcInertialShapeFactor", lambda _m: float("nan")),
            "PBF": getattr(rdMolDescriptors, "CalcPBF", lambda _m: float("nan")),
        }
        out = {}
        for name, fn in fns.items():
            try:
                x = float(fn(m))
                if math.isfinite(x):
                    out[name] = x
            except Exception:
                pass
        try:
            volume = float(AllChem.ComputeMolVolume(m))
            if math.isfinite(volume):
                out["MolVolume3D"] = volume
        except Exception:
            pass
        return out, "ok" if out else "no_3d_descriptors"
    except Exception as e:
        return {}, f"3d_failed:{e}"


def _raw_value(row: PreparedRow, names: Sequence[str]) -> str:
    norm = {_norm_header(k): k for k in row.raw if not str(k).startswith("__source_")}
    for name in names:
        n = _norm_header(name)
        if n in norm:
            value = _text(row.raw.get(norm[n], ""))
            if value:
                return value
    return ""


def _reaction_product(row: PreparedRow, lib: StructureLibrary, core_id: str, expected_formula: str):
    warnings: List[str] = []
    rule = lib.rules.get(core_id)
    if not RDKIT_AVAILABLE or not rule or not rule.reaction_smarts:
        return None, warnings
    mols = {"CORE": _mol(lib.cores.get(core_id).smiles if lib.cores.get(core_id) else "")}
    for role in ("A", "B", "C"):
        label = _text(row.combo_parts.get(role, {}).get("label", ""))
        entry = lib.entries.get((role, label))
        mols[role] = _mol(entry.smiles if entry else "")
        if entry is None:
            warnings.append(f'Not found: {role}/{label}')
    reactants = []
    for role in rule.reactant_order:
        m = mols.get(role)
        if m is None:
            warnings.append(f'Required reaction component {role} has no structure.')
            return None, warnings
        reactants.append(m)
    try:
        rxn = ReactionFromSmarts(rule.reaction_smarts)
        sets = rxn.RunReactants(tuple(reactants))
    except Exception as e:
        return None, warnings + [f'Reaction SMARTS failed: {e}']
    candidates = []
    for pset in sets:
        for p in pset:
            try:
                Chem.SanitizeMol(p)
                smi = _canonical(p)
                if not smi:
                    continue
                formula = rdMolDescriptors.CalcMolFormula(p)
                match = 1 if expected_formula and formula_equal(formula, expected_formula) else 0
                candidates.append((match, p.GetNumHeavyAtoms(), smi, p))
            except Exception:
                pass
    if not candidates:
        warnings.append('Reaction SMARTS produced no usable product.')
        return None, warnings
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    if len(candidates) > 1:
        warnings.append(f'Reaction SMARTS generated {len(candidates)} products; selected by formula match and heavy-atom count.')
    return candidates[0][3], warnings


def enrich_rows(
    rows: Sequence[PreparedRow],
    lib: StructureLibrary,
    *,
    use_3d: bool = False,
    privacy_mode: bool = True,
    product_smiles_col: str = "",
) -> None:
    cache: Dict[str, Tuple[Dict[str, float], Optional[np.ndarray], Dict[str, float], str]] = {}

    def features(smiles: str):
        m = _mol(smiles)
        if m is None:
            return "", {}, None, "smiles_parse_failed"
        can = _canonical(m)
        if can in cache:
            d2, fp, d3, st = cache[can]
        else:
            d2 = _desc2d(m)
            fp = _fp(m)
            d3, st = _desc3d(m) if use_3d else ({}, "not_requested")
            cache[can] = (dict(d2), fp, dict(d3), st)
        d = dict(d2)
        d.update(d3)
        return can, d, fp, st

    for row in rows:
        warnings: List[str] = []
        core_id, ws = resolve_core_id(row, lib)
        # Missing CORE is not an error when a complete Product_SMILES is supplied by the master table.
        if lib.cores or lib.entries or lib.rules:
            warnings.extend(ws)
            formula_calc, ws = calculate_product_formula(row, lib, core_id)
            warnings.extend(ws)
        else:
            formula_calc = ""
        formula_match = "unknown"
        if row.formula and formula_calc:
            formula_match = "yes" if formula_equal(row.formula, formula_calc) else "no"
            if formula_match == "no":
                warnings.append(f'Enumerated Formula={row.formula} differs from structure stoichiometry Formula={formula_calc}; mismatch')

        direct = _text(row.raw.get(product_smiles_col, "")) if product_smiles_col else ""
        if not direct:
            direct = _raw_value(row, ["Product_SMILES", "Product SMILES", 'Product_SMILES'])
        product_entry = lib.products.get(_combo_key(row.combo)) if row.combo else None
        if not direct and product_entry:
            direct = product_entry.product_smiles or product_entry.smiles

        product_mol = _mol(direct) if direct else None
        method = "direct_product_smiles" if product_mol is not None else ""
        if direct and product_mol is None:
            warnings.append('Product_SMILES could not be parsed.')
        if product_mol is None and core_id:
            product_mol, ws = _reaction_product(row, lib, core_id, row.formula or formula_calc)
            warnings.extend(ws)
            if product_mol is not None:
                method = "reaction_smarts"

        # A complete product structure is the strongest formula check.  This also
        # prevents false mismatches when no separate CORE structure library is supplied.
        if product_mol is not None and RDKIT_AVAILABLE:
            try:
                structure_formula = rdMolDescriptors.CalcMolFormula(product_mol)
            except Exception:
                structure_formula = ""
            if structure_formula:
                if formula_calc and not formula_equal(formula_calc, structure_formula):
                    warnings.append(
                        f'Structure stoichiometry Formula={formula_calc} differs from Product_SMILES Formula={structure_formula}; the complete product formula is retained for structure-coverage review.'
                    )
                formula_calc = structure_formula
                if row.formula:
                    formula_match = "yes" if formula_equal(row.formula, structure_formula) else "no"
                    if formula_match == "no":
                        warnings.append(f'Enumerated Formula={row.formula} differs from Product_SMILES Formula={structure_formula}; mismatch')
                else:
                    formula_match = "unknown"

        descriptor: Dict[str, float] = {}
        product_fp = None
        component_fp: Dict[str, np.ndarray] = {}
        three_d = "not_requested"
        product_smiles = ""
        structure_hash = ""
        status = "no_structure_features"

        if product_mol is not None:
            can = _canonical(product_mol)
            can, desc, product_fp, three_d = features(can)
            descriptor.update(desc)
            product_smiles = can
            structure_hash = hashlib.sha256(can.encode("utf-8")).hexdigest()[:20]
            status = "product_structure_ready"

        for role in ("A", "B", "C", "CORE"):
            if role == "CORE":
                entry = lib.cores.get(core_id)
            else:
                label = _text(row.combo_parts.get(role, {}).get("label", ""))
                entry = lib.entries.get((role, label))
            if entry is None:
                continue
            part_desc = dict(entry.numeric)
            if entry.smiles:
                _can, d, fp, st = features(entry.smiles)
                for k, v in d.items():
                    part_desc.setdefault(k, v)
                if fp is not None:
                    component_fp[role] = fp
            for k, v in part_desc.items():
                descriptor.setdefault(f"{role}_{k}", float(v))

        if status != "product_structure_ready" and (component_fp or descriptor):
            status = "fragment_core_features_ready"
            method = "fragment_core_fusion"
            structure_hash = hashlib.sha256(f"{core_id}|{row.combo}".encode("utf-8")).hexdigest()[:20]
            three_d = "component_level" if use_3d else "not_requested"
        elif status == "no_structure_features":
            method = "formula_combo_fallback"

        try:
            counts = parse_formula(row.formula or formula_calc)
            total = float(sum(counts.values())) or 1.0
            # Keep a stable element descriptor schema.  An element that is absent
            # from a valid formula is chemically zero, not a missing value.  This
            # prevents F/Cl/Br/I/S columns from being incorrectly reported as
            # "unavailable" merely because most compounds do not contain them.
            standard_elements = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Na", "K")
            for el in standard_elements:
                n = float(counts.get(el, 0))
                descriptor[f"Formula_{el}"] = n
                descriptor[f"FormulaFrac_{el}"] = n / total
            for el, n_raw in counts.items():
                if el in standard_elements:
                    continue
                n = float(n_raw)
                descriptor[f"Formula_{el}"] = n
                descriptor[f"FormulaFrac_{el}"] = n / total
        except Exception:
            pass

        row.core_id = core_id
        row.structure_status = status
        row.structure_method = method
        row.structure_hash = structure_hash
        row.product_formula_calc = formula_calc
        row.formula_match = formula_match
        row.three_d_status = three_d
        row.structure_features = descriptor
        row.structure_fingerprint = product_fp
        row.component_fingerprints = component_fp
        row.structure_warnings = warnings
        row.product_smiles = "" if privacy_mode else product_smiles


def tanimoto(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0
    if aa.shape != bb.shape or aa.size == 0:
        return None
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def descriptor_similarity(a: PreparedRow, b: PreparedRow) -> Optional[float]:
    da = getattr(a, "structure_features", {}) or {}
    db = getattr(b, "structure_features", {}) or {}
    common = [k for k in da if k in db]
    if not common:
        return None
    d = []
    for k in common:
        try:
            x, y = float(da[k]), float(db[k])
            if math.isfinite(x) and math.isfinite(y):
                d.append(abs(x-y)/(abs(x)+abs(y)+1.0))
        except Exception:
            pass
    return float(math.exp(-4.0*np.mean(d))) if d else None


def structure_similarity(a: PreparedRow, b: PreparedRow) -> Optional[float]:
    prod = tanimoto(getattr(a, "structure_fingerprint", None), getattr(b, "structure_fingerprint", None))
    ca = getattr(a, "component_fingerprints", {}) or {}
    cb = getattr(b, "component_fingerprints", {}) or {}
    vals, weights = [], []
    for role, w in (("A",1.0),("B",1.0),("C",1.0),("CORE",0.6)):
        s = tanimoto(ca.get(role), cb.get(role))
        if s is not None:
            vals.append(s*w); weights.append(w)
    comp = sum(vals)/sum(weights) if weights else None
    desc = descriptor_similarity(a,b)
    pieces = []
    if prod is not None:
        pieces.append((prod,0.70))
    if comp is not None:
        pieces.append((comp,0.22 if prod is not None else 0.72))
    if desc is not None:
        pieces.append((desc,0.08 if prod is not None else 0.28))
    return float(sum(v*w for v,w in pieces)/sum(w for _,w in pieces)) if pieces else None


def same_core(a: PreparedRow, b: PreparedRow) -> bool:
    ca = _text(getattr(a, "core_id", ""))
    cb = _text(getattr(b, "core_id", ""))
    return bool(ca and cb and ca == cb)


def audit_row(row: PreparedRow, privacy_mode: bool = True) -> Dict[str, object]:
    return {
        "Source_file": _text(row.raw.get("__source_file__", "")),
        "Source_sheet": _text(row.raw.get("__source_sheet__", "")),
        "Source_row": row.raw.get("__source_row__", row.source_index),
        "Source_index": row.source_index,
        "Name": row.name,
        "Formula": row.formula,
        "Combo": row.combo,
        "Product_Master_Match": _text(row.raw.get("__product_master_match__", "")),
        "Product_Master_Match_Method": _text(row.raw.get("__product_master_match_method__", "")),
        "Product_Master_Exact_Key": _text(row.raw.get("__product_master_exact_key__", "")),
        "Product_Master_Formula_Key": _text(row.raw.get("__product_master_formula_key__", "")),
        "Product_Master_Index_Key": _text(row.raw.get("__product_master_index_key__", "")),
        "Product_Master_ABC_Key": _text(row.raw.get("__product_master_abc_key__", "")),
        "Product_Master_Used_Key": _text(row.raw.get("__product_master_used_key__", "")),
        "Product_Master_Candidate_Count": row.raw.get("__product_master_candidate_count__", ""),
        "Product_Master_Verification_Mode": _text(row.raw.get("__product_master_verification_mode__", "")),
        "Product_Master_Join_Status": _text(row.raw.get("__product_master_join_status__", "")),
        "Product_Master_Formula_Flag": _text(row.raw.get("__product_master_formula_flag__", "")),
        "Product_Master_Formula_Check": _text(row.raw.get("__product_master_formula_check__", "")),
        "Product_Master_Component_Formula_Check": _text(row.raw.get("__product_master_component_formula_check__", "")),
        "Product_Master_Internal_Formula_Check": _text(row.raw.get("__product_master_internal_formula_check__", "")),
        "Product_Master_Target_Formula": _text(row.raw.get("__product_master_target_formula__", "")),
        "Product_Master_Master_Formula": _text(row.raw.get("__product_master_master_formula__", "")),
        "Product_Master_Master_Product_Formula": _text(row.raw.get("__product_master_master_product_formula__", "")),
        "Product_Master_Formula_Note": _text(row.raw.get("__product_master_formula_note__", "")),
        "Product_Master_Unmatched_Reason": _text(row.raw.get("__product_master_unmatched_reason__", "")),
        "Core_ID": getattr(row, "core_id", ""),
        "Structure_status": getattr(row, "structure_status", ""),
        "Structure_method": getattr(row, "structure_method", ""),
        "Structure_hash": getattr(row, "structure_hash", ""),
        "Product_formula_calc": getattr(row, "product_formula_calc", ""),
        "Formula_match": getattr(row, "formula_match", ""),
        "3D_status": getattr(row, "three_d_status", ""),
        "Descriptor_count": len(getattr(row, "structure_features", {}) or {}),
        "Product_fingerprint": "Yes" if getattr(row, "structure_fingerprint", None) is not None else "No",
        "Component_fingerprint_count": len(getattr(row, "component_fingerprints", {}) or {}),
        "Product_SMILES": "" if privacy_mode else getattr(row, "product_smiles", ""),
        "Warnings": "; ".join(getattr(row, "structure_warnings", []) or []),
    }
