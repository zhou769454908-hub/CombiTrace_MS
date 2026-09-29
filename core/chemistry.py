
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =========================
#  Monoisotopic masses (most abundant isotope)
#  Elements: H~Zn + Br + I
# =========================
MONO_MASS: Dict[str, float] = {
    
    "H": 1.00782503223,
    "D": 2.01410177812,
    "He": 4.00260325413,
    "Li": 7.0160034366,
    "Be": 9.012183065,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Ne": 19.9924401762,
    
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Ar": 39.9623831237,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    
    "Sc": 44.95590828,
    "Ti": 47.94794198,
    "V": 50.94395704,
    "Cr": 51.94050623,
    "Mn": 54.93804391,
    "Fe": 55.93493633,
    "Co": 58.93319429,
    "Ni": 57.93534241,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    # extras
    "Br": 78.9183376,
    "I": 126.9044719,
}

SUPPORTED_ELEMENTS = set(MONO_MASS.keys())


# =========================

#  (common adduct exact mass shifts)
# =========================
MASS_H_PLUS = 1.007276
MASS_NH4_PLUS = 18.033823
MASS_NA_PLUS = 22.989218
MASS_K_PLUS = 38.963158

# electron mass in u
ELECTRON_MASS_U = 5.485799090441e-4
# Li+ ~ atomic Li - electron
MASS_LI_PLUS = float(MONO_MASS["Li"] - ELECTRON_MASS_U)  # ~7.015455

# Formic-acid / formate exact-mass constants for negative-ESI XIC.
# Legacy formate aliases remain accepted internally for backward-compatible input parsing.
MASS_FORMIC_ACID_NEUTRAL = float(
    MONO_MASS["C"] + 2.0 * MONO_MASS["H"] + 2.0 * MONO_MASS["O"]
)
MASS_FORMATE_ANION = float(
    MONO_MASS["C"] + MONO_MASS["H"] + 2.0 * MONO_MASS["O"] + ELECTRON_MASS_U
)
MASS_MH_TO_FORMATE = float(MASS_FORMATE_ANION + MASS_H_PLUS)


@dataclass(frozen=True)
class Adduct:
    key: str
    display: str
    mass_shift: float
    charge: int
    polarity: str  # "positive" | "negative"


POS_ADDUCTS: Dict[str, Adduct] = {
    "H": Adduct("H", "[M+H]+", +MASS_H_PLUS, +1, "positive"),
    "NH4": Adduct("NH4", "[M+NH4]+", +MASS_NH4_PLUS, +1, "positive"),
    "Na": Adduct("Na", "[M+Na]+", +MASS_NA_PLUS, +1, "positive"),
    "K": Adduct("K", "[M+K]+", +MASS_K_PLUS, +1, "positive"),
}


NEG_ADDUCTS: Dict[str, Adduct] = {
    "H": Adduct("H", "[M-H]-", -MASS_H_PLUS, -1, "negative"),
    "FA": Adduct("FA", "[M+HCOO]-", +MASS_FORMATE_ANION, -1, "negative"),
    "Na": Adduct("Na", "[M-Na]-", -MASS_NA_PLUS, -1, "negative"),
    "K": Adduct("K", "[M-K]-", -MASS_K_PLUS, -1, "negative"),
    "Li": Adduct("Li", "[M-Li]-", -MASS_LI_PLUS, -1, "negative"),
}


def calc_mz(neutral_mass: float, adduct: Adduct) -> float:
    num = float(neutral_mass) + float(adduct.mass_shift)
    if num <= 0:
        return float("nan")
    return float(num / abs(int(adduct.charge)))


# =========================
#  Formula parsing (supports parentheses + hydrate dot)
# =========================
def format_formula_hill(counts: Dict[str, int]) -> str:
    counts = {k: int(v) for k, v in counts.items() if int(v) != 0}
    if not counts:
        return ""
    parts: List[str] = []

    def add(el: str) -> None:
        c = counts.get(el, 0)
        if c:
            parts.append(el + (str(c) if c != 1 else ""))

    if "C" in counts:
        add("C")
        add("H")
        for el in sorted(k for k in counts.keys() if k not in ("C", "H")):
            c = counts[el]
            parts.append(el + (str(c) if c != 1 else ""))
    else:
        for el in sorted(counts.keys()):
            c = counts[el]
            parts.append(el + (str(c) if c != 1 else ""))

    return "".join(parts)


def parse_formula(formula: str) -> Dict[str, int]:
    '- C6H12O6 / NaCl / C6H5Br'

    s = str(formula).strip()
    if not s:
        raise ValueError("Empty formula")

    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(\d+)?[+-]+$", "", s)  # strip trailing charge

    parts = re.split(r"[\u00B7\.]", s)

    total: Dict[str, int] = {}
    for part in parts:
        if not part:
            continue

        # allow leading multiplier, e.g. 5H2O
        m = re.match(r"^(\d+)(.*)$", part)
        mult = 1
        body = part
        if m:
            mult = int(m.group(1))
            body = m.group(2)
            if not body:
                raise ValueError(f"Invalid formula component: {part}")

        stack: List[Dict[str, int]] = [dict()]
        i = 0
        while i < len(body):
            ch = body[i]

            if ch == "(":
                stack.append(dict())
                i += 1
                continue

            if ch == ")":
                i += 1
                j = i
                while j < len(body) and body[j].isdigit():
                    j += 1
                group_mult = int(body[i:j]) if j > i else 1
                i = j

                if len(stack) < 2:
                    raise ValueError(f"Unmatched ')' in {formula}")

                group = stack.pop()
                for el, cnt in group.items():
                    stack[-1][el] = stack[-1].get(el, 0) + cnt * group_mult
                continue

            if ch.isupper():
                el = ch
                i += 1
                if i < len(body) and body[i].islower():
                    el += body[i]
                    i += 1

                if el not in SUPPORTED_ELEMENTS:
                    raise ValueError(f"Unsupported element '{el}' in formula '{formula}'")

                j = i
                while j < len(body) and body[j].isdigit():
                    j += 1
                num = int(body[i:j]) if j > i else 1
                i = j

                stack[-1][el] = stack[-1].get(el, 0) + num
                continue

            raise ValueError(f"Unexpected character '{ch}' in formula '{formula}'")

        if len(stack) != 1:
            raise ValueError(f"Unmatched '(' in {formula}")

        comp = stack[0]
        for el, cnt in comp.items():
            total[el] = total.get(el, 0) + cnt * mult

    return total


def monoisotopic_mass(counts: Dict[str, int]) -> float:
    return float(sum(MONO_MASS[el] * int(cnt) for el, cnt in counts.items()))


def compute_neutral_mass_from_formula(formula: str) -> Tuple[str, float]:
    counts = parse_formula(formula)
    hill = format_formula_hill(counts)
    return hill, monoisotopic_mass(counts)


def guess_polarity_from_filter_string(filter_string: Optional[str]) -> Optional[str]:
    '"FTMS - p ..." -> negative'

    if not filter_string:
        return None
    s = filter_string.lower()
    
    if "+ p" in s or "+p" in s:
        return "positive"
    if "- p" in s or "-p" in s:
        return "negative"
    return None


def normalize_polarity(val: str) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s:
        return None
    if s in ("positive", "+", "pos", "p", "+"):
        return "positive"
    if s in ("negative", "-", "neg", "n"):
        return "negative"
    
    if 'positive' in s or '\u6b63' in s:
        return "positive"
    if 'negative' in s or '\u8d1f' in s:
        return "negative"
    return None


def parse_adduct(adduct_text: Optional[str], *, default_polarity: str) -> Adduct:
    '- "H" / "Na" / "NH4" / "K" / "Li" / "FA" / "HCOO"\n    - "[M+H]+" / "M+H" / "+H"\n    - "[M-H]-" / "M-H" / "-H"\n    - "[M+HCOO]-" / "M+HCOO"'

    dp = (default_polarity or "positive").lower().strip()
    if dp not in ("positive", "negative"):
        dp = "positive"

    s = str(adduct_text or "").strip()
    if not s:
        return POS_ADDUCTS["H"] if dp == "positive" else NEG_ADDUCTS["H"]

    
    s0 = s
    s = s.strip()
    s = s.replace(" ", "")
    s_up = s.upper()

    
    if s in POS_ADDUCTS:
        return POS_ADDUCTS[s]
    if s in NEG_ADDUCTS:
        return NEG_ADDUCTS[s]
    if s_up in {"FA", "HCOO", "FORMATE"}:
        return NEG_ADDUCTS["FA"]

    # Formate must be checked before generic +H because HCOO contains "+H".
    if (
        "HCOO" in s_up
        or "FORMATE" in s_up
        or "FA-H" in s_up
        or s_up in {"+FA", "+HCOO"}
    ):
        return NEG_ADDUCTS["FA"]

    
    # positive
    if "+H" in s_up and "+" in s_up:
        return POS_ADDUCTS["H"]
    if "+NA" in s_up and "+" in s_up:
        return POS_ADDUCTS["Na"]
    if "+K" in s_up and "+" in s_up:
        return POS_ADDUCTS["K"]
    if "+NH4" in s_up and "+" in s_up:
        return POS_ADDUCTS["NH4"]

    # negative
    if "-H" in s_up and "-" in s_up:
        return NEG_ADDUCTS["H"]
    if "-NA" in s_up and "-" in s_up:
        return NEG_ADDUCTS["Na"]
    if "-K" in s_up and "-" in s_up:
        return NEG_ADDUCTS["K"]
    if "-LI" in s_up and "-" in s_up:
        return NEG_ADDUCTS["Li"]

    
    if s_up.startswith("+"):
        k = s_up[1:]
        # +NA +K +NH4
        if k == "H":
            return POS_ADDUCTS["H"]
        if k == "NA":
            return POS_ADDUCTS["Na"]
        if k == "K":
            return POS_ADDUCTS["K"]
        if k == "NH4":
            return POS_ADDUCTS["NH4"]

    if s_up.startswith("-"):
        k = s_up[1:]
        if k == "H":
            return NEG_ADDUCTS["H"]
        if k == "NA":
            return NEG_ADDUCTS["Na"]
        if k == "K":
            return NEG_ADDUCTS["K"]
        if k == "LI":
            return NEG_ADDUCTS["Li"]

    
    return POS_ADDUCTS["H"] if dp == "positive" else NEG_ADDUCTS["H"]
