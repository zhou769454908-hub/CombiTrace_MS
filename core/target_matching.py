
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .chemistry import (
    Adduct,
    POS_ADDUCTS,
    NEG_ADDUCTS,
    calc_mz,
    compute_neutral_mass_from_formula,
    guess_polarity_from_filter_string,
    normalize_polarity,
    parse_adduct,
)
from .models import AveragedSpectrum
from .targets_csv import TargetSpec
from .isotope_theory import isotope_peaks_mz


@dataclass
class MatchRow:
    name: str
    formula_input: str
    formula_hill: str
    adduct: str
    polarity: str
    theoretical_mz: float
    observed_mz: Optional[float]
    ppm_error: Optional[float]
    intensity: float
    rel_abundance: float
    found: bool
    note: str = ""


def _find_peak_in_window(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    theo_mz: float,
    ppm_tol: float,
) -> Optional[int]:

    if mz.size == 0:
        return None

    width = float(theo_mz) * float(ppm_tol) / 1e6
    lo = float(theo_mz) - width
    hi = float(theo_mz) + width

    
    i0 = int(np.searchsorted(mz, lo, side="left"))
    i1 = int(np.searchsorted(mz, hi, side="right"))
    if i1 <= i0:
        return None

    sub = intensity[i0:i1]
    if sub.size == 0:
        return None
    j = int(np.argmax(sub))
    return i0 + j



def _candidate_adducts_for_polarity(pol: str) -> List[Adduct]:
    p = (pol or "positive").lower().strip()
    if p.startswith("neg"):
        
        return [NEG_ADDUCTS[k] for k in ("H", "FA", "Na", "K", "Li") if k in NEG_ADDUCTS]
    
    order = ("H", "Na", "K", "NH4")
    return [POS_ADDUCTS[k] for k in order if k in POS_ADDUCTS]


def _candidate_adducts_for_mode(
    mode: str,
    *,
    pol_hint: str,
    inferred_pol: str,
    default_polarity: str,
) -> Tuple[List[Adduct], str]:
    '(candidates, fallback_polarity)'

    m = (mode or "auto").strip().lower()
    if m.startswith("auto_"):
        m = m[len("auto_") :]

    pol_hint_n = (pol_hint or "").strip().lower()
    inferred_pol_n = (inferred_pol or "").strip().lower()
    default_pol_n = (default_polarity or "positive").strip().lower()

    # helper: build pos / neg lists with stable ordering
    pos_order = ("H", "Na", "K", "NH4")
    neg_order = ("H", "FA", "Na", "K", "Li")

    pos_list: List[Adduct] = [POS_ADDUCTS[k] for k in pos_order if k in POS_ADDUCTS]
    neg_list_all: List[Adduct] = [NEG_ADDUCTS[k] for k in neg_order if k in NEG_ADDUCTS]
    neg_list_honly: List[Adduct] = [NEG_ADDUCTS["H"]] if "H" in NEG_ADDUCTS else []

    # default fallback polarity: prefer inferred, then pol_hint, else default
    fallback_pol = normalize_polarity(inferred_pol_n) or normalize_polarity(pol_hint_n) or normalize_polarity(default_pol_n) or "positive"

    if m in ("auto", "by_polarity", "polarity"):
        return _candidate_adducts_for_polarity(pol_hint), (normalize_polarity(pol_hint_n) or fallback_pol)

    if m in ("pos_all", "positive", "pos"):
        return pos_list, "positive"

    if m in ("neg_all", "negative", "neg"):
        return neg_list_all, "negative"

    if m in ("posneg_neghonly", "posneg_honly", "posneg_neg_honly", "posneg_only_negh"):
        # +H then -H, then other positive adducts
        cand: List[Adduct] = []
        # +H first
        if "H" in POS_ADDUCTS:
            cand.append(POS_ADDUCTS["H"])
        # -H next (only)
        cand.extend(neg_list_honly)
        # other positive
        for k in ("Na", "K", "NH4"):
            if k in POS_ADDUCTS:
                cand.append(POS_ADDUCTS[k])
        return cand, fallback_pol

    if m in ("posneg_all", "all", "posneg"):
        # +H then -H, then other positive, then other negative
        cand: List[Adduct] = []
        if "H" in POS_ADDUCTS:
            cand.append(POS_ADDUCTS["H"])
        if "H" in NEG_ADDUCTS:
            cand.append(NEG_ADDUCTS["H"])
        for k in ("Na", "K", "NH4"):
            if k in POS_ADDUCTS:
                cand.append(POS_ADDUCTS[k])
        for k in ("FA", "Na", "K", "Li"):
            if k in NEG_ADDUCTS:
                cand.append(NEG_ADDUCTS[k])
        return cand, fallback_pol

    # unknown mode -> keep old behavior
    return _candidate_adducts_for_polarity(pol_hint), (normalize_polarity(pol_hint_n) or fallback_pol)



def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _score_adduct_candidate(
    mz: np.ndarray,
    inten: np.ndarray,
    nl: float,
    *,
    formula: str,
    neutral_mass: float,
    adduct: Adduct,
    ppm_tol: float,
) -> Optional[Tuple[Tuple[float, float, float, float], int, float]]:
    '- ppm_error'
    theo_mz = calc_mz(float(neutral_mass), adduct)
    if not np.isfinite(theo_mz) or theo_mz <= 0:
        return None

    idx0 = _find_peak_in_window(mz, inten, theo_mz=theo_mz, ppm_tol=float(ppm_tol))
    if idx0 is None:
        return None

    obs_mz = float(mz[idx0])
    obs_int = float(inten[idx0])
    rel0 = (obs_int / nl * 100.0) if nl > 0 else 0.0
    ppm_err = (obs_mz - float(theo_mz)) / float(theo_mz) * 1e6

    
    theo_peaks = []
    try:
        theo_peaks = isotope_peaks_mz(formula, adduct, max_peaks=8, min_rel=0.5)[:5]
    except Exception:
        theo_peaks = []

    matched = 0
    sim = 0.0
    if theo_peaks and nl > 0:
        theo_vec = np.array([r for _, r in theo_peaks], dtype=float)
        obs_vec = []
        for m_theo, _r in theo_peaks:
            idx = _find_peak_in_window(mz, inten, theo_mz=float(m_theo), ppm_tol=float(ppm_tol))
            if idx is None:
                obs_vec.append(0.0)
            else:
                matched += 1
                obs_vec.append(float(inten[idx]) / nl * 100.0)
        obs_vec = np.array(obs_vec, dtype=float)
        sim = _cosine_similarity(theo_vec, obs_vec)

    
    score = (float(matched), float(sim), float(rel0), -abs(float(ppm_err)))
    return score, int(idx0), float(ppm_err)



def match_targets_to_spectrum(
    spec: AveragedSpectrum,
    targets: Iterable[TargetSpec],
    *,
    default_ppm: float = 5.0,
    default_polarity: str = "positive",
    min_rel_for_found: float = 0.0,
    adduct_mode: str = "auto",  # "auto" | "force"
    forced_adduct: str = "",  # e.g. "M+H" / "M-H" / "+H" / "-H"
    prefer_proton: bool = True,
) -> List[MatchRow]:

    mz = spec.mz.astype(float) if spec.mz is not None else np.array([], dtype=float)
    inten = spec.intensity.astype(float) if spec.intensity is not None else np.array([], dtype=float)
    nl = float(np.max(inten)) if inten.size else 0.0

    inferred_pol = guess_polarity_from_filter_string(spec.filter_string) or default_polarity
    inferred_pol = normalize_polarity(inferred_pol) or "positive"

    adduct_mode = (adduct_mode or "auto").strip().lower()
    forced_adduct = str(forced_adduct or "").strip()

    out: List[MatchRow] = []
    for t in targets:
        note_parts: List[str] = []

        
        pol = normalize_polarity(getattr(t, "polarity", "") or "") or inferred_pol

        # 2) formula -> neutral mass
        formula_input = (t.formula or "").strip()
        formula_hill = ""
        neutral_mass: Optional[float] = None
        if not formula_input and t.theoretical_mz is None:
            note_parts.append("missing formula")
        if formula_input:
            try:
                formula_hill, neutral_mass = compute_neutral_mass_from_formula(formula_input)
            except Exception as e:
                note_parts.append(f"formula error: {e}")
                neutral_mass = None

        # 3) ppm tolerance
        ppm_tol = float(t.ppm) if t.ppm is not None else float(default_ppm)
        ppm_tol = max(0.1, ppm_tol)

        # 4) adduct / theoretical m/z
        adduct_text = (getattr(t, "adduct", "") or "").strip()
        
        if adduct_mode == "force" and forced_adduct:
            adduct_text = forced_adduct
        chosen_ad: Optional[Adduct] = None
        theo_mz: Optional[float] = None
        chosen_idx: Optional[int] = None
        ppm_err: Optional[float] = None

        if t.theoretical_mz is not None:
            
            theo_mz = float(t.theoretical_mz)
            chosen_ad = parse_adduct(adduct_text, default_polarity=pol) if adduct_text else parse_adduct("", default_polarity=pol)
            pol = chosen_ad.polarity
        else:
            if neutral_mass is None or not np.isfinite(float(neutral_mass)):
                out.append(
                    MatchRow(
                        name=t.name or "",
                        formula_input=formula_input,
                        formula_hill=formula_hill or "",
                        adduct="",
                        polarity=pol,
                        theoretical_mz=float("nan"),
                        observed_mz=None,
                        ppm_error=None,
                        intensity=0.0,
                        rel_abundance=0.0,
                        found=False,
                        note="; ".join(note_parts) if note_parts else "invalid formula",
                    )
                )
                continue

            if adduct_text:
                chosen_ad = parse_adduct(adduct_text, default_polarity=pol)
                pol = chosen_ad.polarity
                theo_mz = calc_mz(float(neutral_mass), chosen_ad)
            else:
                
                cand_ads, fallback_pol = _candidate_adducts_for_mode(
                    adduct_mode,
                    pol_hint=pol,
                    inferred_pol=inferred_pol,
                    default_polarity=default_polarity,
                )

                
                if bool(prefer_proton):
                    prio_ads = [ad for ad in cand_ads if getattr(ad, "key", "") == "H"]
                    best_prio = None
                    for ad in prio_ads:
                        scored = _score_adduct_candidate(
                            mz,
                            inten,
                            nl,
                            formula=(formula_hill or formula_input),
                            neutral_mass=float(neutral_mass),
                            adduct=ad,
                            ppm_tol=float(ppm_tol),
                        )
                        if scored is None:
                            continue
                        score, idx0, ppm0 = scored
                        if best_prio is None or score > best_prio[0]:
                            best_prio = (score, ad, idx0, ppm0)

                    if best_prio is not None:
                        _score, chosen_ad, chosen_idx, ppm_err = best_prio
                        pol = chosen_ad.polarity
                        theo_mz = calc_mz(float(neutral_mass), chosen_ad)
                    else:
                        
                        cand_ads = [ad for ad in cand_ads if getattr(ad, "key", "") != "H"]

                if chosen_ad is None and theo_mz is None:
                    best: Optional[Tuple[float, float, float, float]] = None
                    best_info: Optional[Tuple[Adduct, int, float]] = None

                    for ad in cand_ads:
                        scored = _score_adduct_candidate(
                            mz,
                            inten,
                            nl,
                            formula=(formula_hill or formula_input),
                            neutral_mass=float(neutral_mass),
                            adduct=ad,
                            ppm_tol=float(ppm_tol),
                        )
                        if scored is None:
                            continue
                        score, idx0, ppm0 = scored
                        if best is None or score > best:
                            best = score
                            best_info = (ad, idx0, ppm0)

                    if best_info is None:
                        
                        chosen_ad = parse_adduct("", default_polarity=fallback_pol)
                        pol = chosen_ad.polarity
                        theo_mz = calc_mz(float(neutral_mass), chosen_ad)
                    else:
                        chosen_ad, chosen_idx, ppm_err = best_info
                        pol = chosen_ad.polarity
                        theo_mz = calc_mz(float(neutral_mass), chosen_ad)


        if theo_mz is None or not np.isfinite(theo_mz) or float(theo_mz) <= 0:
            out.append(
                MatchRow(
                    name=t.name or "",
                    formula_input=formula_input,
                    formula_hill=formula_hill or (formula_input or ""),
                    adduct=chosen_ad.display if chosen_ad else "",
                    polarity=pol,
                    theoretical_mz=float("nan"),
                    observed_mz=None,
                    ppm_error=None,
                    intensity=0.0,
                    rel_abundance=0.0,
                    found=False,
                    note="; ".join(note_parts) if note_parts else "invalid theoretical mz",
                )
            )
            continue

        
        if chosen_idx is None:
            idx = _find_peak_in_window(mz, inten, theo_mz=float(theo_mz), ppm_tol=float(ppm_tol))
        else:
            idx = int(chosen_idx)

        if idx is None:
            out.append(
                MatchRow(
                    name=t.name or "",
                    formula_input=formula_input,
                    formula_hill=formula_hill or (formula_input or ""),
                    adduct=chosen_ad.display if chosen_ad else "",
                    polarity=pol,
                    theoretical_mz=float(theo_mz),
                    observed_mz=None,
                    ppm_error=None,
                    intensity=0.0,
                    rel_abundance=0.0,
                    found=False,
                    note="not found",
                )
            )
            continue

        obs_mz = float(mz[int(idx)])
        obs_int = float(inten[int(idx)])
        rel = (obs_int / nl * 100.0) if nl > 0 else 0.0
        if ppm_err is None:
            ppm_err = (obs_mz - float(theo_mz)) / float(theo_mz) * 1e6

        found = True
        if float(min_rel_for_found) > 0 and rel < float(min_rel_for_found):
            found = False
            note_parts.append(f"below min_rel ({rel:.3g}% < {min_rel_for_found}%)")

        out.append(
            MatchRow(
                name=t.name or "",
                formula_input=formula_input,
                formula_hill=formula_hill or (formula_input or ""),
                adduct=chosen_ad.display if chosen_ad else "",
                polarity=pol,
                theoretical_mz=float(theo_mz),
                observed_mz=obs_mz,
                ppm_error=float(ppm_err) if ppm_err is not None else None,
                intensity=obs_int,
                rel_abundance=float(rel),
                found=found,
                note="; ".join(note_parts),
            )
        )


    return out


def match_rows_to_dicts(rows: List[MatchRow]) -> List[Dict[str, object]]:
    return [asdict(r) for r in rows]
