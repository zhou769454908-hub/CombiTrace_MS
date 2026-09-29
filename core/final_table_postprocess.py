"""Read-only-input final-table postprocessing; optional targeted RAW centroid recovery.

Theory is calculated from the molecular formula, never from the measured mass.
No random jitter, no invented ppm values, and no matching by row position.
"""
from __future__ import annotations
from .legacy_schema import canonical_header, canonical_sheet

import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from .chemistry import MONO_MASS, ELECTRON_MASS_U
from .final_table_ooxml import Book, colname

VERSION = '18.48.2'
MASS = {k: Decimal(str(v)) for k, v in MONO_MASS.items()}
MASS.update({'[13C]': Decimal('13.00335483507'), '[15N]': Decimal('15.00010889888'),
             '[18O]': Decimal('17.99915961286'), '[2H]': MASS['D']})
ELECTRON = Decimal(str(ELECTRON_MASS_U))
NSUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
# Three sections: positive, negative, zero. PPM must remain numeric on screen.
PPM_FIXED = '+0.000000;-0.000000;0.000000'
PPM_SCI = '+0.000000E+00;-0.000000E+00;0.000000'
PPM_SCI_LIMIT = Decimal('0.0000005')
FORMULA_ALIASES = ('Formula', 'Molecular_formula', 'Product_Formula', 'Molecular Formula', 'Formula', 'Formula')
NAME_ALIASES = ('Name', 'Compound', 'Compound_name', 'Compound_Name', 'Name', 'Sample_Name', 'ID')
OBSERVED_ALIASES = ('Observed_mass', 'Measured_mass', 'Observed_mz', 'Measured_mz', 'Obs_mz', 'Observed m/z', 'Observed_mass', 'Observed_mz')
MASS_ALIASES = OBSERVED_ALIASES + ('Exact_mass', 'Exact mass', 'ExactMass', 'Exact_mass', 'Monoisotopic mass')
KEY_ALIASES = ('Combo', 'Product_SMILES', 'SMILES', 'Combination', 'Combo', 'Combo_key')
IMAGE_ALIASES = ('XIC_PNG', 'XIC_path', 'XIC_filename', 'XIC_filename', 'XIC_image_path')
MODES = ('auto', 'observed_neutral', 'observed_mz', 'reference_neutral', 'raw_mz')


@dataclass
class TableConfig:
    path: str
    sheet: str
    header_row: int = 1
    name_col: int = 0
    formula_col: int = 0
    mass_col: int = 0
    mass_mode: str = 'auto'
    adduct_col: int = 0
    fixed_adduct: str = '[M-H]-'
    key_col: int = 0
    image_col: int = 0
    group_col: int = 0
    image_dir: str = ''
    metadata_path: str = ''
    raw_dir: str = ''
    raw_col: int = 0
    rt_col: int = 0


@dataclass
class Options:
    recursive: bool = True
    allow_unique_formula: bool = False
    image_width: int = 480
    image_height: int = 190
    ppm_limit: float = 5.0
    triplicate_xic: bool = True
    use_source_index: bool = True
    allow_abc_formula: bool = True
    xic_display_mode: str = 'first'  # presentation only; keeps triplicate matching enabled
    observed_search_ppm: float = 20.0
    observed_rt_halfwindow: float = 0.05
    xic_caption_replicate: bool = False
    xic_caption_raw: bool = False
    xic_caption_name: bool = False
    xic_caption_custom: str = ''


def norm(value):
    """Case-insensitive text identity; do NOT erase digits, hyphens or spaces inside names."""
    return unicodedata.normalize('NFKC', str(value or '')).strip().casefold()


def hnorm(value):
    value = canonical_header(value)
    return re.sub(r'[\s_\-./()（）:]+', '', norm(value))


def pick(headers, aliases):
    for name in aliases:
        cols = [c for c, h in headers.items() if hnorm(h) == hnorm(name)]
        if len(cols) == 1:
            return cols[0]
    return 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def decimal_number(value):
    try:
        v = Decimal(str(value).strip())
        return v if v.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def ppm_number_format(value):
    """Choose a numeric Excel format without changing the calculated value.

    The half-unit boundary also uses scientific notation, so an application's
    rounding convention cannot make a nonzero half-unit look like zero.
    """
    number = decimal_number(value)
    if number is not None and number != 0 and abs(number) <= PPM_SCI_LIMIT:
        return PPM_SCI
    return PPM_FIXED


def format_ppm_display(value):
    """GUI display matching the spreadsheet; unavailable is blank, never zero."""
    number = decimal_number(value)
    if number is None:
        return ''
    if number == 0:
        return '0.000000'
    if ppm_number_format(number) == PPM_SCI:
        return format(number, '+.6E')
    return format(number, '+.6f')


def parse_molecular_formula(value):
    """Strict neutral formula parser. Reject charges and ambiguous isotope notation.
    Supports nested ()/[], hydrate dot, D and [13C]/[15N]/[18O]/[2H].
    Charged ion formulas must not silently lose their atom count/charge suffix.
    """
    s = re.sub(r'\s+', '', str(value or '').translate(NSUB))
    if not s:
        raise ValueError('MISSING_FORMULA')
    if len(s) > 1000 or re.search(r'[+\-−⁺⁻]', s):
        raise ValueError('CHARGED_OR_INVALID_FORMULA: use neutral formula and separate adduct')
    # Only explicitly supported isotopes are consumed as tokens.
    tokens = re.findall(r'\[(?:13C|15N|18O|2H)\]|[A-Z][a-z]?|\d+|[()\[\].·]', s)
    if ''.join(tokens) != s:
        raise ValueError('UNSUPPORTED_FORMULA_NOTATION: ' + s)
    if not tokens:
        raise ValueError('EMPTY_FORMULA')
    pos = 0

    def number():
        nonlocal pos
        n = 1
        if pos < len(tokens) and tokens[pos].isdigit():
            n = int(tokens[pos]); pos += 1
        if n < 1 or n > 100000:
            raise ValueError('INVALID_ATOM_COUNT')
        return n

    def group(close=None, depth=0):
        nonlocal pos
        if depth > 25:
            raise ValueError('FORMULA_TOO_DEEP')
        counts = Counter()
        found = False
        while pos < len(tokens):
            token = tokens[pos]
            if token in ('.', '·'):
                break
            if token in (')', ']'):
                if close != token:
                    raise ValueError('UNBALANCED_FORMULA')
                break
            pos += 1
            if token in ('(', '['):
                sub = group(')' if token == '(' else ']', depth+1)
                if pos >= len(tokens) or tokens[pos] != (')' if token == '(' else ']'):
                    raise ValueError('UNBALANCED_FORMULA')
                pos += 1
                n = number()
                for key, v in sub.items():
                    counts[key] += v * n
            elif token in MASS:
                counts[token] += number()
            else:
                raise ValueError('UNSUPPORTED_ELEMENT_OR_TOKEN: ' + token)
            found = True
        if not found:
            raise ValueError('EMPTY_FORMULA_GROUP')
        return counts

    total = Counter()
    while pos < len(tokens):
        multiplier = number()
        unit = group()
        for key, count in unit.items():
            total[key] += count * multiplier
        if pos < len(tokens):
            if tokens[pos] not in ('.', '·'):
                raise ValueError('UNBALANCED_FORMULA')
            pos += 1
            if pos == len(tokens):
                raise ValueError('TRAILING_HYDRATE_DOT')
    if sum(total.values()) > 100000:
        raise ValueError('FORMULA_TOO_LARGE')
    return dict(total)


def formula_key(value):
    try:
        return tuple(sorted(parse_molecular_formula(value).items()))
    except ValueError:
        return None


def theoretical_value(formula, ion=False, adduct=''):
    counts = parse_molecular_formula(formula)
    with localcontext() as ctx:
        ctx.prec = 32
        charge = 0
        if ion:
            a = re.sub(r'\s+', '', str(adduct or '')).replace('−', '-').replace('⁺', '+').replace('⁻', '-')
            aliases = {'[M+FA-H]-': '[M+HCOO]-', '[M+CHO2]-': '[M+HCOO]-', '[M+HCO2]-': '[M+HCOO]-',
                       'M-H': '[M-H]-', 'M+H': '[M+H]+', 'M+Na': '[M+Na]+', 'M+HCOO': '[M+HCOO]-'}
            a = aliases.get(a, a)
            m = re.fullmatch(r'\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])', a)
            if not m:
                raise ValueError('MISSING_OR_UNSUPPORTED_ADDUCT: ' + a)
            mult = int(m.group(1) or '1')
            charge = int(m.group(3) or '1') * (1 if m.group(4) == '+' else -1)
            if not 1 <= abs(charge) <= 20 or not 1 <= mult <= 20:
                raise ValueError('INVALID_ADDUCT_MULTIPLICITY_OR_CHARGE')
            counts = {e: n * mult for e, n in counts.items()}
            for sign, expr in re.findall(r'([+-])([^+-]+)', m.group(2)):
                for e, n in parse_molecular_formula(expr).items():
                    counts[e] = counts.get(e, 0) + (n if sign == '+' else -n)
            if any(n < 0 for n in counts.values()):
                raise ValueError('ADDUCT_REMOVES_UNAVAILABLE_ATOMS')
        mass = sum((MASS[e] * n for e, n in counts.items()), Decimal(0))
        result = (mass - charge * ELECTRON) / abs(charge) if ion else mass
        if result <= 0:
            raise ValueError('NONPOSITIVE_THEORY')
        return +result


def _mode(config, headers):
    if config.mass_mode not in MODES:
        raise ValueError('Unknown mass mode')
    if config.mass_mode != 'auto':
        return config.mass_mode
    label = hnorm(headers.get(config.mass_col, ''))
    if label in {hnorm(x) for x in OBSERVED_ALIASES}:
        return 'observed_mz' if 'mz' in label else 'observed_neutral'
    # Exact_mass in the original enumerator is computed, not measured.
    return 'reference_neutral'


def combo_key(text):
    # Keep the complete key; never treat local A#/B#/C# integers alone as global identities.
    return re.sub(r'\s+', '', str(text or '')).casefold()


def autoconfig(path, sheet=None, label='100'):
    book = Book(path)
    if sheet is None:
        preferred = (['Known_Standards_Predictions', 'Calibration_Features'] if label == '100' else
                     ['Unknown_Sample_Predictions', 'Response_Corrected_Targets'])
        sheet = next((s for s in preferred if s in book.sheet_parts), next(iter(book.sheet_parts)))
    grid = book.grid(sheet)
    scores = []
    for r, values in list(sorted(grid.items()))[:30]:
        score = 5 * bool(pick(values, FORMULA_ALIASES)) + 2 * bool(pick(values, NAME_ALIASES)) + 2 * bool(pick(values, MASS_ALIASES))
        scores.append((score, -r))
    header_row = -max(scores)[1] if scores else 1
    headers, _ = book.scan(sheet, header_row)
    return TableConfig(str(Path(path).resolve()), sheet, header_row, pick(headers, NAME_ALIASES),
                       pick(headers, FORMULA_ALIASES), pick(headers, MASS_ALIASES),
                       adduct_col=pick(headers, ('Adduct', 'Ion', 'Primary_Adduct', 'Ion form', 'Adduct')),
                       image_col=pick(headers, IMAGE_ALIASES))


def load_rows(book, cfg):
    headers, raw = book.scan(cfg.sheet, cfg.header_row)
    if not headers:
        raise ValueError('No headers in %s, row %s' % (cfg.sheet, cfg.header_row))
    for label, col in [('name', cfg.name_col), ('formula', cfg.formula_col), ('mass', cfg.mass_col),
                       ('key', cfg.key_col), ('adduct', cfg.adduct_col), ('image', cfg.image_col), ('group', cfg.group_col), ('RAW', cfg.raw_col), ('RT', cfg.rt_col)]:
        if col and col not in headers:
            raise ValueError('Selected %s column is absent: %s' % (label, col))
    if not cfg.name_col:
        raise ValueError('Choose a name column for ' + cfg.sheet)
    if not cfg.formula_col:
        raise ValueError('Choose a molecular formula column for ' + cfg.sheet)
    rows = []
    for r, values in raw:
        # Skip notes/footers that have neither a selected identity nor a formula.
        if not str(values.get(cfg.name_col, '')).strip() and not str(values.get(cfg.formula_col, '')).strip():
            continue
        rows.append({'row': r, 'values': values, 'name': str(values.get(cfg.name_col, '')).strip(),
                     'formula': str(values.get(cfg.formula_col, '')).strip(),
                     'key': str(values.get(cfg.key_col, '')).strip() if cfg.key_col else ''})
    return headers, rows


def match_names(known, unknown, hc, hu, cc, cu, options):
    from .final_table_triplicates import match_table_names
    return match_table_names(known, unknown, hc, hu, cc, cu, options)


def safe_name(text):
    value = re.sub(r'\s+', '_', str(text or ''))
    value = re.sub(r'[^0-9A-Za-z_\-\.\(\)\[\]]+', '_', value).strip('_')
    return (value or 'item')[:120]


def image_key(path):
    stem = Path(path).stem
    m = re.fullmatch(r'\d+__(.+)__mz[-+]?\d+(?:\.\d+)?', stem, flags=re.I)
    if m:
        stem = m.group(1)
    else:
        stem = re.sub(r'^(?:XIC__|XIC_)', '', stem, flags=re.I)
        stem = re.sub(r'(?:__XIC|_XIC)$', '', stem, flags=re.I)
    return norm(stem)


def _groupkey(name):
    return norm(re.sub(r'__XIC_plots$|\.raw$', '', name, flags=re.I))


from .final_table_triplicates import Images


def thumbnail(path, width, height):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert('RGBA')
        ratio = min(width / max(im.width, 1), height / max(im.height, 1), 1.0)
        w, h = max(1, round(im.width * ratio)), max(1, round(im.height * ratio))
        if im.width > w*2 or im.height > h*2:
            resample = getattr(Image, 'Resampling', Image).LANCZOS
            im.thumbnail((w*2, h*2), resample)
        out = io.BytesIO()
        im.save(out, format='PNG')
        return out.getvalue(), w, h, Path(path).name


def mass_result(row, cfg, headers, limit):
    mode = _mode(cfg, headers)
    adduct = row['values'].get(cfg.adduct_col, '') if cfg.adduct_col else cfg.fixed_adduct
    result = {'mode': mode, 'adduct': str(adduct) if mode == 'observed_mz' else '', 'theory': None,
              'observed': row['values'].get(cfg.mass_col, '') if cfg.mass_col else '', 'ppm': None, 'status': '', 'formula': None}
    try:
        theory = theoretical_value(row['formula'], mode in ('observed_mz', 'raw_mz'), adduct)
        result['theory'] = theory
    except ValueError as exc:
        result['status'] = str(exc)
        return result
    if mode == 'raw_mz':
        result.update({'observed': None, 'status': 'RAW_NOT_READ_IN_PREVIEW; run to extract real MS1 centroid', 'adduct': str(adduct)})
        return result
    value = decimal_number(result['observed'])
    if value is None or value <= 0:
        result['status'] = 'NO_VALID_MASS_VALUE; theory_only' if cfg.mass_col else 'NO_MASS_COLUMN; theory_only'
        return result
    with localcontext() as ctx:
        ctx.prec = 32
        ppm = (value - theory) / theory * Decimal(1000000)
    result['ppm'] = ppm
    prefix = 'REFERENCE_ONLY_NOT_MEASUREMENT' if mode == 'reference_neutral' else 'OBSERVED_MZ' if mode == 'observed_mz' else 'OBSERVED_NEUTRAL'
    result['status'] = prefix + ('; MATCH_AT_INPUT_PRECISION' if ppm == 0 else '; OUTSIDE_PPM_LIMIT' if abs(ppm) > Decimal(str(limit)) else '; WITHIN_PPM_LIMIT')
    if mode == 'reference_neutral':
        result['status'] = prefix + ('; MATCH_AT_INPUT_PRECISION' if ppm == 0 else '; NUMERICAL_DIFFERENCE_ONLY')
    result['format'] = ppm_number_format(ppm)
    field = 'Theoretical_mz' if mode == 'observed_mz' else 'Theoretical_mass'
    source = colname(cfg.mass_col) + str(row['row'])
    result['formula'] = 'IFERROR((VALUE(%s)-{%s})/{%s}*1000000,"")' % (source, field, field)
    return result


def prepare_context(known_cfg, unknown_cfg, options, books=None, progress=None):
    """Identical full-table matching for preview and export. No thumbnails in preview."""
    from .final_table_triplicates import SourceCatalog, resolve_claims
    from .final_table_display import validate_display_mode
    validate_display_mode(options.xic_display_mode)
    from .final_table_captions import from_options
    captions = from_options(options)
    books = books or {str(Path(c.path).resolve()): Book(c.path) for c in (known_cfg, unknown_cfg)}
    context = {}
    for label, cfg in [('100', known_cfg), ('1000', unknown_cfg)]:
        book = books[str(Path(cfg.path).resolve())]
        headers, rows = load_rows(book, cfg)
        catalog = SourceCatalog(cfg.image_dir, options.recursive, cfg.metadata_path, label,
                                book, cfg, options.use_source_index)
        catalog.enrich(rows, headers, options)
        images = Images(cfg.image_dir, options.recursive, options.triplicate_xic, catalog)
        matches = resolve_claims({r['row']: images.find(r, cfg) for r in rows})
        context[label] = {'book': book, 'headers': headers, 'rows': rows, 'catalog': catalog,
                          'images': images, 'matches': matches, 'config': cfg}
        if progress:
            progress('%s table: %s image files, %s metadata sources; %s rows have usable images.' %
                     (label, len(images.files), len(catalog.sources), sum(bool(m['paths']) for m in matches.values())))
    context['links'] = match_names(context['100']['rows'], context['1000']['rows'],
                                  context['100']['headers'], context['1000']['headers'],
                                  known_cfg, unknown_cfg, options)
    return context


def _make_plans(book, cfg, options, label, names=None, progress=None, prepared=None):
    from .final_table_triplicates import field, rawkey
    from .final_table_display import render_xic
    from .final_table_captions import from_options
    captions = from_options(options)
    headers, rows = prepared['headers'], prepared['rows']
    matches = prepared['matches']
    cells, pictures, audit = {}, {}, []
    mode = _mode(cfg, headers)
    theory_key = 'Theoretical_mz' if mode in ('observed_mz', 'raw_mz') else 'Theoretical_mass'
    columns = [theory_key, 'Mass_difference_ppm', 'Mass_QC', 'XIC_plot', 'XIC_status']
    if mode == 'raw_mz':
        columns = ['Theoretical_mass', 'Observed_neutral_mass', 'Observed_adduct', 'Theoretical_mz',
                   'Observed_mz', 'Mass_difference_ppm', 'Observed_neutral_ppm', 'Mass_QC',
                   'Observed_mass_source', 'Observed_mass_scan', 'Observed_mass_RT_min', 'XIC_plot', 'XIC_status']
        for rep in '123':
            columns += ['Rep'+rep+'_Observed_mz', 'Rep'+rep+'_Mass_ppm', 'Rep'+rep+'_Mass_status']
    if names is not None:
        columns += ['Name_in_1000', 'Match_status']
    source_image_n, complete_n, partial_n, single_n = 0, 0, 0, 0
    for i, row in enumerate(rows):
        if progress and i % 50 == 0:
            progress('%s table: %s / %s rows (append results; concentration and integration unchanged)' % (label, i, len(rows)))
        rn = row['row']
        extra_cells = {}
        if mode == 'raw_mz':
            from .observed_mass import mass_cells
            m, extra_cells = mass_cells(prepared['observations'][rn])
        else:
            m = mass_result(row, cfg, headers, options.ppm_limit)
        image = matches[rn]
        match_status = image['status']
        used = []
        if image['paths']:
            try:
                pictures[rn] = render_xic(image, options.image_width, options.image_height,
                                          options.xic_display_mode, captions, compound_name=row['name'])
                used = image.get('used', [])
                if options.xic_display_mode == 'all' and image['is_triplicate']:
                    complete_n += int(len(used) == 3)
                    partial_n += int(len(used) < 3)
                else:
                    single_n += 1
                source_image_n += len(used)
            except Exception as exc:
                image['status'] = 'IMAGE_READ_FAILED: ' + type(exc).__name__ + ': ' + str(exc)
        vals = {theory_key: {'value': m['theory'], 'format': '0.000000'},
                'Mass_difference_ppm': {'value': m['ppm'], 'formula': m.get('formula'), 'format': m.get('format', PPM_FIXED)},
                'Mass_QC': {'value': m['status']},
                'XIC_plot': {'value': '' if rn in pictures else ('Unmatched' if image['status'] == 'NOT_FOUND' else 'Not embedded; see XIC_status')},
                'XIC_status': {'value': image['status']}}
        vals.update(extra_cells)
        link = names.get(rn, {}) if names is not None else {}
        if names is not None:
            vals['Name_in_1000'] = {'value': link.get('name', '')}
            vals['Match_status'] = {'value': link.get('status', 'NOT_MATCHED')}
        cells[rn] = vals
        event = {'Table': label, 'Input_file': Path(cfg.path).name, 'Sheet': cfg.sheet, 'Excel_row': rn,
                 'Original_name': row['name'], 'Formula': row['formula'], 'Mass_mode': mode,
                 'Mass_source_column': 'RAW_MS1_CENTROID (not Exact_mass)' if mode == 'raw_mz' else headers.get(cfg.mass_col, ''), 'Source_value': m['observed'],
                 'Adduct': m['adduct'], 'Theoretical_full_precision': str(m['theory']) if m['theory'] else '',
                 'PPM_full_precision': str(m['ppm']) if m['ppm'] is not None else '', 'Mass_QC': m['status'],
                 'Image_status': image['status'], 'Image_file': '|'.join(str(x['path']) for x in used),
                 'Source_image_count': len(used), 'Image_match_method': image['method'],
                 'XIC_display_mode': options.xic_display_mode,
                 'XIC_caption_fields': '|'.join(captions.field_names()),
                 'XIC_caption_custom': captions.custom_text,
                 'XIC_caption_records': json.dumps(image.get('caption_records', []), ensure_ascii=False),
                 'Match_status_before_display': match_status,
                 'Matched_image_count': len(image['selected']),
                 'Matched_image_files': '|'.join(str(x['path']) for x in image['selected']),
                 'Displayed_replicates': '|'.join(str(x.get('rep', '')) for x in used),
                 'Display_selection_note': image.get('display_note', ''),
                 'Image_candidates': image.get('candidates', ''), 'Image_metadata_rejected': '|'.join(image.get('metadata_rejected', [])), 'Name_in_1000': link.get('name', ''),
                 'Name_match_status': link.get('status', ''), 'Name_match_method': link.get('method', ''),
                 'Name_candidates': link.get('candidates', ''), 'Source_metadata_status': row.get('_source_status', ''),
                 'Source_original_names': '|'.join(row.get('_source_names', [])), 'Source_metadata_trace': row.get('_source_trace', '')}
        for rep in '123':
            entries = [x for x in used if x['rep'] == rep]
            matched_entries = [x for x in image['selected'] if x['rep'] == rep]
            info = row.get('_source_info', {})
            found = field(info, 'Rep' + rep + '_Found')
            for src in row.get('_source_records', []):
                si = src['_source_info']
                if matched_entries and rawkey(field(si, 'RAW')) == rawkey(matched_entries[0]['raw']):
                    found = found or field(si, 'Found')
            event['Rep' + rep + '_Found_from_source'] = found
            event['Rep' + rep + '_IS_QC_from_source'] = field(info, 'Rep' + rep + '_IS_QC')
            event['Rep' + rep + '_RAW'] = '|'.join(x['raw'] for x in entries)
            event['Rep' + rep + '_image'] = '|'.join(str(x['path']) for x in entries)
            event['Rep' + rep + '_matched_RAW'] = '|'.join(x['raw'] for x in matched_entries)
            event['Rep' + rep + '_matched_image'] = '|'.join(str(x['path']) for x in matched_entries)
            event['Rep' + rep + '_conflicts'] = '|'.join(str(x['path']) for x in image['conflicts'].get(rep, []))
        for key in ('Observed_mz', 'Observed_neutral_mass', 'Observed_adduct', 'Observed_mass_source', 'Observed_mass_scan', 'Observed_mass_RT_min'):
            event[key] = extra_cells.get(key, {}).get('value', '')
        event['Image_read_errors'] = json.dumps(image.get('read_errors', []), ensure_ascii=False)
        audit.append(event)
    summary = {'rows': len(rows), 'theory_calculated': sum(bool(x[theory_key]['value']) for x in cells.values()),
               'ppm_calculated': sum(x['Mass_difference_ppm']['value'] is not None for x in cells.values()),
               'observed_mz_count': sum(bool(x.get('Observed_mz', {}).get('value')) for x in cells.values()),
               'mass_status_counts': dict(Counter(x['Mass_QC']['value'] for x in cells.values())),
               'mass_mode': mode, 'images_embedded': len(pictures), 'source_images_embedded': source_image_n,
               'three_replicate_rows': complete_n, 'partial_replicate_rows': partial_n,
               'single_image_rows': single_n, 'xic_display_mode': options.xic_display_mode,
               'xic_caption_fields': captions.field_names(), 'xic_caption_custom': captions.custom_text,
               'displayed_replicate_counts': dict(Counter(rep for x in audit for rep in x['Displayed_replicates'].split('|') if rep)),
               'image_files_found': len(prepared['images'].files), 'source_metadata_files': len(prepared['catalog'].sources),
               'image_status_counts': dict(Counter(x['Image_status'] for x in audit)),
               'name_status_counts': dict(Counter(x['Name_match_status'] for x in audit if x['Name_match_status'])),
               'name_matches': sum(x.get('status', '').startswith('MATCHED') for x in (names or {}).values()),
               'name_matching_applicable': names is not None}
    return columns, cells, pictures, audit, summary


def execute(known_cfg, unknown_cfg, output_parent, options=None, progress=None, *, reader_factory=None):
    options = options or Options()
    from .final_table_display import validate_display_mode, DISPLAY_LABELS
    validate_display_mode(options.xic_display_mode)
    from .final_table_captions import from_options
    captions = from_options(options)
    if not 120 <= options.image_width <= 1200 or not 40 <= options.image_height <= 500:
        raise ValueError('Image width must be 120..1200 px and height 40..500 px.')
    if not math.isfinite(options.ppm_limit) or options.ppm_limit <= 0:
        raise ValueError('PPM limit must be finite and positive.')
    if Path(known_cfg.path).resolve() == Path(unknown_cfg.path).resolve() and known_cfg.sheet == unknown_cfg.sheet:
        raise ValueError('The 100 and 1000 table cannot be the same worksheet.')
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / ('final_tables_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    folder.mkdir()
    books = {}
    audit, summaries, files_out = [], {}, []
    try:
        paths = list(dict.fromkeys(str(Path(c.path).resolve()) for c in (known_cfg, unknown_cfg)))
        inputs = {p: sha256(p) for p in paths}
        for cfg in (known_cfg, unknown_cfg):
            if cfg.metadata_path:
                inputs[str(Path(cfg.metadata_path).resolve())] = sha256(cfg.metadata_path)
        for p in paths:
            books[p] = Book(p)
        kb, ub = books[str(Path(known_cfg.path).resolve())], books[str(Path(unknown_cfg.path).resolve())]
        hk, kr = load_rows(kb, known_cfg)
        hu, ur = load_rows(ub, unknown_cfg)
        context = prepare_context(known_cfg, unknown_cfg, options, books, progress)
        links = context['links']
        for role in ('100', '1000'):
            view = context[role]
            if _mode(view['config'], view['headers']) == 'raw_mz':
                from .observed_mass import extract_table
                extract_table(view, options, progress=progress, reader_factory=reader_factory)
        for label, cfg, book, names in [('100', known_cfg, kb, links), ('1000', unknown_cfg, ub, None)]:
            cols, cells, images, events, summary = _make_plans(book, cfg, options, label, names, progress, context[label])
            book.append(cfg.sheet, cfg.header_row, cols, cells, images, options.image_width, options.image_height)
            audit.extend(events)
            summaries[label] = summary
        for p, book in books.items():
            records = [a for a in audit if a['Table'] == ('100' if book is kb else '1000')] if len(books) > 1 else audit
            # Full audit is also provided separately; ambiguous equal basenames are distinguished by Table.
            headers = list(audit[0]) if audit else ['Table']
            book.add_audit('Postprocess_Audit', headers, [[a.get(h, '') for h in headers] for a in records])
            mass_events = []
            for role in ('100', '1000'):
                if context[role]['book'] is book:
                    mass_events += [dict(Table=role, **e) for e in context[role].get('observed_source_audit', [])]
            if mass_events:
                all_keys = list(dict.fromkeys(k for e in mass_events for k in e))
                book.add_audit('Observed_Mass_Audit', all_keys, [[e.get(k, '') for k in all_keys] for e in mass_events])
            notes = [
                ['Version', VERSION], ['Operation', 'Appends only; original row order, values and existing parts preserved.'],
                ['Theory display', '6 decimals; stored value not rounded before ppm calculation.'],
                ['PPM formula', '(source value - theory) / theory * 1000000; sign is retained.'],
                ['RAW observation', 'raw_mz mode: nearest full-MS1 native centroid at the existing XIC apex. Not a binned average, predicted mass or extraction center.'],
                ['Observed neutral mass', 'Back-calculated from Observed_mz and the assigned single adduct, not directly measured neutral mass.'],
                ['Observation search', 'Search +/- %s ppm; QC +/- %s ppm; apex RT halfwindow %s min. Search window is NOT measured mass error.' % (options.observed_search_ppm, options.ppm_limit, options.observed_rt_halfwindow)],
                ['Replicate mass', 'Rep1 first, then Rep2/3 only if no usable Rep1 measurement. Independent of image display; all per-replicate values and failed attempts retained.'],
                ['Mass evidence limits', 'Peak- and mass-window-conditioned observations, not independent instrument validation or unique structure confirmation. Missing and conflicting results are blank, never theoretical substitutes.'],
                ['Zero display', '0.000000. Every calculable ppm is displayed numerically; positive/negative signs are retained. Tiny nonzero values use scientific notation. Missing/invalid values remain blank with Mass_QC.'],
                ['Exact_mass default', 'Reference comparison only. Original enumerator Exact_mass is theoretical; not an observed mass error.'],
                ['Mass model', 'core.chemistry.MONO_MASS; ionic mass = atomic sum - charge * electron mass, divided by abs(charge).'],
                ['Formula edit', 'Theory is a computed snapshot. Rerun postprocessing after editing formula/adduct.'],
                ['Names', 'Full Combo; ordered reagent labels/formulas ignoring local indices; exact SMILES/name; unique ordered ABC-formula key flagged for review. No row-position/mass-only matching.'],
                ['Formula-only option', 'Optional unique formula match is unconfirmed identity; isomers cannot be proved by formula.'],
                ['XIC display mode', options.xic_display_mode + ' / ' + DISPLAY_LABELS[options.xic_display_mode]],
                ['XIC added caption', captions.description()],
                ['XIC custom text', captions.custom_text],
                ['Caption scope', 'Presentation only. Empty by default: no new names/labels above successful images. Original image titles/axes/legends are not removed. Missing/conflicting slots stay visibly marked. Full provenance remains in the audit/alternative description; this is not anonymization.'],
                ['Images', 'first: one readable unambiguous image, Rep1 then Rep2 then Rep3. all: three original-image panels, NOT an averaged trace. Same-replicate conflicts are never guessed.'],
                ['Image provenance', 'Rep*_image records displayed images; Rep*_matched_image retains accepted but undisplayed sources. First-image selection does not change concentrations or QC.'],
                ['Rows', 'All original rows remain. No training, peak integration or concentration modification.'],
                ['Sources', 'https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl'],
                ['OOXML', 'https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/structure-of-a-spreadsheetml-document']]
            book.add_audit('Postprocess_Readme', ['Item', 'Value'], notes)
            prefix = 'combined' if len(books) == 1 else ('100' if book is kb else '1000')
            filename = prefix + '__' + Path(p).stem + '_postprocessed' + Path(p).suffix.lower()
            out = folder / filename
            book.save(out)
            files_out.append(str(out))
        for p, before in inputs.items():
            if sha256(p) != before:
                raise RuntimeError('Input changed during processing: ' + p)
        with (folder / 'postprocess_audit.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(audit[0]) if audit else ['Table'])
            w.writeheader(); w.writerows(audit)
        import json as _json
        for suffix, key in [('sources', 'observed_source_audit'), ('scans', 'observed_scan_audit')]:
            events = [dict(Table=role, **e) for role in ('100','1000') for e in context[role].get(key, [])]
            if events:
                fields = list(dict.fromkeys(k for e in events for k in e))
                with (folder / ('observed_mass_' + suffix + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
                    writer.writerows({k: _json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in e.items()} for e in events)
        inventory = []
        source_messages = []
        for label in ('100', '1000'):
            view = context[label]
            for p in view['images'].files:
                x = view['images']._context(p)
                inventory.append({'Table': label, 'File': str(p), 'RAW': x['raw'], 'Replicate': x['rep'],
                                  'Group': x['group'], 'Directory': str(p.parent)})
            source_messages.extend([label + ': ' + x for x in view['catalog'].messages])
        with (folder / 'xic_file_index.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['Table', 'File', 'RAW', 'Replicate', 'Group', 'Directory'])
            w.writeheader(); w.writerows(inventory)
        diagnostics = ['CombiTrace-MS final-table diagnostics (observed m/z from RAW; no reintegration)',
                       'ppm: zero is displayed as 0.000000; very small nonzero values use scientific notation. Missing or invalid values remain blank with Mass_QC notes.',
                       'XIC layout: ' + DISPLAY_LABELS[options.xic_display_mode],
                       'Additional captions: ' + captions.description(),
                       'Only additional captions are changed; original plot content and source audit are retained.']
        for label, st in summaries.items():
            diagnostics += ['\n' + label + ' table', 'Images scanned: %s; rows with images: %s; source images used: %s; single-image rows: %s; complete triplicates: %s; incomplete triplicates: %s' %
                            (st['image_files_found'], st['images_embedded'], st['source_images_embedded'], st['single_image_rows'], st['three_replicate_rows'], st['partial_replicate_rows']),
                            'Observed m/z: %s rows; mass statuses: %s' % (st['observed_mz_count'], json.dumps(st['mass_status_counts'], ensure_ascii=False)),
                            'Displayed replicates: ' + json.dumps(st['displayed_replicate_counts'], ensure_ascii=False),
                            'XIC statuses: ' + json.dumps(st['image_status_counts'], ensure_ascii=False),
                            'Name-match statuses: ' + (json.dumps(st['name_status_counts'], ensure_ascii=False) if st['name_matching_applicable'] else 'Not applicable (target table has no reverse name mapping)')]
        diagnostics += ['\nNO_FOLDER or zero files: check the image directory and recursive search. NOT_FOUND: check names and retain the original enumeration/quantification CSVs or supply the original summary.',
                        'AMBIGUOUS_IMAGES: multiple source groups or unclear replicate identity. SOURCE_GROUP_MISMATCH: table group/RAW names differ from the directory. Multiple candidates within one replicate are not resolved by taking the first.',
                        'In single-image mode, undisplayed replicates are not necessarily missing or invalid. All matched sources remain in the audit.',
                        'Single-image mode uses the first readable, unambiguous source in Rep 1-2-3 order, not the strongest or best-QC result. Three images do not establish three valid quantitative replicates.',
                        'Ordered A/B/C formula-key matches require review; they do not confirm isomers. Product formula alone cannot establish unique identity.'] + source_messages
        (folder / 'postprocess_diagnostics.txt').write_text('\n'.join(diagnostics), encoding='utf-8-sig')
        if progress:
            progress('\n'.join(diagnostics[1:]))
        result = {'version': VERSION, 'status': 'COMPLETED', 'tables': summaries, 'outputs': files_out,
                  'input_hashes': inputs, 'config_100': asdict(known_cfg), 'config_1000': asdict(unknown_cfg), 'options': asdict(options),
                  'privacy': 'Local data audit may contain names/paths/formulas. Do not share without review.'}
        (folder / 'postprocess_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        (folder / 'COMPLETED.txt').write_text('Completed. Input files were not modified.\n' + '\n'.join(files_out), encoding='utf-8')
        if progress:
            progress('Complete:\n' + '\n'.join(files_out))
        return folder, result
    except Exception as exc:
        (folder / 'FAILED.json').write_text(json.dumps({'status': 'FAILED', 'error': type(exc).__name__ + ': ' + str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
