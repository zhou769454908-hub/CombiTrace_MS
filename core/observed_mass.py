"""Read measured MS1 centroid m/z at existing XIC peaks; never reconstruct from theory.

RAW files are opened read-only by fisher_py/RawFileReader. Original quantitative
areas, chromatographic assignments, RAW files and input spreadsheets are untouched.
The spectrum used for the primary value is selected by RT, not by mass error.
"""
from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
import math
import re
from collections import Counter, defaultdict, OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
from .final_table_triplicates import hn, rawkey, split_raw

METHOD = 'nearest_full_MS1_centroid_at_existing_XIC_apex'
RAW_MODE = 'raw_mz'


class RawBackendUnavailable(RuntimeError):
    pass


def get(info, *keys):
    for key in keys:
        value = info.get(hn(key))
        if value is not None and str(value).strip() not in ('', 'None', 'nan', 'NaN'):
            return str(value).strip()
    return ''


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def flag(value):
    text = str(value).strip().lower()
    if text in ('true', 'yes', '1', '1.0', 'found', 'Yes', '是'): return True
    if text in ('false', 'no', '0', '0.0', 'not found', 'Not found', 'No', '未找到', '否'): return False
    return None


def canonical_adduct(text):
    from .final_table_postprocess import theoretical_value
    a = re.sub(r'\s+', '', str(text)).replace('−', '-').replace('⁺', '+').replace('⁻', '-')
    aliases = {'M-H': '[M-H]-', 'M+H': '[M+H]+', 'M+Na': '[M+Na]+',
               'M+HCOO': '[M+HCOO]-', '[M+FA-H]-': '[M+HCOO]-',
               '[M+CHO2]-': '[M+HCOO]-', '[M+HCO2]-': '[M+HCOO]-'}
    a = aliases.get(a, a)
    if not re.fullmatch(r'\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])', a):
        raise ValueError('UNSUPPORTED_SINGLE_ION: ' + a)
    return a


def neutral_from_mz(formula, adduct, observed):
    """Back-calculated neutral mass, not a directly measured neutral molecule."""
    from .final_table_postprocess import theoretical_value
    a = canonical_adduct(adduct)
    m = re.fullmatch(r'\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])', a)
    mult, z = int(m.group(1) or 1), int(m.group(3) or 1)
    neutral = theoretical_value(formula)
    theo_mz = theoretical_value(formula, True, a)
    return neutral + (Decimal(str(observed)) - theo_mz) * z / mult


def event_matches(event, adduct):
    """Only full MS1 scans of the requested polarity; no MS2/SIM substitution."""
    text = str(event).lower()
    if not re.search(r'\bfull\s+ms(?:1)?\b', text):
        return False
    sign = '-' if adduct.endswith('-') else '+'
    return bool(re.search(r'(?:^|\s)' + re.escape(sign) + r'(?:\s|$)', text))


def pick_centroid(masses, intensities, theory, search_ppm, secondary_fraction=0.20):
    """Select the strongest measured centroid. Ambiguous competitive peaks reject.

    No nearest-to-theory tie-break, no averaging different same-scan peaks,
    and no replacement with the search center when no peak was observed.
    """
    mz, height = np.asarray(masses, dtype=float), np.asarray(intensities, dtype=float)
    if mz.ndim != 1 or height.ndim != 1 or mz.shape != height.shape:
        return {'status': 'INVALID_CENTROID_ARRAYS', 'mz': None, 'candidates': []}
    valid = np.isfinite(mz) & np.isfinite(height) & (mz > 0) & (height > 0)
    valid &= np.abs(mz - float(theory)) <= float(theory) * float(search_ppm) / 1e6
    ids = np.where(valid)[0]
    ordered = sorted(ids, key=lambda i: (-height[i], mz[i]))
    candidates = [{'mz': float(mz[i]), 'intensity': float(height[i])} for i in ordered]
    if not ordered:
        return {'status': 'NO_PEAK_IN_SEARCH_WINDOW', 'mz': None, 'candidates': []}
    if len(ordered) > 1 and height[ordered[1]] >= height[ordered[0]] * secondary_fraction:
        return {'status': 'AMBIGUOUS_CENTROIDS', 'mz': None, 'candidates': candidates}
    i = ordered[0]
    return {'status': 'MEASURED', 'mz': float(mz[i]), 'intensity': float(height[i]),
            'candidates': candidates}


class FisherSource:
    """Small explicit adapter using native centroid/label data, without m/z binning."""
    def __init__(self, path):
        try:
            from fisher_py import RawFile
        except Exception as exc:
            raise RawBackendUnavailable(
                'Could not load fisher_py/RawFileReader/.NET. Use the environment that already reads RAW files; do not reinstall the full scientific stack. Original error: ' + repr(exc)) from exc
        self.raw = RawFile(str(path))
        self.access = getattr(self.raw, '_raw_file_access', None)
        if self.access is None:
            self.close()
            raise RuntimeError('RAW_READER_INTERFACE_UNAVAILABLE: _raw_file_access')
        if bool(getattr(self.access, 'in_acquisition', False)):
            self.close()
            raise RuntimeError('RAW_STILL_IN_ACQUISITION')
        scans = getattr(self.raw, '_ms1_scan_numbers', None)
        rts = getattr(self.raw, '_ms1_retention_times', None)
        if scans is None or rts is None or len(scans) != len(rts):
            self.close()
            raise RuntimeError('RAW_READER_MS1_INDEX_UNAVAILABLE')
        self.index = sorted([(int(s), float(rt)) for s, rt in zip(scans, rts)
                             if math.isfinite(float(rt))], key=lambda x: (x[1], x[0]))
        self.events = {}
        self.spectra = OrderedDict()
        try:
            from importlib.metadata import version
            self.version = version('fisher-py')
        except Exception:
            self.version = 'unknown'

    def event(self, sn):
        if sn not in self.events:
            self.events[sn] = str(self.access.get_scan_event_string_for_scan_number(int(sn)))
        return self.events[sn]

    def spectrum(self, sn):
        if sn in self.spectra:
            self.spectra.move_to_end(sn)
            return self.spectra[sn]
        stream = self.access.get_centroid_stream(int(sn), False)
        mz = np.asarray(list(stream.masses), dtype=float) if stream is not None else np.array([])
        intensity = np.asarray(list(stream.intensities), dtype=float) if stream is not None else np.array([])
        if len(mz):
            result = (mz, intensity, 'RAW_NATIVE_CENTROID_STREAM')
        else:
            stats = self.access.get_scan_stats_for_scan_number(int(sn))
            if not bool(getattr(stats, 'is_centroid_scan', False)):
                raise ValueError('NO_CENTROID_STREAM_PROFILE_NOT_CONVERTED')
            segment = self.access.get_segmented_scan_from_scan_number(int(sn), stats)
            result = (np.asarray(list(segment.positions), dtype=float),
                      np.asarray(list(segment.intensities), dtype=float), 'RAW_CONFIRMED_CENTROID_SEGMENT')
        self.spectra[sn] = result
        while len(self.spectra) > 96:
            self.spectra.popitem(last=False)
        return result

    def close(self):
        access = getattr(self, 'access', None) or getattr(getattr(self, 'raw', None), '_raw_file_access', None)
        for method in ('dispose', 'Dispose', 'close'):
            f = getattr(access, method, None)
            if callable(f):
                f(); break
        if hasattr(self, 'spectra'): self.spectra.clear()


def _info(values, headers):
    return {hn(h): str(values.get(c, '')).strip() for c, h in headers.items()
            if values.get(c) is not None}


def _rep(raw, info, catalog):
    r = get(info, 'Replicate', 'Rep').lower().replace('rep', '').strip('-_ ')
    if r in ('1', '2', '3'): return r
    candidates = catalog.raw_map.get(rawkey(raw), set())
    reps = {x[1] for x in candidates}
    if len(reps) == 1: return next(iter(reps))
    return split_raw(raw)[1]


def row_requests(row, headers, cfg, catalog):
    """Resolve existing peak/RAW/ion provenance without borrowing another replicate RT."""
    from .final_table_postprocess import theoretical_value
    original = _info(row['values'], headers)
    adduct_text = row['values'].get(cfg.adduct_col, '') if cfg.adduct_col else cfg.fixed_adduct
    try:
        adduct = canonical_adduct(adduct_text)
        theory = theoretical_value(row['formula'], True, adduct)
        neutral = theoretical_value(row['formula'])
    except ValueError as exc:
        return [], {'status': str(exc), 'adduct': str(adduct_text), 'theory': None, 'neutral': None}
    meta = {'status': '', 'adduct': adduct, 'theory': theory, 'neutral': neutral}
    if any(x in row.get('_source_status', '') for x in ('CONFLICT', 'AMBIGUOUS')):
        meta['status'] = 'UNRESOLVED_SOURCE_IDENTITY: ' + row['_source_status']
        return [], meta
    inputs = [(dict(r['_source_info']), r.get('_source_trace', ''),
               30 if r.get('_source_trace', '').lower().endswith('__xic_quant.csv') else 20)
              for r in row.get('_source_records', [])]
    # Never use the merged _source_info RAW/Apex: it may be the first of 3 replicates.
    if cfg.raw_col:
        original[hn('RAW')] = str(row['values'].get(cfg.raw_col, ''))
    if cfg.rt_col:
        original[hn('Apex_RT')] = str(row['values'].get(cfg.rt_col, ''))
    inputs.append((original, 'selected_final_table', 40 if cfg.raw_col and cfg.rt_col else 10))
    normalized = []
    for info, trace, priority in inputs:
        if get(info, 'RAW', 'RAW_file', 'Raw_name'):
            normalized.append((info, trace, priority))
        for rep in ('1', '2', '3'):
            raw = get(info, 'Rep' + rep + '_RAW', 'raw_' + rep)
            if not raw: continue
            short = {hn('RAW'): raw, hn('Replicate'): rep}
            for key in ('Apex_RT', 'RT_start', 'RT_end', 'Found', 'Adduct', 'Matched_Adduct',
                        'MH_Found', 'MH_Apex_RT', 'Formate_Found', 'Formate_Apex_RT', 'Formate_Accepted'):
                val = get(info, 'Rep' + rep + '_' + key)
                if val: short[hn(key)] = val
            normalized.append((short, trace + '#Rep' + rep, priority))
    choices = defaultdict(list)
    for info, trace, priority in normalized:
        raw = get(info, 'RAW', 'RAW_file', 'Raw_name')
        rt = number(get(info, 'Apex_RT', 'Apex_RT_min', 'RT_min', 'RT'))
        lo, hi = number(get(info, 'RT_start')), number(get(info, 'RT_end'))
        source_found = flag(get(info, 'Found'))
        status = 'READY'
        # Ion-specific apex/Found take precedence over a summed area peak.
        prefix = 'MH' if adduct == '[M-H]-' else 'Formate' if adduct == '[M+HCOO]-' else ''
        ion_found = flag(get(info, prefix + '_Found')) if prefix else None
        ion_rt = number(get(info, prefix + '_Apex_RT')) if prefix else None
        source_ion = get(info, 'Matched_Adduct', 'Adduct')
        try: source_ion = canonical_adduct(source_ion) if source_ion else ''
        except ValueError: source_ion = ''  # combined labels do not establish a single ion identity
        if ion_found is False or source_found is False:
            status = 'SOURCE_PEAK_NOT_FOUND'
        elif prefix == 'Formate' and flag(get(info, 'Formate_Accepted')) is False:
            status = 'SOURCE_CHANNEL_NOT_ACCEPTED'
        elif source_ion and source_ion != adduct and ion_found is not True:
            status = 'SOURCE_ION_MISMATCH'
        if ion_rt is not None: rt = ion_rt
        if rt is None or rt < 0:
            status = status if status != 'READY' else 'MISSING_PER_RAW_APEX_RT'
        elif lo is not None and hi is not None and (hi < lo or not lo <= rt <= hi):
            status = 'INCONSISTENT_PEAK_RT_BOUNDS'
        # A quant CSV with invalid Found/RT must not be overridden by a generic summary.
        q = {'raw_name': raw, 'rep': _rep(raw, info, catalog), 'apex_rt': rt, 'rt_start': lo,
             'rt_end': hi, 'adduct': adduct, 'theory': float(theory), 'formula': row['formula'],
             'status': status, 'trace': trace, 'priority': priority, 'found': get(info, 'Found')}
        choices[rawkey(raw)].append(q)
    requests = []
    for key, candidates in sorted(choices.items()):
        top = max(x['priority'] for x in candidates)
        best = [x for x in candidates if x['priority'] == top]
        # Blank bounds can be supplied by the same-level record only if no disagreement exists.
        conflicts = []
        for prop in ('apex_rt', 'rt_start', 'rt_end', 'status', 'rep'):
            values = {round(x[prop], 7) if isinstance(x[prop], float) else x[prop]
                      for x in best if x[prop] not in (None, '')}
            if len(values) > 1: conflicts.append(prop)
        q = dict(best[0])
        for prop in ('rt_start', 'rt_end'):
            q[prop] = next((x[prop] for x in best if x[prop] is not None), None)
        q['trace'] = '|'.join(sorted({x['trace'] for x in best}))
        if conflicts: q['status'] = 'CONFLICTING_PEAK_METADATA: ' + ','.join(conflicts)
        requests.append(q)
    reps = Counter(q['rep'] for q in requests)
    for q in requests:
        if reps[q['rep']] > 1:
            q['status'] = 'MULTIPLE_RAW_FOR_SAME_REPLICATE' if q['rep'] else 'MULTIPLE_UNLABELLED_RAW_SOURCES'
    if not requests: meta['status'] = 'NEED_ORIGINAL_XIC_CSV_OR_PER_RAW_RT_MAPPING'
    return requests, meta


def measure_request(source, request, options):
    out = dict(request)
    out.update({'observed_mz': None, 'scan': None, 'scan_rt': None, 'centroid_method': '',
                'scan_checks': [], 'support_mz_weighted': None, 'support_scan_count': 0,
                'support_spread_ppm': None, 'reader_version': getattr(source, 'version', 'unknown')})
    rt = float(request['apex_rt'])
    window = options.observed_rt_halfwindow
    lo, hi = rt - window, rt + window
    if request['rt_start'] is not None: lo = max(lo, float(request['rt_start']))
    if request['rt_end'] is not None: hi = min(hi, float(request['rt_end']))
    out['window_start'], out['window_end'] = lo, hi
    candidates = []
    if not hasattr(source, '_rt_lookup'):
        source._rt_lookup = [x[1] for x in source.index]
    i0, i1 = bisect_left(source._rt_lookup, lo), bisect_right(source._rt_lookup, hi)
    for sn, scan_rt in source.index[i0:i1]:
        if lo <= scan_rt <= hi:
            if event_matches(source.event(sn), request['adduct']):
                candidates.append((int(sn), float(scan_rt)))
    candidates.sort(key=lambda x: (abs(x[1] - rt), x[0]))
    if not candidates:
        out['status'] = 'NO_FULL_MS1_OF_REQUESTED_POLARITY_AT_APEX'
        return out
    selected = candidates[:3]  # fixed small RT-local support; no use of full-run averaged spectrum
    for sn, scan_rt in selected:
        try:
            masses, intensities, kind = source.spectrum(sn)
            hit = pick_centroid(masses, intensities, request['theory'], options.observed_search_ppm)
        except Exception as exc:
            hit, kind = {'status': 'CENTROID_READ_FAILED: ' + str(exc), 'mz': None, 'candidates': []}, ''
        hit.update({'scan': sn, 'rt': scan_rt, 'centroid_method': kind})
        out['scan_checks'].append(hit)
    apex = out['scan_checks'][0]
    out['scan'], out['scan_rt'], out['centroid_method'] = apex['scan'], apex['rt'], apex['centroid_method']
    if apex['mz'] is None:
        out['status'] = 'APEX_' + apex['status']
        return out  # no replacement by a more convenient neighbouring scan
    out['observed_mz'] = apex['mz']
    valid = [h for h in out['scan_checks'] if h['mz'] is not None]
    out['support_scan_count'] = len(valid)
    weights = [h['intensity'] for h in valid]
    out['support_mz_weighted'] = float(np.average([h['mz'] for h in valid], weights=weights))
    spread = (max(h['mz'] for h in valid) - min(h['mz'] for h in valid)) / request['theory'] * 1e6
    out['support_spread_ppm'] = spread
    ppm = (apex['mz'] - request['theory']) / request['theory'] * 1e6
    out['status'] = 'OBSERVED_RAW_MS1' + ('; OUTSIDE_PPM_LIMIT' if abs(ppm) > options.ppm_limit else '; WITHIN_PPM_LIMIT')
    if len(valid) < len(selected): out['status'] += '; INCOMPLETE_NEIGHBOUR_SUPPORT'
    if len(valid) == 1: out['status'] += '; SINGLE_SCAN_SUPPORT'
    if spread > options.ppm_limit: out['status'] += '; UNSTABLE_NEIGHBOUR_MASS'
    return out


def validate_options(cfg, options):
    if not cfg.raw_dir or not Path(cfg.raw_dir).expanduser().is_dir():
        raise ValueError('Observed-m/z mode requires the original Thermo RAW directory, not the XIC image directory.')
    for name, value, lo, hi in [('Mass search tolerance (ppm)', options.observed_search_ppm, 0.1, 200),
                              ('Apex RT half-window (min)', options.observed_rt_halfwindow, .001, 1)]:
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('%s must be in the range %s-%s.' % (name, lo, hi))
    if options.observed_search_ppm < options.ppm_limit:
        raise ValueError('The mass search window cannot be smaller than the ppm QC limit. Search tolerance and QC threshold are different parameters.')


def extract_table(prepared, options, progress=None, reader_factory=None):
    cfg = prepared['config']
    validate_options(cfg, options)
    factory = reader_factory or FisherSource
    root = Path(cfg.raw_dir).expanduser().resolve()
    paths = defaultdict(list)
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() == '.raw': paths[rawkey(p.name)].append(p.resolve())
    if not paths:
        raise ValueError('No .raw files were found in: ' + str(root))
    row_results, by_path = {}, defaultdict(list)
    scan_audit, source_audit = [], []
    for row in prepared['rows']:
        requests, meta = row_requests(row, prepared['headers'], cfg, prepared['catalog'])
        meta.update({'measurements': requests, 'observed': None, 'chosen': None})
        row_results[row['row']] = meta
        for q in requests:
            q['excel_row'], q['name'] = row['row'], row['name']
            q['observed_mz'] = None
            if q['status'] != 'READY': continue
            candidates = paths.get(rawkey(q['raw_name']), [])
            if not candidates:
                q['status'] = 'RAW_FILE_NOT_FOUND'; continue
            if len(candidates) > 1:
                q['status'] = 'AMBIGUOUS_RAW_FILENAME'; continue
            q['raw_path'] = str(candidates[0])
            by_path[candidates[0]].append(q)
    for i, (path, requests) in enumerate(sorted(by_path.items(), key=lambda x: str(x[0]))):
        if progress: progress('Observed RAW m/z: %s/%s %s; %s existing chromatographic peaks' % (i+1, len(by_path), path.name, len(requests)))
        before = path.stat()
        source = None
        try:
            source = factory(path)
            for q in sorted(requests, key=lambda x: (x['apex_rt'], x['excel_row'])):
                try: q.update(measure_request(source, q, options))
                except Exception as exc: q['status'] = 'RAW_MEASUREMENT_FAILED: ' + repr(exc)
        except RawBackendUnavailable:
            raise
        except Exception as exc:
            for q in requests: q['status'] = 'RAW_OPEN_FAILED: ' + repr(exc)
        finally:
            if source is not None: source.close()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('RAW changed during read-only processing: ' + str(path))
        for q in requests:
            q['raw_size'], q['raw_mtime_ns'] = after.st_size, after.st_mtime_ns
    # Do not let two records claim the same scan centroid from one RAW/ion.
    claims = defaultdict(list)
    for meta in row_results.values():
        for q in meta['measurements']:
            if q.get('observed_mz') is not None:
                claims[(q['raw_path'], q['scan'], q['observed_mz'])].append(q)
    for key, group in claims.items():
        if len({q['excel_row'] for q in group}) > 1:
            for q in group:
                q['candidate_before_conflict'] = q['observed_mz']
                q['observed_mz'] = None
                q['status'] = 'CENTROID_CLAIMED_BY_MULTIPLE_ROWS'
    for rn, meta in row_results.items():
        available = [q for q in meta['measurements'] if q.get('observed_mz') is not None]
        available.sort(key=lambda q: (int(q['rep']) if q['rep'] in ('1','2','3') else 9, rawkey(q['raw_name'])))
        if available:
            meta['chosen'] = available[0]
            meta['observed'] = Decimal(str(available[0]['observed_mz']))
            meta['status'] = available[0]['status']
            if available[0]['rep'] not in ('', '1'): meta['status'] += '; MASS_REPLICATE_FALLBACK'
        elif not meta['status']:
            meta['status'] = '; '.join(sorted({q['status'] for q in meta['measurements']})) or 'NO_OBSERVATION'
        for q in meta['measurements']:
            source_audit.append({k: v for k,v in q.items() if k != 'scan_checks'})
            for h in q.get('scan_checks', []):
                scan_audit.append({'Excel_row': rn, 'Name': q['name'], 'RAW': q.get('raw_path', q['raw_name']),
                                  'Replicate': q['rep'], 'Adduct': q['adduct'], 'Theoretical_mz': q['theory'],
                                  'Search_ppm': options.observed_search_ppm, **h})
    prepared['observations'] = row_results
    prepared['observed_source_audit'], prepared['observed_scan_audit'] = source_audit, scan_audit
    return row_results


def mass_cells(meta):
    """New measured columns with formula-backed ppm; full precision in cached values."""
    from .final_table_postprocess import ppm_number_format
    m = meta.get('chosen') or {}
    observed, theory = meta['observed'], meta['theory']
    neutral = meta['neutral']
    back = neutral_from_mz(m['formula'], meta['adduct'], observed) if observed is not None else None
    ppm = (observed - theory) / theory * 1000000 if observed is not None else None
    nppm = (back - neutral) / neutral * 1000000 if back is not None else None
    cells = {
        'Theoretical_mass': {'value': neutral, 'format': '0.000000'},
        'Observed_mz': {'value': observed, 'format': '0.000000'},
        'Observed_neutral_mass': {'value': back, 'format': '0.000000'},
        'Observed_adduct': {'value': meta['adduct']},
        'Observed_neutral_ppm': {'value': nppm, 'format': ppm_number_format(nppm),
                                'formula': 'IFERROR(({Observed_neutral_mass}-{Theoretical_mass})/{Theoretical_mass}*1000000,"")' if back is not None else None},
        'Observed_mass_source': {'value': ('Rep' + m['rep'] + ' | ' if m.get('rep') else '') + m.get('raw_name', '')},
        'Observed_mass_scan': {'value': m.get('scan')},
        'Observed_mass_RT_min': {'value': m.get('scan_rt'), 'format': '0.000000'},
    }
    for rep in ('1','2','3'):
        items = [x for x in meta['measurements'] if x['rep'] == rep]
        q = items[0] if len(items) == 1 else {}
        mz = q.get('observed_mz')
        p = (Decimal(str(mz)) - theory) / theory * 1000000 if mz is not None else None
        cells['Rep'+rep+'_Observed_mz'] = {'value': mz, 'format': '0.000000'}
        cells['Rep'+rep+'_Mass_ppm'] = {'value': p, 'format': ppm_number_format(p),
            'formula': 'IFERROR(({Rep'+rep+'_Observed_mz}-{Theoretical_mz})/{Theoretical_mz}*1000000,"")' if mz is not None else None}
        cells['Rep'+rep+'_Mass_status'] = {'value': q.get('status', 'NO_SOURCE_RECORD')}
    result = {'mode': RAW_MODE, 'adduct': meta['adduct'], 'theory': theory, 'observed': observed,
              'ppm': ppm, 'status': meta['status'], 'format': ppm_number_format(ppm),
              'formula': 'IFERROR(({Observed_mz}-{Theoretical_mz})/{Theoretical_mz}*1000000,"")' if observed is not None else None}
    return result, cells
