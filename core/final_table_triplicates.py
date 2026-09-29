"""Postprocess identity/three-replicate image support. Does not read RAW or fit models.

Original exporter contracts:
  <Sheet>/<RAW stem>__XIC_plots/<No:03>__<safe(Name)>__mz....png
  Name=<Sheet>_<No:04>; Name is shared across replicates.
  *__enumerated.csv and *__XIC_quant.csv provide portable provenance.
Do not assign pictures by row number, nearest mass, or Name1 substring matching.
"""
from __future__ import annotations
from .legacy_schema import canonical_header, canonical_sheet
import csv
import io
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def norm(x):
    return unicodedata.normalize('NFKC', str(x or '')).strip().casefold()


def hn(x):
    x = canonical_header(x)
    return re.sub(r'[\s_\-./()（）:]+', '', norm(x))


def field(info, *keys):
    for k in keys:
        v = info.get(hn(k), '')
        if str(v or '').strip():
            return str(v).strip()
    return ''


def basename(x):
    return re.split(r'[\\/]', str(x or '').strip())[-1]


@lru_cache(maxsize=16384)
def rawkey(x):
    x = re.sub(r'__XIC_plots$|\.raw$', '', basename(x), flags=re.I)
    x = norm(x).translate(str.maketrans({'–': '-', '—': '-', '−': '-', '＿': '_'}))
    return re.sub(r'[-\s_]+', '-', x).strip('-')


def split_raw(x):
    r = rawkey(x)
    m = re.fullmatch(r'(.+?)-(?:rep(?:licate)?-?|r)?([123])', r)
    return (m.group(1), m.group(2)) if m else (r, '')


@lru_cache(maxsize=16384)
def fk(x):
    from .final_table_postprocess import formula_key
    return formula_key(x)


def exact_combo(x):
    return re.sub(r'\s+', '', str(x or '')).casefold()


@lru_cache(maxsize=16384)
def parts_from_combo(x):
    """Keep roles and complete reagent labels, ignore LOCAL # index only."""
    result = {}
    for s in re.split(r'\s*[|｜]\s*', str(x or '')):
        m = re.fullmatch(r'([ABC])(?:#\d+)?\s*:\s*(.+)', s.strip(), re.I)
        if not m or m.group(1).upper() in result:
            continue
        rest = m.group(2)
        label, formula = rest.rsplit(':', 1) if ':' in rest else ('', rest)
        formula_sig = fk(formula)
        if formula_sig:
            result[m.group(1).upper()] = (norm(label), formula_sig)
    return result if set(result) == {'A', 'B', 'C'} else {}


def abc_formula_token(x):
    if not norm(x).startswith('abc_formula|'):
        return None
    result = {}
    for t in str(x).split('|')[1:]:
        m = re.fullmatch(r'\s*([abcABC])=(.+)', t)
        if not m:
            return None
        counts = []
        for bit in m.group(2).split(','):
            c = re.fullmatch(r'([A-Z][a-z]?)=(\d+)', bit.strip())
            if not c or int(c.group(2)) <= 0:
                return None
            counts.append((c.group(1), int(c.group(2))))
        result[m.group(1).upper()] = tuple(sorted(counts))
    return tuple(result[r] for r in 'ABC') if set(result) == set('ABC') else None


def info_for(row, headers=None):
    info = {hn(h): str(row.get('values', {}).get(c, '') or '').strip() for c, h in (headers or {}).items()}
    info.update({k: v for k, v in row.get('_source_info', {}).items() if not info.get(k)})
    info.setdefault('name', row.get('name', ''))
    info.setdefault('formula', row.get('formula', ''))
    return info


def identity_keys(row, headers=None):
    info = info_for(row, headers)
    combo = field(info, 'Combo', 'Combination', 'Combo', 'Combo_key')
    parts = parts_from_combo(combo)
    if not parts:
        temp = {}
        for role in 'ABC':
            form = field(info, role + '_Formula', 'Formula_' + role)
            lab = field(info, role + '_ID', role + '_Name', role + '_Label')
            if fk(form):
                temp[role] = (norm(lab), fk(form))
        if len(temp) == 3:
            parts = temp
    labeled = None
    if parts and all(parts[r][0] and not re.search(r'\[r\d+\]', parts[r][0], re.I) for r in 'ABC'):
        labeled = tuple(parts[r] for r in 'ABC')
    formulas = tuple(parts[r][1] for r in 'ABC') if parts else None
    if formulas is None:
        for key in ('ABC_Formula_Key', 'Product_Master_Formula_Key'):
            formulas = abc_formula_token(field(info, key))
            if formulas is not None:
                break
    names = [row.get('name', '')] + row.get('_source_names', [])
    for key in ('Name', 'Compound', 'Original_Name', 'Target_Name'):
        n = field(info, key)
        if n:
            names.append(n)
    return {'combo': exact_combo(combo), 'component_labels': labeled,
            'smiles': field(info, 'Product_SMILES', 'SMILES'),
            'name': tuple(dict.fromkeys(norm(n) for n in names if norm(n))),
            'abc_formula': formulas}


def _indices(rows, headers=None):
    idx = {k: defaultdict(list) for k in ('combo', 'component_labels', 'smiles', 'name', 'abc_formula')}
    for r in rows:
        for kind, key in identity_keys(r, headers).items():
            for item in (key if kind == 'name' else (key,)):
                if item:
                    idx[kind][item].append(r)
    return idx


def _candidates(row, headers, indices, allow_abc=True):
    keys = identity_keys(row, headers)
    for kind in ('combo', 'component_labels', 'smiles', 'name', 'abc_formula'):
        if kind == 'abc_formula' and not allow_abc:
            continue
        key = keys[kind]
        candidates = []
        for item in (key if kind == 'name' else (key,)):
            if item:
                candidates.extend(indices[kind].get(item, []))
        if candidates:
            seen = set()
            return [x for x in candidates if not (id(x) in seen or seen.add(id(x)))], kind
    return [], ''


def match_table_names(known, unknown, hk, hu, ck, cu, options):
    if bool(ck.key_col) != bool(cu.key_col):
        raise ValueError('Select matching-key columns for BOTH tables, or leave BOTH automatic.')
    idx = _indices(unknown, hu)
    keyidx, formidx = defaultdict(list), defaultdict(list)
    for u in unknown:
        if u.get('key'):
            keyidx[norm(u['key'])].append(u)
        if fk(u['formula']):
            formidx[fk(u['formula'])].append(u)
    out = {}
    for k in known:
        if ck.key_col:
            matches, method = keyidx.get(norm(k.get('key')), []) if k.get('key') else [], 'explicit_key'
            if ck.key_col == ck.formula_col and cu.key_col == cu.formula_col:
                matches, method = formidx.get(fk(k['formula']), []), 'unique_formula_unconfirmed_identity'
        else:
            matches, method = _candidates(k, hk, idx, getattr(options, 'allow_abc_formula', True))
            if method == 'component_labels':
                method = 'ABC_reagent_labels_and_formulas_ignore_local_indices'
            elif method == 'abc_formula':
                method = 'ABC_ordered_component_formulas_review_isomers'
            if not matches and options.allow_unique_formula:
                matches, method = formidx.get(fk(k['formula']), []), 'unique_formula_unconfirmed_identity'
        result = {'name': '', 'status': 'NOT_MATCHED', 'method': method, 'candidates': ''}
        if matches:
            result['candidates'] = '|'.join(u['name'] for u in matches)
            if fk(k['formula']):
                matches = [u for u in matches if fk(u['formula']) == fk(k['formula'])]
                if not matches:
                    result['status'] = 'FORMULA_CONFLICT'
            if len(matches) == 1 and matches[0]['name']:
                review = method in ('ABC_ordered_component_formulas_review_isomers', 'unique_formula_unconfirmed_identity')
                result.update(name=matches[0]['name'], unknown_row=matches[0]['row'],
                              status=('MATCHED_REVIEW_ABC_FORMULAS' if method.startswith('ABC_ordered') else
                                      'MATCHED_REVIEW_FORMULA_ONLY') if review else 'MATCHED', candidates='')
            elif matches:
                result['status'] = 'AMBIGUOUS'
        out[k['row']] = result
    return out


class SourceCatalog:
    """Read exporter metadata from the SELECTED image tree and optional source table.
    No RAW, trace intensity matrices, PDF, arbitrary parent trees or network access.
    """
    def __init__(self, folder='', recursive=True, metadata_path='', role='', book=None, cfg=None, enabled=True):
        self.records = []
        self.messages = []
        self.sources = []
        self.raw_map = defaultdict(set)
        self._strong_raw = set()
        if not enabled:
            self.indices = _indices([])
            return
        root = Path(folder).resolve() if folder else None
        if root and root.is_dir():
            iterator = root.rglob('*') if recursive else root.glob('*')
            for p in sorted(iterator):
                if p.is_file() and re.search('__enumerated\\.csv$|__XIC_quant\\.csv$|all_targets|全部targets|三平行平均|triplicate[ _]average', p.name, re.I):
                    self._csv(p)
        if book and cfg:
            for name in book.sheet_parts:
                if name == cfg.sheet:
                    continue
                if canonical_sheet(name) in ('Triplicate Average', 'Triplicate XIC Summary', 'XIC Long Table', 'ESI_Features', '三平行平均结果', 'triplicate average', '三平行XIC汇总', 'triplicate xic summary', 'XIC长表', 'xic long table'):
                    self._sheet(book, name, role)
        if metadata_path:
            p = Path(metadata_path).expanduser().resolve()
            if not p.is_file():
                raise ValueError('Source mapping table not found: ' + str(p))
            if p.suffix.lower() == '.csv':
                self._csv(p)
            else:
                from .final_table_ooxml import Book
                b = Book(p)
                for name in b.sheet_parts:
                    if not name.startswith(('Postprocess_', 'Model_', 'Descriptor_', 'Feature_')):
                        self._sheet(b, name, role)
        self.indices = _indices(self.records)

    def _csv(self, path):
        if path.stat().st_size > 50 * 1024 * 1024:
            self.messages.append('Skipped metadata CSV larger than 50MB: ' + str(path)); return
        rows = None
        for encoding in ('utf-8-sig', 'gb18030'):
            try:
                with path.open(encoding=encoding, newline='') as f:
                    rows = list(csv.DictReader(f))
                break
            except UnicodeError:
                pass
        if rows is None:
            self.messages.append('Cannot decode metadata CSV: ' + str(path)); return
        if len(rows) > 200000:
            self.messages.append('Skipped metadata CSV larger than 200000 rows: ' + str(path)); return
        self.sources.append(str(path))
        raw = path.name[:-len('__XIC_quant.csv')] if path.name.lower().endswith('__xic_quant.csv') else ''
        for r in rows:
            info = {hn(k): v for k, v in r.items() if k is not None}
            if raw and not field(info, 'RAW'):
                info['raw'] = raw
            self._add(info, str(path))

    def _sheet(self, book, name, role):
        grid = book.grid(name)
        for rn, row in sorted(grid.items())[:30]:
            headers = {c: str(h) for c, h in row.items()}
            hs = {hn(h) for h in headers.values()}
            if not hs.intersection({'name', 'compound', 'Compound_Name', 'Name'}) or not hs.intersection({'formula', 'productformula', 'Formula'}):
                continue
            self.sources.append(str(book.path) + '#' + name)
            for rownum, vals in sorted(grid.items()):
                if rownum <= rn:
                    continue
                info = {hn(h): vals.get(c, '') for c, h in headers.items()}
                ds = norm(field(info, 'Dataset', 'Set', 'Role'))
                if (role == '100' and ds in ('target', 'unknown', '1000')) or (role == '1000' and ds in ('training', 'known', '100')):
                    continue
                self._add(info, str(book.path) + '#' + name + ':' + str(rownum))
            return

    def _add(self, info, source):
        name = field(info, 'Name', 'Compound', 'Name', 'Compound_Name')
        formula = field(info, 'Formula', 'Product_Formula', 'Formula')
        if not name or not fk(formula):
            return
        row = {'name': name, 'formula': formula, 'values': {}, '_source_info': info, '_source_trace': source}
        self.records.append(row)
        sheet = field(info, 'Sheet', 'Source_Sheet', 'Injection_Group')
        for r in ('1', '2', '3'):
            raw = field(info, 'raw_' + r, 'Rep' + r + '_RAW')
            if raw:
                self._map_raw(raw, sheet, r)
        raw = field(info, 'RAW')
        rep = field(info, 'Replicate').lstrip('-_ ')
        if raw:
            base, r = split_raw(raw)
            rep = rep if rep in ('1', '2', '3') else r
            if rep:
                self._map_raw(raw, sheet, rep)

    def _map_raw(self, raw, sheet, rep):
        key = rawkey(raw)
        if sheet and key not in self._strong_raw:
            self.raw_map[key].clear()
            self._strong_raw.add(key)
        if sheet or key not in self._strong_raw:
            self.raw_map[key].add((rawkey(sheet) if sheet else split_raw(raw)[0], rep))

    def enrich(self, rows, headers, options):
        for row in rows:
            info = info_for(row, headers)
            row['_source_info'] = info
            candidates, method = _candidates(row, headers, self.indices, getattr(options, 'allow_abc_formula', True))
            if not candidates:
                row['_source_status'] = 'NO_METADATA_MATCH' if self.records else 'NO_METADATA_AVAILABLE'
                continue
            good = [r for r in candidates if fk(r['formula']) == fk(row['formula'])]
            if not good:
                row['_source_status'] = 'METADATA_FORMULA_CONFLICT'
                continue
            group = field(info, 'Sheet', 'Source_Sheet', 'Injection_Group')
            if group:
                scoped = [r for r in good if not field(r['_source_info'], 'Sheet', 'Source_Sheet') or
                          rawkey(field(r['_source_info'], 'Sheet', 'Source_Sheet')) == rawkey(group)]
                if scoped:
                    good = scoped
            # Repeated records of the same name/Combo across Rep1-3 are ONE identity.
            names = {norm(r['name']) for r in good}
            combos = {identity_keys(r)['component_labels'] or identity_keys(r)['combo'] for r in good}
            combos.discard(''); combos.discard(None)
            sheets = {rawkey(field(r['_source_info'], 'Sheet', 'Source_Sheet')) for r in good if field(r['_source_info'], 'Sheet', 'Source_Sheet')}
            if len(names) != 1 or len(combos) > 1 or len(sheets) > 1:
                row['_source_status'] = 'AMBIGUOUS_METADATA_IDENTITY'
                continue
            # A precise row group may reduce metadata from repeated source exports.
            group = field(info, 'Sheet', 'Source_Sheet', 'Injection_Group')
            if sheets and group and rawkey(group) not in sheets and split_raw(group)[0] not in sheets:
                row['_source_status'] = 'METADATA_GROUP_CONFLICT'; continue
            for r in good:
                for key, value in r['_source_info'].items():
                    if not info.get(key) and str(value or '').strip():
                        info[key] = str(value).strip()
            row['_source_names'] = sorted({r['name'] for r in good})
            row['_source_records'] = good
            row['_source_status'] = 'METADATA_MATCHED_' + method
            row['_source_trace'] = '|'.join(sorted({r['_source_trace'] for r in good}))


def safe_name(x):
    from .final_table_postprocess import safe_name as f
    return f(x)


def image_identity(path):
    stem = Path(path).stem
    m = re.fullmatch(r'(\d+)__(.+)__mz[-+]?\d+(?:\.\d+)?', stem, re.I)
    number = ''
    if m:
        number, stem = m.groups()
    else:
        stem = re.sub(r'^(?:XIC__|XIC_)', '', stem, flags=re.I)
        stem = re.sub(r'(?:__XIC|_XIC)$', '', stem, flags=re.I)
    return norm(stem), number


class Images:
    def __init__(self, folder, recursive=True, triplicates=True, catalog=None):
        self.folder = Path(folder).resolve() if folder else None
        self.triplicates = triplicates
        self.catalog = catalog
        self.index = defaultdict(list)
        self.files = []
        self.files_by_name = defaultdict(list)
        self._context_cache = {}
        if not self.folder:
            return
        if not self.folder.is_dir():
            raise ValueError('XIC folder not found: ' + str(self.folder))
        for p in sorted(self.folder.rglob('*') if recursive else self.folder.glob('*')):
            if not p.is_file() or p.suffix.lower() not in EXTENSIONS or re.search(r'__MS2(?:\.|$)', p.name, re.I):
                continue
            self.files.append(p)
            self.files_by_name[norm(p.name)].append(p)
            key, num = image_identity(p)
            self.index[key].append(p)
            # Explicitly labelled file replicate, never remove a plain compound -1 suffix.
            m = re.fullmatch(r'(.+?)[_-]+(?:rep|replicate|r)[_-]?([123])', key, re.I)
            if m:
                self.index[m.group(1)].append(p)

    def _context(self, p):
        if p not in self._context_cache:
            self._context_cache[p] = self._context_uncached(p)
        return self._context_cache[p]

    def _context_uncached(self, p):
        parents = list(p.parents)
        directory = next((x for x in parents if x.name.lower().endswith('__xic_plots')), None)
        if directory:
            raw = re.sub(r'__XIC_plots$', '', directory.name, flags=re.I)
            mapped = self.catalog.raw_map.get(rawkey(raw), set()) if self.catalog else set()
            # CSV sources agree on a sheet and explicit replicate label.
            if len(mapped) == 1:
                base, rep = next(iter(mapped))
            else:
                base, rep = split_raw(raw)
            return {'family': (str(directory.parent), base), 'group': base, 'rep': rep, 'raw': raw,
                    'folder': directory, 'path': p}
        # Explicit Rep1 subfolders or file suffixes; generic run1/run2 are NOT replicates.
        for par in parents:
            m = re.fullmatch(r'(?:rep(?:licate)?|r)[_-]?([123])', par.name, re.I)
            if m:
                return {'family': (str(par.parent), ''), 'group': '', 'rep': m.group(1), 'raw': par.name,
                        'folder': par, 'path': p}
            if par == self.folder:
                break
        m = re.fullmatch(r'(.+?)[_-]+(?:rep(?:licate)?|r)[_-]?([123])', image_identity(p)[0], re.I)
        if m:
            return {'family': (str(p.parent), m.group(1)), 'group': '', 'rep': m.group(2), 'raw': 'Rep' + m.group(2), 'folder': p.parent, 'path': p}
        return {'family': (str(p.parent), ''), 'group': '', 'rep': '', 'raw': '', 'folder': p.parent, 'path': p}

    def _path_matches(self, value, prefer_root=False):
        p = Path(str(value))
        if p.is_absolute() and p.is_file() and not (prefer_root and self.folder):
            return [p.resolve()]
        if self.folder:
            rp = self.folder / str(value).replace('\\', '/')
            inside = False
            try:
                rp.resolve().relative_to(self.folder)
                inside = True
            except ValueError:
                pass
            if rp.is_file() and (not prefer_root or inside):
                return [rp.resolve()]
        # Moved outputs: retain old RAW-folder and file-name correspondence, not path root.
        base = norm(basename(value))
        candidates = self.files_by_name.get(base, [])
        old = re.split(r'[\\/]', str(value))
        oldfolder = next((x for x in reversed(old[:-1]) if x.lower().endswith('__xic_plots')), '')
        if oldfolder:
            candidates = [p for p in candidates if any(rawkey(x.name) == rawkey(oldfolder) for x in p.parents)]
        return candidates

    def find(self, row, cfg):
        info = row.get('_source_info', {})
        explicit = str(row['values'].get(cfg.image_col, '')).strip() if cfg.image_col else ''
        names = [row['name']] + row.get('_source_names', [])
        method = 'explicit_image_column' if explicit else 'exact_name_or_exporter_name'
        if explicit:
            candidates = self._path_matches(explicit)
        else:
            candidates = []
            for name in names:
                candidates += self.index.get(norm(name), []) + self.index.get(norm(safe_name(name)), [])
            # Original XIC_PNG links are more specific than a sanitized name.
            # They can disambiguate truncated/CJK names without choosing by row order.
            source_paths = []
            for record in row.get('_source_records', []):
                v = field(record['_source_info'], 'XIC_PNG')
                if v:
                    source_paths += self._path_matches(v, prefer_root=True)
            if source_paths:
                candidates = source_paths
                method = 'source_XIC_PNG'
        candidates = sorted(set(candidates))
        context = [self._context(p) for p in candidates]
        group = str(row['values'].get(cfg.group_col, '')).strip() if cfg.group_col else field(info, 'Sheet', 'Source_Sheet', 'Injection_Group')
        explicit_raws = set()
        for r in '123':
            val = field(info, 'Rep' + r + '_RAW', 'raw_' + r)
            if val:
                explicit_raws.add(rawkey(val))
        before_group = list(context)
        if explicit_raws:
            context = [x for x in context if rawkey(x['raw']) in explicit_raws]
            method += '+RAW_columns'
        elif group:
            g = rawkey(group)
            context = [x for x in context if x['group'] == g or rawkey(x['raw']) == g or
                       any(rawkey(p.name) == g for p in x['path'].parents) or
                       (x['group'] and x['group'].endswith('-' + g))]
            method += '+source_group'
        if before_group and not context:
            return self._result([], 'SOURCE_GROUP_MISMATCH', method, before_group)
        # Validate formula from matched original XIC csv when present, never mass proximity.
        metadata_rejected = []
        if self.catalog:
            byname = self.catalog.indices['name']
            checked = []
            for x in context:
                key = image_identity(x['path'])[0]
                records = [r for n in names for r in byname.get(norm(n), [])
                           if norm(safe_name(r['name'])) == key or norm(r['name']) == key]
                scoped = []
                for r in records:
                    ri = r['_source_info']
                    rr = field(ri, 'RAW')
                    rrs = {rawkey(field(ri, 'raw_' + j, 'Rep' + j + '_RAW')) for j in '123'} - {''}
                    if rr and rawkey(rr) != rawkey(x['raw']):
                        continue
                    if rrs and rawkey(x['raw']) not in rrs:
                        continue
                    scoped.append(r)
                forms = {fk(r['formula']) for r in scoped}
                if forms and (len(forms) != 1 or fk(row['formula']) not in forms):
                    metadata_rejected.append(str(x['path']))
                    continue
                checked.append(x)
            if context and not checked:
                return self._result([], 'IMAGE_METADATA_FORMULA_CONFLICT', method, context)
            context = checked
        if not context:
            return self._result([], 'NOT_FOUND' if self.folder or explicit else 'NO_FOLDER', method, [])
        if not self.triplicates:
            if len(context) == 1:
                return self._result(context, 'MATCHED', method, context)
            return self._result([], 'AMBIGUOUS_IMAGES', method, context)
        if len(context) == 1 and not context[0]['rep']:
            return self._result(context, 'MATCHED', method, context)
        families = {x['family'] for x in context}
        if len(families) != 1 or any(not x['rep'] for x in context):
            return self._result([], 'AMBIGUOUS_IMAGES', method, context)
        slots = defaultdict(list)
        for x in context:
            slots[x['rep']].append(x)
        selected, conflicts = [], {}
        from .final_table_postprocess import sha256
        for rep, entries in sorted(slots.items()):
            # Exact duplicated file bytes within one replicate are safe copies, not new runs.
            if len(entries) == 1 or len({sha256(e['path']) for e in entries}) == 1:
                selected.append(entries[0])
            else:
                conflicts[rep] = entries
        status = 'MATCHED_TRIPLICATES' if len(selected) == 3 else 'MATCHED_PARTIAL_REPLICATES' if selected else 'AMBIGUOUS_REPLICATE_IMAGES'
        result = self._result(selected, status, method + '+replicate_group', context)
        result['conflicts'] = conflicts
        result['is_triplicate'] = True
        result['metadata_rejected'] = metadata_rejected
        return result

    @staticmethod
    def _result(selected, status, method, candidates):
        paths = [x['path'] for x in selected]
        return {'path': paths[0] if paths else None, 'paths': paths, 'selected': selected, 'status': status,
                'method': method, 'candidates': '|'.join(str(x['path']) for x in candidates),
                'conflicts': {}, 'is_triplicate': False}


def resolve_claims(matches):
    """Block reuse across different rows; retain unaffected replicate panels."""
    claims = defaultdict(list)
    for rn, result in matches.items():
        for path in result.get('paths', []):
            claims[str(path)].append(rn)
    for path, rownums in claims.items():
        if len(set(rownums)) > 1:
            for rn in rownums:
                r = matches[rn]
                blocked = [x for x in r['selected'] if str(x['path']) == path]
                for x in blocked:
                    r['conflicts'].setdefault(x['rep'] or 'single', []).append(x)
                r['selected'] = [x for x in r['selected'] if str(x['path']) != path]
                r['paths'] = [x['path'] for x in r['selected']]
                r['path'] = r['paths'][0] if r['paths'] else None
                r['status'] = 'PARTIAL_IMAGE_CLAIM_CONFLICT' if r['paths'] else 'IMAGE_CLAIMED_BY_MULTIPLE_ROWS'
    return matches


def compose_triplicates(result, width, height, captions=None, *, compound_name=''):
    """Three fixed slots of original pixels, optional captions; never an average."""
    from PIL import Image, ImageDraw
    from .final_table_captions import CaptionSettings, draw_caption, paste_original, draw_placeholder
    captions = (captions or CaptionSettings()).validate()
    height = max(360, min(500, height))
    W, H = width * 2, height * 2
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    byrep = {x['rep']: x for x in result['selected']}
    failed, used, caption_records = [], [], []
    for i, rep in enumerate('123'):
        top, bottom = i * H // 3, (i + 1) * H // 3
        entry = byrep.get(rep)
        label_entry = entry or {'rep': rep, 'raw': ''}
        area, detail = draw_caption(canvas, (0, top, W, bottom), label_entry, captions, compound_name)
        caption_records.append(detail)
        if entry:
            try:
                paste_original(canvas, entry, area)
                used.append(entry)
            except Exception as exc:
                failed.append((rep, str(entry['path']), type(exc).__name__ + ': ' + str(exc)))
                draw_placeholder(canvas, area, 'IMAGE READ FAILED (see audit)')
        else:
            text = 'CONFLICT - not embedded' if rep in result['conflicts'] else 'NOT AVAILABLE - no substitute'
            draw_placeholder(canvas, area, text)
        if i < 2:
            draw.line((5, bottom-1, W-5, bottom-1), fill='#B0B0B0', width=1)
    result['read_errors'] = failed
    result['used'] = used
    result['caption_records'] = caption_records
    if not used:
        raise ValueError('No readable replicate images')
    out = io.BytesIO(); canvas.save(out, format='PNG', optimize=True)
    label = 'Replicate XICs | ' + ' | '.join(str(x['path']) for x in used)
    return out.getvalue(), width, height, label
