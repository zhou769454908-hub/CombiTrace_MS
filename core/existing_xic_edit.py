"""Edit existing postprocessed XIC drawings, never recalculate measurements.

All data sources are existing images or rigorously delimited, tool-generated
image panels. No RAW reader, model fitting, integration or mass computation is
called. Workbook edits use the project's lossless OPC-part patching approach.
"""
from __future__ import annotations
from .legacy_schema import canonical_header, canonical_sheet

import copy
import csv
import hashlib
import io
import json
import math
import posixpath
import re
import tempfile
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import List
from lxml import etree as E
from PIL import Image, ImageOps

from .final_table_ooxml import (Book, S, R, P, C, D, A, N, tag, colindex,
                                 colname, relfile, resolve, _xml, _bytes, clean_text)
from .final_table_captions import CaptionSettings
from .final_table_display import render_xic

VERSION = '18.48.2'
MODE_LABELS = {'keep': 'Keep existing layout and displayed replicates',
               'first': 'Use first available image (prefer Rep 1)', 'all': 'Display all three replicates'}
LEGACY_VERSIONS = {'18.47.1', '18.47.2', '18.47.3', '18.48'}
AUDIT_BASE = 'XIC_Edit_Audit'


@dataclass
class EditOptions:
    mode: str = 'keep'
    captions: CaptionSettings = field(default_factory=CaptionSettings)
    keep_size: bool = True
    width: int = 520
    height: int = 220
    image_directory: str = ''  # optional relocation; never searches RAW
    latest_column_only: bool = True
    allow_embedded_recovery: bool = True

    def validate(self):
        if self.mode not in MODE_LABELS:
            raise ValueError('Invalid image layout.')
        self.captions.validate()
        if not isinstance(self.width, int) or not 120 <= self.width <= 1200:
            raise ValueError('Image width must be 120-1200 pixels.')
        if not isinstance(self.height, int) or not 40 <= self.height <= 500:
            raise ValueError('Image height must be 40-500 pixels.')
        if self.image_directory and not Path(self.image_directory).is_dir():
            raise ValueError('The optional XIC image directory does not exist.')
        return self


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def path_parts(value):
    return [x for x in str(value).replace('\\', '/').split('/') if x]


def path_name(value):
    p = path_parts(value)
    return p[-1] if p else ''


def splitpaths(text):
    return [s.strip() for s in str(text or '').split('|') if s.strip()]


def table_records(book, name):
    header, rows = book.scan(name, 1)
    return [{header[c]: v for c, v in values.items() if c in header} for _, values in rows]


def numbered_sheets(book, base):
    result = []
    for name in book.sheet_parts:
        match = re.fullmatch(re.escape(base) + r'(?:_(\d+))?', name)
        if match:
            result.append((int(match.group(1) or '1'), name))
    return [name for _, name in sorted(result)]


def _audit_index(book):
    original = []
    for name in numbered_sheets(book, 'Postprocess_Audit'):
        data = table_records(book, name)
        original.append((name, {(str(e.get('Sheet', '')), str(e.get('Excel_row', ''))): e for e in data}))
    versions = []
    for name in numbered_sheets(book, 'Postprocess_Readme'):
        pairs = {str(e.get('Item', '')): str(e.get('Value', '')) for e in table_records(book, name)}
        versions.append(pairs.get('Version', '').lstrip('v'))
    edits = []
    for name in numbered_sheets(book, AUDIT_BASE):
        edits.extend(table_records(book, name))
    return original, versions, edits


def _sheet_headers(book, sheet):
    grid = book.grid(sheet)
    hits = []
    for rn, row in grid.items():
        if rn > 1000:
            continue
        for col, val in row.items():
            match = re.fullmatch(r'XIC_plot(?:_PP(\d+))?', str(val).strip(), re.I)
            if match:
                hits.append({'row': rn, 'col': col, 'ordinal': int(match.group(1) or '1'),
                             'header': str(val), 'headers': row})
    # A data cell containing this literal is not enough: the header needs XIC_status.
    return grid, [h for h in hits if any(re.fullmatch(r'XIC_status(?:_PP\d+)?', str(v), re.I)
                                         for v in h['headers'].values())]


def _key_name(text):
    text = canonical_header(text)
    return re.sub(r'[\s_\-./()（）:]+', '', str(text).strip().casefold())


def _validate_identity(grid, header, rn, event):
    values = grid.get(rn, {})
    original = str(event.get('Original_name', '')).strip()
    if not original:
        raise ValueError('MISSING_AUDIT_IDENTITY: original name is missing from the audit.')
    names = [c for c, v in header.items() if _key_name(v) in
             {'name', 'compound', 'compoundname', 'samplename', 'Name', 'Compound_Name', 'Sample_Name', '名称', '化合物名称', 'compound_name', '样品名称', 'sample_name'}]
    if names:
        if not any(str(values.get(c, '')).strip() == original for c in names):
            raise ValueError('ROW_IDENTITY_CHANGED: the image anchor does not match the audited name. The image was retained; check whether only the cells were sorted.')
    elif original not in [str(v).strip() for v in values.values()]:
        raise ValueError('ROW_IDENTITY_UNVERIFIED: the row cannot be linked to its audited image source.')
    formula = str(event.get('Formula', '')).strip()
    formula_cols = [c for c, v in header.items() if _key_name(v) in
                    {'formula', 'molecularformula', 'productformula', 'Formula', 'Formula', '分子式', '化学式'}]
    if formula and formula_cols:
        # Identity check only; NEVER use mass proximity or recompute ppm.
        if not any(re.sub(r'\s+', '', str(values.get(c, ''))) == re.sub(r'\s+', '', formula)
                   for c in formula_cols):
            raise ValueError('ROW_FORMULA_CHANGED: the row formula differs from the image audit.')
    return original


def _json_list(value):
    if not value:
        return []
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError('Invalid XIC presentation provenance')
    return data


def _from_old_audit(book, event, version, media, width, height, description):
    """Accepted source paths plus exact tool-created panel bounds, not image OCR."""
    result = []
    displayed = [str(x) for x in splitpaths(event.get('Displayed_replicates', ''))]
    mode = str(event.get('XIC_display_mode', ''))
    three_panels = description.startswith('Replicate XICs |') or str(event.get('Image_status', '')).startswith('EMBEDDED_REPLICATES_')
    triple = three_panels
    if not mode:
        mode = 'all' if triple else 'first'
    used_paths = splitpaths(event.get('Image_file', ''))
    for rep in '123':
        matched = splitpaths(event.get('Rep' + rep + '_matched_image', ''))
        used = splitpaths(event.get('Rep' + rep + '_image', ''))
        paths = matched or used
        if len(paths) > 1:
            continue  # Never relax same-replicate conflicts.
        if paths:
            raw = event.get('Rep' + rep + '_matched_RAW') or event.get('Rep' + rep + '_RAW', '')
            result.append({'rep': rep, 'raw': str(raw), 'original_path': paths[0],
                           'was_displayed': bool(used) or rep in displayed or paths[0] in used_paths})
    if not result:
        matched = splitpaths(event.get('Matched_image_files', '')) or used_paths
        if len(matched) != 1:
            raise ValueError('NO_UNIQUE_SOURCE: no unambiguous image or replicate source is available.')
        result.append({'rep': '', 'raw': '', 'original_path': matched[0], 'was_displayed': bool(used_paths)})
    triple = triple or any(x['rep'] in ('1', '2', '3') for x in result)
    if not any(x['was_displayed'] for x in result):
        raise ValueError('NO_DISPLAYED_SOURCE: the audit does not identify the embedded image.')
    image_bytes = book.part(media)
    if not image_bytes:
        raise ValueError('EMBEDDED_IMAGE_MISSING')
    with Image.open(io.BytesIO(image_bytes)) as image:
        W, H = image.size
    # Only exact generated PNGs, not an arbitrary user-edited image.
    generated = posixpath.basename(media) == 'pp_' + digest(image_bytes) + '.png'
    records = {str(x.get('rep', '')): x for x in _json_list(event.get('XIC_caption_records', ''))}
    for item in result:
        if not item['was_displayed'] or not generated:
            continue
        rep = item['rep']
        bounds = None
        if three_panels:
            if rep not in '123' or not rep:
                continue
            idx = int(rep) - 1
            top, bottom = idx * H // 3, (idx + 1) * H // 3
            if version in ('18.48.1', '18.48.2') and rep in records:
                h = int(records[rep]['header_pixels'])
                bounds = [10, top + max(6, h), W - 10, bottom - 8]
            elif version in LEGACY_VERSIONS:
                bounds = [10, top + 30, W - 10, bottom - 8]
        elif version in ('18.48.1', '18.48.2'):
            if rep or any(x.get('full_caption') for x in records.values()):
                if rep in records:
                    h = int(records[rep]['header_pixels'])
                    bounds = [10, max(6, h), W - 10, H - 8]
            else:  # plain thumbnail: no generated label was added
                bounds = [0, 0, W, H]
        elif version in LEGACY_VERSIONS:
            bounds = [10, 30, W - 10, H - 8] if mode == 'first' and rep else [0, 0, W, H]
        if bounds and (not (triple or rep or any(x.get('full_caption') for x in records.values()))
                       or (W == width * 2 and H == height * 2)):
            if 0 <= bounds[0] < bounds[2] <= W and 0 <= bounds[1] < bounds[3] <= H:
                item['embedded'] = {'part': media, 'sha256': digest(image_bytes), 'bounds': bounds}
    return result, mode, triple


class SourceResolver:
    def __init__(self, book, options):
        self.book, self.options = book, options
        self.index = None
        self.notes = []
        self._pixel_cache = {}

    def _relocate(self, original):
        root = Path(self.options.image_directory) if self.options.image_directory else None
        if not root:
            return None
        if self.index is None:
            index = {}
            for p in root.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
                    index.setdefault(p.name.casefold(), []).append(p)
            self.index = index
        candidates = self.index.get(path_name(original).casefold(), [])
        target = [x.casefold() for x in path_parts(original)]
        # Match the full RAW__XIC_plots folder plus basename, not Name1/Name10.
        if len(target) >= 2:
            candidates = [p for p in candidates if [x.casefold() for x in p.parts[-2:]] == target[-2:]]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            self.notes.append('AMBIGUOUS_RELOCATION: ' + original)
        return None

    def get(self, item):
        """Return PNG bytes and origin without changing any original file."""
        # Prefer pixels already embedded in this result: a later overwritten PNG
        # must not silently replace the measured trace when only editing labels.
        existing = self._recover_embedded(item)
        if existing:
            return existing
        original = item.get('original_path', '')
        paths = []
        if self.options.image_directory:
            p = self._relocate(original)
            if p:
                paths.append(p)
        elif original:
            p = Path(original)
            # Never access Windows UNC shares implicitly.
            if not str(original).startswith(('\\\\', '//')) and p.is_file():
                paths.append(p)
            if not p.is_absolute():
                relative = self.book.path.parent / p
                if relative.is_file() and relative not in paths:
                    paths.append(relative)
        for p in paths:
            try:
                before = p.stat()
                with Image.open(p) as raw:
                    im = ImageOps.exif_transpose(raw).convert('RGBA')
                    out = io.BytesIO(); im.save(out, format='PNG')
                after = p.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Image changed while reading')
                return out.getvalue(), 'ORIGINAL_IMAGE', str(p)
            except Exception as exc:
                self.notes.append('ORIGINAL_READ_FAILED: %s: %s' % (p, exc))
        return self._recover_embedded(item)

    def _recover_embedded(self, item):
        source = item.get('embedded', {})
        if self.options.allow_embedded_recovery and source:
            b = self.book.part(source.get('part', ''))
            if not b or digest(b) != source.get('sha256'):
                self.notes.append('EMBEDDED_SOURCE_CHANGED_OR_MISSING: ' + source.get('part', ''))
                return None
            cache_key = (source.get('sha256'), tuple(source.get('bounds', [])))
            if cache_key in self._pixel_cache:
                return self._pixel_cache[cache_key], 'EMBEDDED_PANEL_RECOVERY', source['part']
            with Image.open(io.BytesIO(b)) as raw:
                box = list(map(int, source.get('bounds', [])))
                if len(box) != 4 or not (0 <= box[0] < box[2] <= raw.width and 0 <= box[1] < box[3] <= raw.height):
                    return None
                # Crop only the known generated panel area, not graph/title pixels.
                im = raw.crop(tuple(box)).convert('RGBA')
                out = io.BytesIO(); im.save(out, format='PNG')
                png = out.getvalue()
                if len(self._pixel_cache) < 128:
                    self._pixel_cache[cache_key] = png
                return png, 'EMBEDDED_PANEL_RECOVERY', source['part']
        return None


def _media_and_position(book, drawing_part, anchor, relationships):
    pic = anchor.find('{%s}pic' % D)
    if pic is None:
        return None
    nv = pic.find('{%s}nvPicPr/{%s}cNvPr' % (D, D))
    if nv is None or not re.fullmatch(r'Postprocess XIC \d+', nv.get('name', '')):
        return None
    mark = anchor.find('{%s}from' % D)
    if mark is None:
        return None
    col = int(mark.findtext('{%s}col' % D)) + 1
    row = int(mark.findtext('{%s}row' % D)) + 1
    blip = pic.find('.//{%s}blip' % A)
    if blip is None:
        return None
    rid = blip.get('{%s}embed' % R)
    rel = next((r for r in relationships if r.get('Id') == rid), None)
    if rel is None or rel.get('TargetMode') == 'External':
        return None
    media = resolve(drawing_part, rel.get('Target'))
    ext = anchor.find('{%s}ext' % D)
    if ext is None:
        ext = pic.find('.//{%s}xfrm/{%s}ext' % (A, A))
    if ext is None:
        return None
    width = round(int(ext.get('cx')) / 9525)
    height = round(int(ext.get('cy')) / 9525)
    return dict(pic=pic, nv=nv, blip=blip, media=media, col=col, row=row,
                width=width, height=height, id=nv.get('id'), original_row=nv.get('name').rsplit(' ', 1)[-1])


def _set_col_width(root, column, pixels):
    cols = root.find(tag('cols'))
    if cols is None:
        cols = E.Element(tag('cols')); root.insert(list(root).index(root.find(tag('sheetData'))), cols)
    new = []
    for node in list(cols):
        lo, hi = int(node.get('min')), int(node.get('max'))
        if lo <= column <= hi:
            if lo < column:
                left = copy.deepcopy(node); left.set('max', str(column - 1)); new.append(left)
            if hi > column:
                right = copy.deepcopy(node); right.set('min', str(column + 1)); new.append(right)
        else:
            new.append(node)
        cols.remove(node)
    new.append(E.Element(tag('col'), min=str(column), max=str(column), width=str((pixels + 12) / 7), customWidth='1'))
    cols.extend(sorted(new, key=lambda x: int(x.get('min'))))


def _protected(book, root):
    if root.find(tag('sheetProtection')) is not None:
        return True
    protection = book.book.find(tag('workbookProtection'))
    return protection is not None and protection.get('lockStructure') in ('1', 'true')


def _update_status(book, sheetroot, header, column, rownum, text):
    label = str(header[column])
    status_header = label.replace('XIC_plot', 'XIC_status')
    cols = [c for c, h in header.items() if h == status_header]
    if len(cols) != 1:
        return None
    ref = colname(cols[0]) + str(rownum)
    cell = sheetroot.find('.//s:sheetData/s:row/s:c[@r="%s"]' % ref, N)
    if cell is None or cell.find(tag('f')) is not None:
        return None  # don't overwrite formulas, even in a presentation-status column
    for child in list(cell):
        cell.remove(child)
    cell.set('t', 'inlineStr')
    E.SubElement(E.SubElement(cell, tag('is')), tag('t')).text = clean_text(text)
    return ref


def edit_book(book, options, progress=None, preview=False):
    """Returns audit rows. All failures retain their original picture untouched."""
    options.validate()
    original, versions, history = _audit_index(book)
    resolver = SourceResolver(book, options)
    events = []
    with tempfile.TemporaryDirectory(prefix='thermo_xic_edit_') as tmp:
        for sheet, part in list(book.sheet_parts.items()):
            grid, headers = _sheet_headers(book, sheet)
            if not headers:
                continue
            if options.latest_column_only:
                headers = [max(headers, key=lambda h: h['col'])]
            root = book.sheet(sheet)
            node = root.find(tag('drawing'))
            if node is None:
                continue
            rp = book.part(relfile(part))
            if not rp:
                continue
            sheetrels = _xml(rp)
            rel = next((r for r in sheetrels if r.get('Id') == node.get('{%s}id' % R)), None)
            if rel is None or rel.get('TargetMode') == 'External':
                continue
            dp = resolve(part, rel.get('Target'))
            drawing = _xml(book.part(dp))
            drel = _xml(book.part(relfile(dp)))
            pictures = []
            for anchor in drawing:
                info = _media_and_position(book, dp, anchor, drel)
                if info and info['col'] in {h['col'] for h in headers}:
                    pictures.append((anchor, info))
            counts = Counter((p['row'], p['col']) for _, p in pictures)
            changed = False
            for index, (anchor, info) in enumerate(pictures):
                rn, col = info['row'], info['col']
                ev = {'Workbook': book.path.name, 'Sheet': sheet, 'Excel_row': rn,
                      'XIC_column': colname(col), 'Drawing_part': dp, 'Picture_id': info['id'],
                      'Original_name': '', 'Formula': '', 'Status': 'UNCHANGED', 'Reason': '',
                      'Old_media': info['media'], 'Output_media': info['media'],
                      'Output_sha256': digest(book.part(info['media']) or b''),
                      'Mode': '', 'Is_triplicate': '', 'Displayed_replicates': '',
                      'Source_kinds': '', 'Source_details': '', 'Sources_json': '',
                      'Caption_fields': '|'.join(options.captions.field_names()),
                      'Caption_custom': options.captions.custom_text,
                      'Caption_records': '', 'Width_px': info['width'], 'Height_px': info['height']}
                events.append(ev)
                if progress and index % 50 == 0:
                    progress('%s / %s: %s / %s existing XIC images (no RAW access)' % (book.path.name, sheet, index, len(pictures)))
                committing = False
                try:
                    if _protected(book, root):
                        raise ValueError('PROTECTED_SHEET_OR_WORKBOOK: protection is not bypassed.')
                    if anchor.tag != '{%s}oneCellAnchor' % D:
                        raise ValueError('UNSUPPORTED_ANCHOR: the image layout was changed. The original image was retained; supported images use oneCellAnchor.')
                    if info['pic'].find('.//{%s}srcRect' % A) is not None:
                        raise ValueError('USER_CROPPED_PICTURE: manually cropped images are retained unchanged.')
                    transform = info['pic'].find('.//{%s}xfrm' % A)
                    if transform is not None and any(transform.get(a) not in (None, '0', 'false') for a in ('rot', 'flipH', 'flipV')):
                        raise ValueError('TRANSFORMED_PICTURE: rotated or flipped images are retained unchanged.')
                    if counts[(rn, col)] != 1:
                        raise ValueError('MULTIPLE_PICTURES_IN_CELL: multiple pictures occupy this cell; no replacement was guessed.')
                    h = next(h for h in headers if h['col'] == col)
                    found = [e for e in history if e.get('Sheet') == sheet and e.get('Drawing_part') == dp
                             and str(e.get('Picture_id')) == info['id']
                             and e.get('Output_sha256') == ev['Output_sha256'] and e.get('Status') == 'UPDATED']
                    if found:
                        entry = found[-1]
                        name = _validate_identity(grid, h['headers'], rn, entry)
                        sources = _json_list(entry.get('Sources_json', ''))
                        oldmode = str(entry['Mode'])
                        triple = str(entry.get('Is_triplicate', '')).lower() in ('1', 'true')
                        used_reps = splitpaths(entry.get('Displayed_replicates', ''))
                        for s in sources:
                            s['was_displayed'] = s.get('rep', '') in used_reps or (not s.get('rep') and not used_reps)
                    else:
                        ordinal = h['ordinal']
                        if ordinal > len(original):
                            raise ValueError('NO_AUDIT_FOR_XIC_COLUMN: source audit not found for this image column.')
                        entry = original[ordinal - 1][1].get((sheet, info['original_row']))
                        if entry is None:
                            raise ValueError('NO_ROW_AUDIT: original row-level image audit not found.')
                        name = _validate_identity(grid, h['headers'], rn, entry)
                        ver = versions[ordinal - 1] if len(versions) >= ordinal else ''
                        sources, oldmode, triple = _from_old_audit(book, entry, ver, info['media'],
                                                                    info['width'], info['height'], info['nv'].get('descr', ''))
                    ev['Original_name'], ev['Formula'] = name, str(entry.get('Formula', ''))
                    mode = oldmode if options.mode == 'keep' else options.mode
                    mode = mode if mode in ('first', 'all') else 'first'
                    available, kinds, details, retained = [], [], [], []
                    for i, item in enumerate(sources):
                        content = resolver.get(item)
                        descriptor = copy.deepcopy(item)
                        if content:
                            png, kind, location = content
                            temp_path = Path(tmp) / (digest(png) + '.png')
                            if not temp_path.exists():
                                temp_path.write_bytes(png)
                            use = dict(path=temp_path, rep=str(item.get('rep', '')), raw=item.get('raw', ''))
                            if options.mode != 'keep' or item.get('was_displayed'):
                                available.append(use)
                            kinds.append(kind); details.append('%s: %s -> %s' % (item.get('rep') or 'single', kind, location))
                            # Keep recovered clean panels inside the output OPC package. Native data untouched.
                            cache = 'xl/media/xic_edit_source_' + digest(png) + '.png'
                            with Image.open(io.BytesIO(png)) as im:
                                bounds = [0, 0, im.width, im.height]
                            descriptor['embedded'] = {'part': cache, 'sha256': digest(png), 'bounds': bounds}
                            if not preview:
                                book.changed[cache] = png
                        retained.append(descriptor)
                    if not available:
                        raise ValueError('NO_SAFE_IMAGE_SOURCE: the original image is unavailable and cannot be recovered from the audit. Select the original XIC image directory.')
                    width = info['width'] if options.keep_size else options.width
                    height = info['height'] if options.keep_size else options.height
                    if not (120 <= width <= 1200 and 40 <= height <= 500):
                        raise ValueError('UNSUPPORTED_SIZE: turn off existing-size preservation and set width 120-1200 and height 40-500.')
                    if options.keep_size and options.mode != 'keep':
                        if mode == 'first' and triple and oldmode == 'all':
                            height = 220
                        elif mode == 'all' and triple:
                            height = 480
                    conflicts = {r: [{}] for r in '123' if str(entry.get('Rep' + r + '_conflicts', '')).strip()}
                    match = {'selected': available, 'paths': [e['path'] for e in available],
                             'is_triplicate': triple, 'conflicts': conflicts}
                    picture = render_xic(match, width, height, mode, options.captions, compound_name=name)
                    png, width, height, _ = picture
                    outmedia = 'xl/media/xic_reedit_' + digest(png) + '.png'
                    ev.update(Status='PREVIEW_READY' if preview else 'UPDATED', Reason=match['status'],
                              Mode=mode, Is_triplicate=triple,
                              Displayed_replicates='|'.join(e.get('rep', '') for e in match['used']),
                              Source_kinds='|'.join(kinds), Source_details='\n'.join(details),
                              Sources_json=json.dumps(retained, ensure_ascii=False),
                              Caption_records=json.dumps(match['caption_records'], ensure_ascii=False),
                              Output_media=outmedia, Output_sha256=digest(png), Width_px=width, Height_px=height)
                    if preview:
                        continue
                    committing = True
                    newrid = book._new_rel(drel, 'image', posixpath.relpath(outmedia, posixpath.dirname(dp)))
                    info['blip'].set('{%s}embed' % R, newrid)
                    # Add source-only image relationships for future edits; not displayed drawings.
                    targets = {r.get('Target') for r in drel if r.get('Type') == R + '/image'}
                    for source in retained:
                        emb = source.get('embedded', {})
                        if not emb.get('part'):
                            continue
                        target = posixpath.relpath(emb['part'], posixpath.dirname(dp))
                        if target not in targets and book.part(emb['part']):
                            book._new_rel(drel, 'image', target); targets.add(target)
                    exts = [anchor.find('{%s}ext' % D), info['pic'].find('.//{%s}xfrm/{%s}ext' % (A, A))]
                    for ext in exts:
                        if ext is not None:
                            ext.set('cx', str(width * 9525)); ext.set('cy', str(height * 9525))
                    label = 'XIC edit %s | %s | %s' % (VERSION, name, ev['Displayed_replicates'])
                    info['nv'].set('descr', clean_text(label))
                    book.changed[outmedia] = png
                    _update_status(book, root, h['headers'], col, rn, match['status'] + '; XIC_EDIT_' + VERSION)
                    if width != info['width']:
                        _set_col_width(root, col, width)
                    if height != info['height']:
                        row = root.find('.//s:sheetData/s:row[@r="%s"]' % rn, N)
                        oldh = float(row.get('ht', '15'))
                        newh = min(409, (height + 12) * .75)
                        # Never shrink if another drawing occupies this row.
                        others = []
                        for other in drawing:
                            if other is anchor:
                                continue
                            mark = other.find('{%s}from' % D)
                            if mark is not None and mark.findtext('{%s}row' % D) == str(rn - 1):
                                others.append(other)
                        if not others and abs(oldh - min(409, (info['height'] + 12) * .75)) < 1:
                            row.set('ht', str(newh))
                        else:
                            row.set('ht', str(max(oldh, newh)))
                        row.set('customHeight', '1')
                    changed = True
                except Exception as exc:
                    if committing:
                        raise RuntimeError('Failed while updating XIC drawing; output not completed: %s' % exc) from exc
                    ev['Status'] = 'UNCHANGED'
                    ev['Reason'] = str(exc)
            if changed:
                book.changed[part] = _bytes(root)
                book.changed[dp] = _bytes(drawing)
                book.changed[relfile(dp)] = _bytes(drel)
    return events


def _save_without_recalculation(book, output):
    """No formula evaluation, calcPr alteration, column append, or mass imports."""
    output = Path(output)
    if output.resolve() == book.path or output.exists():
        raise ValueError('Save to a new file; input files and existing outputs cannot be overwritten.')
    book.changed.update({'xl/workbook.xml': _bytes(book.book),
                         'xl/_rels/workbook.xml.rels': _bytes(book.bookrels),
                         'xl/styles.xml': _bytes(book.styles), '[Content_Types].xml': _bytes(book.types)})
    tmp = output.with_suffix(output.suffix + '.partial')
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in book.entries:
                z.writestr(path, book.part(path))
            for path, b in book.changed.items():
                if path not in book.entries:
                    z.writestr(path, b)
        with zipfile.ZipFile(tmp) as z:
            if z.testzip():
                raise ValueError('Output archive failed CRC check')
        tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()


def _assert_data_preserved(before, after, events):
    """Fail hard if any original data/formula cell except XIC_status changed."""
    permitted = {}
    header_cache = {}
    for e in events:
        if e['Status'] != 'UPDATED':
            continue
        if e['Sheet'] not in header_cache:
            header_cache[e['Sheet']] = _sheet_headers(before, e['Sheet'])[1]
        hs = header_cache[e['Sheet']]
        match = next(h for h in hs if colname(h['col']) == e['XIC_column'])
        label = match['header'].replace('XIC_plot', 'XIC_status')
        for c, v in match['headers'].items():
            if v == label:
                permitted.setdefault(e['Sheet'], set()).add(colname(c) + str(e['Excel_row']))
    for name in before.sheet_parts:
        b, a = before.sheet(name), after.sheet(name)
        bc = {c.get('r'): E.tostring(c, method='c14n') for c in b.findall('.//s:sheetData/s:row/s:c', N)}
        ac = {c.get('r'): E.tostring(c, method='c14n') for c in a.findall('.//s:sheetData/s:row/s:c', N)}
        if bc.keys() != ac.keys():
            raise RuntimeError('Original cell set changed in %s' % name)
        for key in bc:
            if key not in permitted.get(name, set()) and bc[key] != ac[key]:
                raise RuntimeError('Original cell changed: %s!%s' % (name, key))
    bcalc, acalc = before.book.find(tag('calcPr')), after.book.find(tag('calcPr'))
    if (E.tostring(bcalc) if bcalc is not None else None) != (E.tostring(acalc) if acalc is not None else None):
        raise RuntimeError('Calculation configuration unexpectedly changed')


def execute(paths, output_parent, options=None, progress=None, preview=False):
    options = (options or EditOptions()).validate()
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
    if not paths:
        raise ValueError('Select existing result workbooks. Calibration and target workbooks may be selected together.')
    hashes = {p: file_hash(p) for p in paths}
    folder = None
    if not preview:
        folder = Path(output_parent).expanduser().resolve() / ('xic_reedit_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        folder.mkdir(parents=True, exist_ok=False)
    all_events, outputs = [], []
    try:
        for i, p in enumerate(paths, 1):
            before = Book(p)
            book = Book(p)
            events = edit_book(book, options, progress, preview)
            all_events.extend(events)
            if not events:
                all_events.append({'Workbook': Path(p).name, 'Status': 'UNCHANGED',
                                   'Reason': 'NO_SUPPORTED_XIC: no supported XIC image with source audit was found.'})
            if not preview:
                ev_headers = list(dict.fromkeys(k for e in events for k in e)) or ['Status', 'Reason']
                book.add_audit(AUDIT_BASE, ev_headers, [[e.get(k, '') for k in ev_headers] for e in events])
                book.add_audit('XIC_Edit_Readme', ['Item', 'Value'], [
                    ['Version', VERSION], ['Input', Path(p).name], ['Input_SHA256', hashes[p]],
                    ['Operation', 'Replace existing XIC drawings; no RAW/model/mass/area recalculation.'],
                    ['Time', datetime.now().isoformat(timespec='seconds')],
                    ['Mode', options.mode], ['Captions', options.captions.description()],
                    ['Custom text', options.captions.custom_text],
                    ['Original data', 'Original value/formula cells retained exactly; only XIC_status presentation cells updated. Existing audits are historical and unchanged.'],
                    ['Embedded recovery', 'Only program-generated panel boundaries are used; original plot title/axes/legend are not erased. Resolution cannot exceed existing embedded pixels.'],
                    ['Unavailable replicas', 'A workbook originally containing only one image cannot supply unseen replicate pixels without original images or retained embedded sources.'],
                    ['Privacy', 'Names, source paths and clean source panels remain in audit/media. No labels does not anonymize this workbook.'],
                    ['Excel resave', 'Third-party editors may remove unused source media. Missing sources are reported, not invented.']])
                if not any(x.get('Extension') == 'png' for x in book.types):
                    E.SubElement(book.types, '{%s}Default' % C, Extension='png', ContentType='image/png')
                out = folder / ('%02d__%s_xic_edited%s' % (i, Path(p).stem, Path(p).suffix.lower()))
                _save_without_recalculation(book, out)
                try:
                    _assert_data_preserved(before, Book(out), events)
                except Exception:
                    out.unlink(missing_ok=True)
                    raise
                outputs.append(str(out))
            if file_hash(p) != hashes[p]:
                raise RuntimeError('Input changed during operation: ' + p)
        counts = Counter(e.get('Status') for e in all_events)
        summary = {'version': VERSION, 'preview': preview, 'updated': counts.get('UPDATED', 0),
                   'ready': counts.get('PREVIEW_READY', 0), 'unchanged': counts.get('UNCHANGED', 0),
                   'outputs': outputs, 'options': asdict(options), 'input_hashes': hashes,
                   'status': ('PREVIEW' if preview else 'COMPLETED_WITH_UNCHANGED' if counts.get('UNCHANGED') else 'COMPLETED'),
                   'events': all_events}
        if folder:
            fields = list(dict.fromkeys(k for e in all_events for k in e)) or ['Status']
            with (folder / 'xic_edit_audit.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_events)
            (folder / 'xic_edit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            lines = ['v' + VERSION + ': XIC update complete. Only images and their status were changed; RAW files, masses and concentrations were not reprocessed.',
                     'Replaced: %s; retained unchanged: %s.' % (summary['updated'], summary['unchanged']),
                     'Reasons for retaining images: ' + json.dumps(dict(Counter(e['Reason'] for e in all_events if e['Status'] == 'UNCHANGED')), ensure_ascii=False),
                     'Input hashes and original numeric/formula cells verified unchanged.', *outputs]
            (folder / 'xic_edit_diagnostics.txt').write_text('\n'.join(lines), encoding='utf-8-sig')
            (folder / 'COMPLETED.txt').write_text('\n'.join(lines), encoding='utf-8')
        return folder, summary
    except Exception as exc:
        if folder:
            (folder / 'FAILED.json').write_text(json.dumps({'status': 'FAILED', 'error': type(exc).__name__ + ': ' + str(exc)}, ensure_ascii=False), encoding='utf-8')
        raise


def run_existing_xic_selftest():
    """Frozen-build smoke check, synthetic images only; never uses instrument data."""
    import shutil
    from dataclasses import replace
    from .final_table_selftest import fixture
    from .final_table_postprocess import autoconfig, execute as postprocess, Options
    with tempfile.TemporaryDirectory(prefix='xic_reedit_smoke_') as temp:
        root = Path(temp)
        h = ['Name', 'Formula', 'Exact_mass', 'Combo', 'Sheet']
        path = fixture(root / 'synthetic.xlsx', {'Known': [h, ['K', 'C2H6O', 46.041865, 'KEY', 'STD']],
                                                'Unknown': [h, ['U', 'C2H6O', 46.041865, 'KEY', 'LIB']]})
        images = root / 'images'
        for group, name in [('STD', 'K'), ('LIB', 'U')]:
            for rep in '123':
                folder = images / (group + '-' + rep + '__XIC_plots'); folder.mkdir(parents=True)
                Image.new('RGB', (100, 50), 'white').save(folder / (name + '.png'))
        k = replace(autoconfig(path, 'Known'), image_dir=str(images))
        u = replace(autoconfig(path, 'Unknown'), image_dir=str(images))
        _, r = postprocess(k, u, root / 'original', Options(image_width=520, image_height=480,
                                                          xic_display_mode='all', xic_caption_raw=True))
        before = r['outputs'][0]; hsh = file_hash(before)
        shutil.rmtree(images)
        _, edit = execute([before], root / 'edited')
        if edit['updated'] != 2 or file_hash(before) != hsh:
            raise RuntimeError('Existing XIC edit/input integrity check failed')
        if not all(e['Displayed_replicates'] == '1|2|3' for e in edit['events']):
            raise RuntimeError('Replicate provenance lost during embedded recovery')
        if not all('EMBEDDED_PANEL_RECOVERY' in e['Source_kinds'] for e in edit['events']):
            raise RuntimeError('Source-free embedded panel recovery failed')
    return 'existing XIC replaced; embedded-panel recovery; original cells/cached formulas/input hash unchanged; no RAW/model calls'
