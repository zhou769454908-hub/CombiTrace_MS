"""Small append-only OOXML editor for final-table enrichment.

Does not round-trip the workbook through a tabular exporter: untouched ZIP parts
(charts, macros, other worksheets, original media, etc.) are copied byte-for-byte.
Uses lxml/Pillow already required by this application. Does not evaluate formulas;
reads cached values and writes a cached value for each NEW ppm formula.
"""
from __future__ import annotations

import copy
import io
import posixpath
import re
import zipfile
from pathlib import Path
from decimal import Decimal
from typing import Dict, List
from lxml import etree as E

S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
P = 'http://schemas.openxmlformats.org/package/2006/relationships'
C = 'http://schemas.openxmlformats.org/package/2006/content-types'
D = 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
N = {'s': S, 'r': R, 'p': P}


def tag(name):
    return '{%s}%s' % (S, name)


def colname(n):
    if n < 1 or n > 16384:
        raise ValueError('Excel column index out of range: %s' % n)
    text = ''
    while n:
        n, k = divmod(n - 1, 26)
        text = chr(65 + k) + text
    return text


def colindex(text):
    n = 0
    for ch in re.match(r'\$?([A-Z]+)', text.upper()).group(1):
        n = n * 26 + ord(ch) - 64
    return n


def relfile(part):
    return posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels')


def resolve(part, target):
    if target.startswith('/'):
        return target.lstrip('/')
    return posixpath.normpath(posixpath.join(posixpath.dirname(part), target))


def _xml(data):
    return E.fromstring(data, E.XMLParser(resolve_entities=False, no_network=True, remove_blank_text=False))


def _bytes(root):
    return E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)


def clean_text(value):
    return re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value))[:32767]


class Book:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.suffix.lower() not in ('.xlsx', '.xlsm'):
            raise ValueError('Only .xlsx / .xlsm are supported; save legacy .xls as .xlsx first.')
        try:
            with zipfile.ZipFile(self.path) as z:
                self.entries = {i.filename: z.read(i.filename) for i in z.infolist()}
        except zipfile.BadZipFile as exc:
            raise ValueError('Not an unencrypted OOXML workbook: %s' % self.path.name) from exc
        if any(p.startswith('_xmlsignatures/') for p in self.entries):
            raise ValueError('Digitally signed workbook: use an unsigned derivative copy first.')
        self.changed = {}
        self.book = _xml(self.entries['xl/workbook.xml'])
        if self.book.tag != tag('workbook'):
            raise ValueError('Strict OOXML is not supported. Save as a normal Excel .xlsx first.')
        self.bookrels = _xml(self.entries['xl/_rels/workbook.xml.rels'])
        rels = {r.get('Id'): resolve('xl/workbook.xml', r.get('Target')) for r in self.bookrels}
        self.sheet_parts = {r.get('name'): rels[r.get('{%s}id' % R)] for r in self.book.find(tag('sheets'))}
        self.shared = []
        if 'xl/sharedStrings.xml' in self.entries:
            self.shared = [''.join(si.itertext()) for si in _xml(self.entries['xl/sharedStrings.xml'])]
        self.styles = _xml(self.entries.get('xl/styles.xml', b'<styleSheet xmlns="' + S.encode() + b'"/>'))
        self.types = _xml(self.entries['[Content_Types].xml'])
        self._style_cache = {}

    def part(self, name):
        return self.changed.get(name, self.entries.get(name))

    def sheet(self, name):
        if name not in self.sheet_parts:
            raise ValueError('Worksheet not found: %s' % name)
        return _xml(self.part(self.sheet_parts[name]))

    def cell_value(self, cell):
        t = cell.get('t')
        if t == 'inlineStr':
            return ''.join(cell.find(tag('is')).itertext()) if cell.find(tag('is')) is not None else ''
        v = cell.find(tag('v'))
        if v is None or v.text is None:
            return ''
        if t == 's':
            return self.shared[int(v.text)]
        if t == 'b':
            return 'TRUE' if v.text == '1' else 'FALSE'
        return v.text

    def grid(self, name):
        result = {}
        for row in self.sheet(name).findall('s:sheetData/s:row', N):
            r = int(row.get('r'))
            result[r] = {colindex(c.get('r')): self.cell_value(c) for c in row.findall(tag('c'))}
        return result

    def scan(self, name, header_row=1):
        grid = self.grid(name)
        header = {k: str(v).strip() for k, v in grid.get(header_row, {}).items() if str(v).strip()}
        rows = [(r, values) for r, values in sorted(grid.items()) if r > header_row and any(str(v).strip() for v in values.values())]
        return header, rows

    def _style(self, base, fmt=None, header=False):
        key = (base, fmt, header)
        if key in self._style_cache:
            return self._style_cache[key]
        xfs = self.styles.find(tag('cellXfs'))
        if xfs is None:
            raise ValueError('Workbook has no valid style table.')
        xf = copy.deepcopy(xfs[base if base < len(xfs) else 0])
        if fmt is not None:
            nfs = self.styles.find(tag('numFmts'))
            if nfs is None:
                nfs = E.Element(tag('numFmts'), count='0')
                self.styles.insert(0, nfs)
            existing = next((e.get('numFmtId') for e in nfs if e.get('formatCode') == fmt), None)
            if existing is None:
                existing = str(max([163] + [int(e.get('numFmtId')) for e in nfs]) + 1)
                E.SubElement(nfs, tag('numFmt'), numFmtId=existing, formatCode=fmt)
                nfs.set('count', str(len(nfs)))
            xf.set('numFmtId', existing)
            xf.set('applyNumberFormat', '1')
        alignment = xf.find(tag('alignment'))
        if alignment is None:
            alignment = E.SubElement(xf, tag('alignment'))
        alignment.set('vertical', 'center')
        alignment.set('wrapText', '1')
        if header:
            alignment.set('horizontal', 'center')
        xf.set('applyAlignment', '1')
        idx = len(xfs)
        xfs.append(xf)
        xfs.set('count', str(len(xfs)))
        self._style_cache[key] = idx
        return idx

    @staticmethod
    def _cell(row, col, value, style, formula=None):
        c = E.SubElement(row, tag('c'), r=colname(col) + row.get('r'), s=str(style))
        if formula:
            E.SubElement(c, tag('f')).text = formula.lstrip('=')
            if value is not None:
                E.SubElement(c, tag('v')).text = str(value)
        elif isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
            E.SubElement(c, tag('v')).text = str(value)
        elif value is not None and str(value) != '':
            c.set('t', 'inlineStr')
            E.SubElement(E.SubElement(c, tag('is')), tag('t')).text = clean_text(value)
        return c

    def _add_type(self, path, typ):
        if not any(x.get('PartName') == '/' + path for x in self.types):
            E.SubElement(self.types, '{%s}Override' % C, PartName='/' + path, ContentType=typ)

    def _new_rel(self, root, typ, target):
        ids = {x.get('Id') for x in root}
        k = 1
        while 'rIdPP%s' % k in ids:
            k += 1
        rid = 'rIdPP%s' % k
        E.SubElement(root, '{%s}Relationship' % P, Id=rid, Type=R + '/' + typ, Target=target)
        return rid

    def append(self, name, header_row, columns, rows, pictures=None, image_width=480, image_height=190):
        """rows: Excel row -> column-name -> {value, formula, format}; images are real PNG bytes."""
        part = self.sheet_parts[name]
        root = self.sheet(name)
        if root.find(tag('sheetProtection')) is not None:
            raise ValueError('Worksheet is protected: %s. Make an authorized unprotected copy first.' % name)
        data = root.find(tag('sheetData'))
        used = [colindex(c.get('r')) for c in data.findall('s:row/s:c', N)]
        # Do not append into an existing merged header block or reserved styled cells.
        for m in root.findall('s:mergeCells/s:mergeCell', N):
            used.append(colindex(m.get('ref').split(':')[-1]))
        last_col = max(used or [1])
        if last_col + len(columns) > 16384:
            raise ValueError('Too many Excel columns.')
        mapping = {key: last_col + i + 1 for i, key in enumerate(columns)}
        rowmap = {int(r.get('r')): r for r in data.findall(tag('row'))}
        if header_row not in rowmap:
            raise ValueError('Header row not found.')
        existing_headers = {self.cell_value(c) for c in rowmap[header_row].findall(tag('c'))}
        output_headers = {}
        for key in columns:
            text = key
            n = 2
            while text in existing_headers:
                text = key + '_PP%s' % n
                n += 1
            existing_headers.add(text)
            output_headers[key] = text
        hc = rowmap[header_row].findall(tag('c'))
        header_style = self._style(int(hc[-1].get('s', '0')) if hc else 0, header=True)
        rowmap[header_row].set('ht', str(max(32, float(rowmap[header_row].get('ht', '15')))))
        rowmap[header_row].set('customHeight', '1')
        for key, col in mapping.items():
            self._cell(rowmap[header_row], col, output_headers[key], header_style)
        for rn, vals in sorted(rows.items()):
            row = rowmap.get(rn)
            if row is None:
                row = E.SubElement(data, tag('row'), r=str(rn))
                rowmap[rn] = row
            old_cells = row.findall(tag('c'))
            base = int(old_cells[-1].get('s', '0')) if old_cells else 0
            for key, col in mapping.items():
                spec = vals.get(key, {})
                formula = spec.get('formula')
                if formula:
                    for refkey, refcol in mapping.items():
                        formula = formula.replace('{' + refkey + '}', colname(refcol) + str(rn))
                self._cell(row, col, spec.get('value'), self._style(base, spec.get('format')), formula)
            row.set('spans', '1:%s' % (last_col + len(columns)))
        # New column widths; all original widths and print settings are retained.
        cols = root.find(tag('cols'))
        if cols is None:
            cols = E.Element(tag('cols'))
            root.insert(list(root).index(data), cols)
        # Split any existing width range covering previously empty appended columns.
        # Overlapping <col> ranges can otherwise trigger Excel repair on wide templates.
        lo_new, hi_new = last_col + 1, last_col + len(columns)
        kept_cols = []
        for definition in list(cols):
            lo, hi = int(definition.get('min')), int(definition.get('max'))
            if hi < lo_new or lo > hi_new:
                kept_cols.append(definition)
            else:
                if lo < lo_new:
                    left = copy.deepcopy(definition); left.set('max', str(lo_new - 1)); kept_cols.append(left)
                if hi > hi_new:
                    right = copy.deepcopy(definition); right.set('min', str(hi_new + 1)); kept_cols.append(right)
            cols.remove(definition)
        for key, col in mapping.items():
            width = (image_width + 12) / 7.0 if key == 'XIC_plot' else (32 if key in ('Mass_QC', 'Match_status', 'XIC_status', 'Name_in_1000') else 22)
            kept_cols.append(E.Element(tag('col'), min=str(col), max=str(col), width=str(width), customWidth='1'))
        cols.extend(sorted(kept_cols, key=lambda x: int(x.get('min'))))
        max_row = max(rowmap)
        dim = root.find(tag('dimension'))
        if dim is None:
            dim = E.Element(tag('dimension'))
            root.insert(1 if root.find(tag('sheetPr')) is not None else 0, dim)
        dim.set('ref', 'A1:%s%s' % (colname(last_col + len(columns)), max_row))
        af = root.find(tag('autoFilter'))
        if af is not None and af.get('ref'):
            start, end = (af.get('ref').split(':') + [af.get('ref')])[:2]
            if colindex(end) == last_col:
                af.set('ref', '%s:%s%s' % (start, colname(last_col + len(columns)), re.search(r'\d+$', end).group()))
        if pictures:
            self._pictures(part, root, mapping['XIC_plot'], pictures, rowmap, image_width, image_height)
        # Existing Excel Tables covering the whole original header gain the appended columns.
        rels_data = self.part(relfile(part))
        if rels_data:
            rels = _xml(rels_data)
            for rel in rels:
                if not rel.get('Type', '').endswith('/table'):
                    continue
                tp = resolve(part, rel.get('Target'))
                table = _xml(self.part(tp))
                end = table.get('ref').split(':')[-1]
                start = table.get('ref').split(':')[0]
                if colindex(end) == last_col and int(re.search(r'\d+$', start).group()) == header_row:
                    newref = start + ':' + colname(last_col + len(columns)) + re.search(r'\d+$', end).group()
                    table.set('ref', newref)
                    ta = table.find(tag('autoFilter'))
                    if ta is not None:
                        ta.set('ref', newref)
                    tc = table.find(tag('tableColumns'))
                    next_id = max([0] + [int(x.get('id')) for x in tc]) + 1
                    for key in columns:
                        E.SubElement(tc, tag('tableColumn'), id=str(next_id), name=output_headers[key])
                        next_id += 1
                    tc.set('count', str(len(tc)))
                    self.changed[tp] = _bytes(table)
        self.changed[part] = _bytes(root)
        return mapping

    def _pictures(self, part, root, col, pictures, rowmap, width, height):
        rp = relfile(part)
        rels = _xml(self.part(rp)) if self.part(rp) else E.Element('{%s}Relationships' % P, nsmap={None: P})
        drawing_el = root.find(tag('drawing'))
        if drawing_el is not None:
            rid = drawing_el.get('{%s}id' % R)
            target = next(x.get('Target') for x in rels if x.get('Id') == rid)
            dp = resolve(part, target)
            drawing = _xml(self.part(dp))
        else:
            k = 1
            while self.part('xl/drawings/pp_drawing%s.xml' % k):
                k += 1
            dp = 'xl/drawings/pp_drawing%s.xml' % k
            drawing = E.Element('{%s}wsDr' % D, nsmap={'xdr': D, 'a': A})
            rid = self._new_rel(rels, 'drawing', posixpath.relpath(dp, posixpath.dirname(part)))
            drawing_el = E.Element(tag('drawing'), {'{%s}id' % R: rid})
            # Insert in SpreadsheetML child order, before legacy drawings/tables/extensions.
            late = {'legacyDrawing', 'legacyDrawingHF', 'drawingHF', 'picture', 'oleObjects', 'controls', 'webPublishItems', 'tableParts', 'extLst'}
            idx = next((i for i, x in enumerate(root) if E.QName(x).localname in late), len(root))
            root.insert(idx, drawing_el)
        drp = relfile(dp)
        drel = _xml(self.part(drp)) if self.part(drp) else E.Element('{%s}Relationships' % P, nsmap={None: P})
        pic_id = max([0] + [int(x.get('id', '0')) for x in drawing.findall('.//{%s}cNvPr' % D)]) + 1
        import hashlib
        for rn, image in sorted(pictures.items()):
            png, w, h, label = image
            media = 'xl/media/pp_' + hashlib.sha256(png).hexdigest() + '.png'
            self.changed[media] = png
            irid = self._new_rel(drel, 'image', posixpath.relpath(media, posixpath.dirname(dp)))
            anchor = E.SubElement(drawing, '{%s}oneCellAnchor' % D)
            mark = E.SubElement(anchor, '{%s}from' % D)
            for tagname, val in [('col', col-1), ('colOff', 4 * 9525), ('row', rn-1), ('rowOff', 4 * 9525)]:
                E.SubElement(mark, '{%s}%s' % (D, tagname)).text = str(val)
            E.SubElement(anchor, '{%s}ext' % D, cx=str(w*9525), cy=str(h*9525))
            pic = E.SubElement(anchor, '{%s}pic' % D)
            nv = E.SubElement(pic, '{%s}nvPicPr' % D)
            E.SubElement(nv, '{%s}cNvPr' % D, id=str(pic_id), name='Postprocess XIC %s' % rn, descr=clean_text(label))
            E.SubElement(nv, '{%s}cNvPicPr' % D)
            fill = E.SubElement(pic, '{%s}blipFill' % D)
            E.SubElement(fill, '{%s}blip' % A, {'{%s}embed' % R: irid})
            E.SubElement(E.SubElement(fill, '{%s}stretch' % A), '{%s}fillRect' % A)
            sp = E.SubElement(pic, '{%s}spPr' % D)
            transform = E.SubElement(sp, '{%s}xfrm' % A)
            E.SubElement(transform, '{%s}off' % A, x='0', y='0')
            E.SubElement(transform, '{%s}ext' % A, cx=str(w*9525), cy=str(h*9525))
            geom = E.SubElement(sp, '{%s}prstGeom' % A, prst='rect')
            E.SubElement(geom, '{%s}avLst' % A)
            E.SubElement(anchor, '{%s}clientData' % D)
            rowmap[rn].set('ht', str(min(409, max(float(rowmap[rn].get('ht', '15')), (h + 12) * .75))))
            rowmap[rn].set('customHeight', '1')
            pic_id += 1
        self.changed[rp] = _bytes(rels)
        self.changed[dp] = _bytes(drawing)
        self.changed[drp] = _bytes(drel)
        self._add_type(dp, 'application/vnd.openxmlformats-officedocument.drawing+xml')
        if not any(x.get('Extension') == 'png' for x in self.types):
            E.SubElement(self.types, '{%s}Default' % C, Extension='png', ContentType='image/png')

    def add_audit(self, base_name, headers, records):
        name = base_name[:31]
        i = 2
        while name in self.sheet_parts:
            name = base_name[:26] + '_%s' % i
            i += 1
        k = 1
        while self.part('xl/worksheets/pp_audit%s.xml' % k):
            k += 1
        part = 'xl/worksheets/pp_audit%s.xml' % k
        root = E.Element(tag('worksheet'), nsmap={None: S, 'r': R})
        E.SubElement(root, tag('dimension'), ref='A1:%s%s' % (colname(len(headers)), len(records)+1))
        view = E.SubElement(E.SubElement(root, tag('sheetViews')), tag('sheetView'), workbookViewId='0')
        E.SubElement(view, tag('pane'), ySplit='1', topLeftCell='A2', activePane='bottomLeft', state='frozen')
        cols = E.SubElement(root, tag('cols'))
        E.SubElement(cols, tag('col'), min='1', max=str(len(headers)), width='25', customWidth='1')
        data = E.SubElement(root, tag('sheetData'))
        for idx, values in enumerate([headers] + records, 1):
            r = E.SubElement(data, tag('row'), r=str(idx))
            for c, value in enumerate(values, 1):
                self._cell(r, c, value, self._style(0, header=idx == 1))
        E.SubElement(root, tag('autoFilter'), ref='A1:%s%s' % (colname(len(headers)), len(records)+1))
        self.changed[part] = _bytes(root)
        rid = self._new_rel(self.bookrels, 'worksheet', posixpath.relpath(part, 'xl'))
        sheets = self.book.find(tag('sheets'))
        sid = max(int(x.get('sheetId')) for x in sheets) + 1
        E.SubElement(sheets, tag('sheet'), name=name, sheetId=str(sid), attrib={'{%s}id' % R: rid})
        self.sheet_parts[name] = part
        self._add_type(part, 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')

    def save(self, path):
        path = Path(path)
        if path.resolve() == self.path:
            raise ValueError('Input overwrite is forbidden.')
        if path.exists():
            raise FileExistsError('Output already exists: %s' % path)
        calc = self.book.find(tag('calcPr'))
        if calc is None:
            calc = E.Element(tag('calcPr'))
            late = {'oleSize', 'customWorkbookViews', 'pivotCaches', 'smartTagPr', 'smartTagTypes', 'webPublishing', 'fileRecoveryPr', 'webPublishObjects', 'extLst'}
            index = next((i for i, x in enumerate(self.book) if E.QName(x).localname in late), len(self.book))
            self.book.insert(index, calc)
        calc.set('fullCalcOnLoad', '1')
        calc.set('forceFullCalc', '1')
        self.changed.update({'xl/workbook.xml': _bytes(self.book), 'xl/_rels/workbook.xml.rels': _bytes(self.bookrels),
                             'xl/styles.xml': _bytes(self.styles), '[Content_Types].xml': _bytes(self.types)})
        tmp = path.with_suffix(path.suffix + '.partial')
        try:
            with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for name, content in self.entries.items():
                    z.writestr(name, self.changed.get(name, content))
                for name, content in self.changed.items():
                    if name not in self.entries:
                        z.writestr(name, content)
            with zipfile.ZipFile(tmp) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError('Corrupt output ZIP part: %s' % bad)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path
