"""Synthetic final-table smoke check for frozen builds; no user files are read."""
import tempfile
import zipfile
from pathlib import Path
from decimal import Decimal
from lxml import etree as E
from .final_table_ooxml import S, P, R, C, tag, colname, Book
def fixture(path, sheets, table=False, extra=None):
    """Create synthetic workbook with explicit styles and original formula cache."""
    wb = E.Element(tag('workbook'), nsmap={None:S, 'r':R})
    sn = E.SubElement(wb, tag('sheets'))
    rels = E.Element('{%s}Relationships'%P, nsmap={None:P})
    types = E.Element('{%s}Types'%C, nsmap={None:C})
    E.SubElement(types, '{%s}Default'%C, Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    E.SubElement(types, '{%s}Default'%C, Extension='xml', ContentType='application/xml')
    E.SubElement(types, '{%s}Override'%C, PartName='/xl/workbook.xml', ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml')
    E.SubElement(types, '{%s}Override'%C, PartName='/xl/styles.xml', ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml')
    files={}
    for i,(name,rows) in enumerate(sheets.items(),1):
        part='xl/worksheets/sheet%s.xml'%i
        E.SubElement(sn, tag('sheet'), name=name, sheetId=str(i), attrib={'{%s}id'%R:'rId%s'%i})
        E.SubElement(rels, '{%s}Relationship'%P, Id='rId%s'%i, Type=R+'/worksheet', Target='worksheets/sheet%s.xml'%i)
        E.SubElement(types, '{%s}Override'%C, PartName='/'+part, ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
        root=E.Element(tag('worksheet'), nsmap={None:S, 'r':R})
        end=colname(max(map(len,rows)))+str(len(rows))
        E.SubElement(root,tag('dimension'),ref='A1:'+end)
        cols=E.SubElement(root,tag('cols')); E.SubElement(cols,tag('col'),min='1',max='20',width='18',customWidth='1')
        data=E.SubElement(root,tag('sheetData'))
        for rn,vs in enumerate(rows,1):
            row=E.SubElement(data,tag('row'),r=str(rn),ht='22',customHeight='1')
            for cn,v in enumerate(vs,1):
                c=E.SubElement(row,tag('c'),r=colname(cn)+str(rn),s='1' if rn==1 else '0')
                if isinstance(v,tuple):
                    E.SubElement(c,tag('f')).text=v[0]; E.SubElement(c,tag('v')).text=str(v[1])
                elif isinstance(v,(int,float,Decimal)):
                    E.SubElement(c,tag('v')).text=str(v)
                elif v is not None:
                    c.set('t','inlineStr'); E.SubElement(E.SubElement(c,tag('is')),tag('t')).text=str(v)
        E.SubElement(root,tag('autoFilter'),ref='A1:'+end)
        if table and i==1:
            E.SubElement(E.SubElement(root,tag('tableParts'),count='1'),tag('tablePart'),attrib={'{%s}id'%R:'rIdTable'})
            tr=E.Element('{%s}Relationships'%P,nsmap={None:P})
            E.SubElement(tr,'{%s}Relationship'%P,Id='rIdTable',Type=R+'/table',Target='../tables/table1.xml')
            files['xl/worksheets/_rels/sheet1.xml.rels']=E.tostring(tr)
            t=E.Element(tag('table'),nsmap={None:S},id='1',name='SyntheticTable',displayName='SyntheticTable',ref='A1:'+end,totalsRowShown='0')
            E.SubElement(t,tag('autoFilter'),ref='A1:'+end)
            tc=E.SubElement(t,tag('tableColumns'),count=str(len(rows[0])))
            for n,h in enumerate(rows[0],1): E.SubElement(tc,tag('tableColumn'),id=str(n),name=str(h))
            files['xl/tables/table1.xml']=E.tostring(t)
            E.SubElement(types,'{%s}Override'%C,PartName='/xl/tables/table1.xml',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml')
        files[part]=E.tostring(root)
    E.SubElement(rels,'{%s}Relationship'%P,Id='rIdStyles',Type=R+'/styles',Target='styles.xml')
    styles='''<styleSheet xmlns="%s"><fonts count="2"><font><sz val="11"/><name val="Arial"/></font><font><b/><sz val="11"/><name val="Arial"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''%S
    files.update({'xl/workbook.xml':E.tostring(wb),'xl/_rels/workbook.xml.rels':E.tostring(rels),'xl/styles.xml':styles.encode(),'[Content_Types].xml':E.tostring(types),
                  '_rels/.rels':('<Relationships xmlns="%s"><Relationship Id="rId1" Type="%s/officeDocument" Target="xl/workbook.xml"/></Relationships>'%(P,R)).encode()})
    files.update(extra or {})
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in files.items():z.writestr(n,b)
    return Path(path)


def run_synthetic_selftest():
    from .final_table_postprocess import autoconfig, execute, sha256
    from dataclasses import replace
    from PIL import Image
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        path = fixture(root/'pair.xlsx', {
            'Known': [['Name','Formula','Exact mass','Combo'], ['S1','C2',24.00001,'A1/B2/C3']],
            'Unknown': [['Name','Formula','Exact mass','Combo'], ['P1','C2',24.00001,'A1/B2/C3']]}, table=True)
        image_dir = root/'images'; image_dir.mkdir()
        Image.new('RGB',(200,80),'white').save(image_dir/'S1.png')
        before = sha256(path)
        known = replace(autoconfig(path,'Known'), image_dir=str(image_dir))
        folder, result = execute(known, autoconfig(path,'Unknown'), root/'out')
        if sha256(path) != before or result['tables']['100']['images_embedded'] != 1:
            raise RuntimeError('Final-table synthetic image/input-integrity check failed')
        out = Book(result['outputs'][0])
        if not any(x.startswith('xl/media/pp_') for x in out.entries):
            raise RuntimeError('Embedded image was not included in Excel')
        h, rows = out.scan('Known',1)
        if not any(v=='P1' for v in rows[0][1].values()):
            raise RuntimeError('100-to-1000 name mapping failed')
    return 'native OOXML + formula/ppm + embedded PNG + name matching + read-only inputs'


def run_triplicate_selftest():
    """Synthetic contract for the frozen executable, not real RAW validation."""
    from .final_table_postprocess import autoconfig, execute, Options, sha256
    from dataclasses import replace
    from PIL import Image
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        combo1 = 'A#1:amine:C2H7N | B#1:azide:C6H5N3 | C#1:alcohol:CH4O'
        combo2 = 'A#4:amine:C2H7N | B#8:azide:C6H5N3 | C#2:alcohol:CH4O'
        path = fixture(root/'pair.xlsx', {
            'Known': [['Name','Formula','Exact mass','Combo','Sheet'], ['S_0001','C2',24.00001,combo1,'S']],
            'Unknown': [['Name','Formula','Exact mass','Combo','Sheet'], ['T_0008','C2',24.00001,combo2,'T']]})
        for group, name in [('S','S_0001'),('T','T_0008')]:
            for rep in '123':
                d = root/'xic'/group/(group+'-'+rep+'__XIC_plots')
                d.mkdir(parents=True)
                Image.new('RGB',(240,70),'white').save(d/('001__'+name+'__mz24.00000.png'))
        before = sha256(path)
        known = replace(autoconfig(path,'Known'), image_dir=str(root/'xic'))
        unknown = replace(autoconfig(path,'Unknown'), image_dir=str(root/'xic'))
        folder, result = execute(known, unknown, root/'out', Options(image_height=480, xic_display_mode='all'))
        for label in ('100','1000'):
            if result['tables'][label]['source_images_embedded'] != 3 or result['tables'][label]['images_embedded'] != 1:
                raise RuntimeError('Three-replicate image smoke check failed')
        if result['tables']['100']['name_matches'] != 1 or sha256(path) != before:
            raise RuntimeError('Renumbered Combo or read-only-input smoke check failed')
    return '3 original images per row + renumbered ABC reagent key + unchanged input'


def run_display_choice_selftest():
    """Both presentation modes in the frozen build; no instrument/sample files."""
    from .final_table_postprocess import autoconfig, execute, Options, sha256
    from dataclasses import replace
    from PIL import Image
    import csv
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        header = ['Name', 'Formula', 'Exact mass', 'Combo', 'Sheet']
        path = fixture(root/'pair.xlsx', {
            'Known': [header, ['S_0001', 'C2', 24.00001, 'same', 'S']],
            'Unknown': [header, ['T_0001', 'C2', 24.00001, 'same', 'T']]})
        for group in ['S', 'T']:
            for rep in '123':
                d = root/'xic'/group/(group+'-'+rep+'__XIC_plots')
                d.mkdir(parents=True)
                Image.new('RGB', (240, 70), 'white').save(d/('001__'+group+'_0001__mz24.00000.png'))
        known = replace(autoconfig(path, 'Known'), image_dir=str(root/'xic'))
        unknown = replace(autoconfig(path, 'Unknown'), image_dir=str(root/'xic'))
        before = sha256(path)
        for mode, count in [('first', 1), ('all', 3)]:
            folder, result = execute(known, unknown, root/'out', Options(xic_display_mode=mode))
            for label in ['100', '1000']:
                if result['tables'][label]['source_images_embedded'] != count:
                    raise RuntimeError('XIC display mode source count failed: '+mode)
            with (folder/'postprocess_audit.csv').open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    expected = '1' if mode=='first' else '1|2|3'
                    if row['Displayed_replicates'] != expected or row['Matched_image_count'] != '3':
                        raise RuntimeError('XIC display provenance failed')
        if sha256(path) != before:
            raise RuntimeError('XIC display change modified input workbook')
    return 'default first Rep1 / all Rep1-3, both tables, full matched-source audit, unchanged input'


def run_numeric_ppm_selftest():
    """Check real zero + signed/tiny numbers are numeric cells, not status text."""
    from .final_table_postprocess import autoconfig, execute, sha256, PPM_FIXED, PPM_SCI, format_ppm_display
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        header = ['Name', 'Formula', 'Exact mass', 'Combo']
        values = [Decimal('24'), Decimal('24.000024'), Decimal('23.999952'), Decimal('24.000000000001')]
        path = fixture(root/'numeric_ppm.xlsx', {
            'Known': [header] + [['S%s'%i, 'C2', v, 'K%s'%i] for i, v in enumerate(values)],
            'Unknown': [header] + [['T%s'%i, 'C2', v, 'K%s'%i] for i, v in enumerate(values)]})
        before = sha256(path)
        folder, result = execute(autoconfig(path, 'Known'), autoconfig(path, 'Unknown'), root/'out')
        out = Book(result['outputs'][0])
        formats = {e.get('numFmtId'): e.get('formatCode') for e in out.styles.find(tag('numFmts'))}
        styles = out.styles.find(tag('cellXfs'))
        for sheet in ['Known', 'Unknown']:
            headers, _ = out.scan(sheet, 1)
            col = next(k for k, v in headers.items() if v == 'Mass_difference_ppm')
            for rn, expected in [(2, Decimal(0)), (3, Decimal(1)), (4, Decimal(-2)), (5, None)]:
                c = out.sheet(sheet).find('.//'+tag('c')+'[@r="'+colname(col)+str(rn)+'"]')
                if c is None or c.get('t') in ('inlineStr', 's', 'str') or c.find(tag('f')) is None:
                    raise RuntimeError('PPM must be a numeric formula cell')
                value = Decimal(c.find(tag('v')).text)
                fmt = formats[styles[int(c.get('s'))].get('numFmtId')]
                if expected is not None and value != expected:
                    raise RuntimeError('Numeric ppm changed: '+str(value))
                if fmt != (PPM_SCI if rn == 5 else PPM_FIXED):
                    raise RuntimeError('PPM format mismatch: '+fmt)
                if rn == 5 and value == 0:
                    raise RuntimeError('Tiny nonzero ppm was rounded to zero')
            if 'consistent' in formats[styles[int(out.sheet(sheet).find('.//'+tag('c')+'[@r="'+colname(col)+'2"]').get('s'))].get('numFmtId')]:
                raise RuntimeError('Zero is still displayed as a text label')
        if format_ppm_display(Decimal(0)) != '0.000000' or sha256(path) != before:
            raise RuntimeError('Preview or original-file integrity failed')
    return 'both tables: numeric zero + signed ppm + tiny nonzero + cached formulas + unchanged inputs'
