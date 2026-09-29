"""Synthetic software tests only; never use real experimental data.

Fixtures are small native OOXML packages, so tests require no Office engine and
exercise preservation of original cell XML, relationships, and cached formulas.
"""
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from lxml import etree as E
from PIL import Image, ImageDraw
from core.final_table_ooxml import Book, S, P, R, C, tag, colname
from core.final_table_postprocess import (TableConfig, Options, theoretical_value, parse_molecular_formula,
    formula_key, mass_result, autoconfig, load_rows, match_names, Images, execute, sha256, PPM_SCI)


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


def synthetic_png(path, label='SYNTHETIC XIC / NOT EXPERIMENTAL DATA'):
    path.parent.mkdir(parents=True,exist_ok=True)
    im=Image.new('RGB',(900,260),'white'); d=ImageDraw.Draw(im)
    d.text((20,15),label,fill='black')
    d.line([(45,55),(45,225),(860,225)],fill='black',width=2)
    points=[]
    import math
    for x in range(60,851):
        y=215-140*math.exp(-.5*((x-400)/30)**2)-60*math.exp(-.5*((x-610)/18)**2)
        points.append((x,y))
    d.line(points,fill='black',width=3)
    im.save(path)
    return path


class FinalPostTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(); self.root=Path(self.t.name)
        self.headers=['Name','Formula','Exact mass','Combo','Concentration','Adduct','XIC_PNG','Group']
        self.krows=[self.headers,['STD A','C6H12O6',180.0634,'A1/B2/C3',.5,'[M-H]-','','run1'],['STD B','C2',24,'A2/B3/C4',.7,'[M-H]-','','run1']]
        self.urows=[self.headers,['P002','C2',24,'A2/B3/C4',.3,'[M-H]-','','run1'],['P001','C6H12O6',180.0634,'A1/B2/C3',.6,'[M-H]-','','run1']]
        self.path=fixture(self.root/'pair.xlsx', {'Known':self.krows,'Unknown':self.urows,'Untouched':[['Note','Cached formula'],['keep',('1+2',3)]]}, table=True, extra={'custom/retain.bin':b'ORIGINAL_BYTES'})
        self.k=autoconfig(self.path,'Known','100'); self.u=autoconfig(self.path,'Unknown','1000')

    def tearDown(self): self.t.cleanup()

    def test_formula_hydrates_parentheses(self):
        self.assertEqual(parse_molecular_formula('Fe2(SO4)3'),{'Fe':2,'S':3,'O':12})
        self.assertEqual(parse_molecular_formula('CuSO4·5H2O'),{'Cu':1,'S':1,'O':9,'H':10})
        self.assertEqual(parse_molecular_formula('C₆H₁₂O₆'),{'C':6,'H':12,'O':6})

    def test_isotopes(self):
        self.assertEqual(theoretical_value('[13C]H4'),Decimal('13.00335483507')+4*Decimal('1.00782503223'))
        self.assertNotEqual(theoretical_value('C2D6'),theoretical_value('C2H6'))

    def test_invalid_formula_does_not_silently_fix(self):
        for f in ['C6H12O6-', 'C6H12O6 2+', 'C0H2', 'C6(H12', 'Xx3', 'C6H12O6·', 'foo', 'C6H12O6垃圾']:
            with self.subTest(f=f),self.assertRaises(ValueError):parse_molecular_formula(f)

    def test_neutral_mass_precision(self):
        self.assertEqual(theoretical_value('C6H12O6'),Decimal('180.06338810418'))

    def test_adducts_charges(self):
        from core.final_table_postprocess import ELECTRON, MASS
        m=theoretical_value('C6H12O6')
        self.assertEqual(theoretical_value('C6H12O6',True,'[M-H]-'),m-MASS['H']+ELECTRON)
        self.assertEqual(theoretical_value('C6H12O6',True,'[M-2H]2-'),(m-2*MASS['H']+2*ELECTRON)/2)
        self.assertEqual(theoretical_value('C6H12O6',True,'[2M+H]+'),2*m+MASS['H']-ELECTRON)
        self.assertEqual(theoretical_value('C6H12O6',True,'[M+FA-H]-'),theoretical_value('C6H12O6',True,'[M+HCOO]-'))
        with self.assertRaises(ValueError):theoretical_value('C2',True,'[M-H]-')
        with self.assertRaises(ValueError):theoretical_value('C2H6',True,'unknown')

    def test_auto_exact_mass_is_reference(self):
        h, rows=load_rows(Book(self.path),self.k)
        m=mass_result(rows[0],self.k,h,5)
        self.assertEqual(m['mode'],'reference_neutral'); self.assertIn('NOT_MEASUREMENT',m['status'])
        self.assertNotEqual(m['ppm'],0)
        obs=mass_result(rows[0],replace(self.k,mass_mode='observed_neutral'),h,5)
        self.assertIn('OBSERVED_NEUTRAL',obs['status'])

    def test_ppm_zero_truth_and_tiny_format(self):
        h,rows=load_rows(Book(self.path),self.k)
        m=mass_result(rows[1],self.k,h,5); self.assertEqual(m['ppm'],0)
        self.assertEqual(m['format'], '+0.000000;-0.000000;0.000000')
        rows[1]['values'][3]='24.000000000001'
        m=mass_result(rows[1],self.k,h,5); self.assertNotEqual(m['ppm'],0); self.assertEqual(m['format'],PPM_SCI)

    def test_invalid_and_missing_mass_not_zero(self):
        h,rows=load_rows(Book(self.path),self.k)
        for v in ['', 'nan', '0', '-3', '#VALUE!']:
            rows[0]['values'][3]=v
            m=mass_result(rows[0],self.k,h,5)
            self.assertIsNone(m['ppm']);self.assertIsNotNone(m['theory'])
        m=mass_result(rows[0],replace(self.k,mass_col=0),h,5); self.assertIn('NO_MASS_COLUMN',m['status'])

    def test_match_uses_key_not_row(self):
        b=Book(self.path);h,k=load_rows(b,self.k);j,u=load_rows(b,self.u)
        m=match_names(k,u,h,j,self.k,self.u,Options())
        self.assertEqual(m[2]['name'],'P001'); self.assertEqual(m[3]['name'],'P002')

    def test_formula_only_default_off_and_ambiguous(self):
        b=Book(self.path);h,k=load_rows(b,self.k);j,u=load_rows(b,self.u)
        for x in k+u:x['values'][4]=''
        self.assertEqual(match_names(k,u,h,j,self.k,self.u,Options())[2]['status'],'NOT_MATCHED')
        m=match_names(k,u,h,j,self.k,self.u,Options(allow_unique_formula=True))
        self.assertEqual(m[2]['status'],'MATCHED_REVIEW_FORMULA_ONLY')
        u.append(dict(u[1]))
        self.assertEqual(match_names(k,u,h,j,self.k,self.u,Options(allow_unique_formula=True))[2]['status'],'AMBIGUOUS')

    def test_formula_conflict_blocks_identity(self):
        b=Book(self.path);h,k=load_rows(b,self.k);j,u=load_rows(b,self.u)
        u[1]['formula']='C2'
        self.assertEqual(match_names(k,u,h,j,self.k,self.u,Options())[2]['status'],'FORMULA_CONFLICT')

    def test_explicit_key_must_be_set_both(self):
        b=Book(self.path);h,k=load_rows(b,self.k);j,u=load_rows(b,self.u)
        with self.assertRaises(ValueError):match_names(k,u,h,j,replace(self.k,key_col=4),self.u,Options())

    def test_filename_formats_not_substring(self):
        d=self.root/'images';synthetic_png(d/'003__STD_A__mz179.05611.png');synthetic_png(d/'STD A10.png')
        h,rows=load_rows(Book(self.path),self.k)
        self.assertEqual(Images(d).find(rows[0],self.k)['path'].name,'003__STD_A__mz179.05611.png')
        rows[0]['name']='STD A1';self.assertEqual(Images(d).find(rows[0],self.k)['status'],'NOT_FOUND')

    def test_images_duplicate_and_group(self):
        d=self.root/'images';synthetic_png(d/'run1'/'STD A.png');synthetic_png(d/'run2'/'STD A.png')
        h,rows=load_rows(Book(self.path),self.k)
        self.assertEqual(Images(d).find(rows[0],self.k)['status'],'AMBIGUOUS_IMAGES')
        self.assertEqual(Images(d).find(rows[0],replace(self.k,group_col=8))['path'].parent.name,'run1')

    def test_full_export_preserves_original_and_formula(self):
        d=self.root/'images'; synthetic_png(d/'STD A.png');synthetic_png(d/'STD B.png')
        before=sha256(self.path); b=Book(self.path)
        folder,result=execute(replace(self.k,image_dir=str(d)),self.u,self.root/'out')
        self.assertEqual(sha256(self.path),before);self.assertEqual(len(result['outputs']),1)
        out=Book(result['outputs'][0]); self.assertEqual(out.entries['custom/retain.bin'],b.entries['custom/retain.bin'])
        self.assertEqual(out.entries['xl/worksheets/sheet3.xml'],b.entries['xl/worksheets/sheet3.xml'])
        for name in ('Known','Unknown'):
            orig=b.sheet(name).findall('.//'+tag('c'))
            now={c.get('r'):c for c in out.sheet(name).findall('.//'+tag('c'))}
            for cell in orig:self.assertEqual(E.tostring(cell),E.tostring(now[cell.get('r')]))
        heads,rs=out.scan('Known',1)
        self.assertIn('Name_in_1000',heads.values()); self.assertEqual(result['tables']['100']['images_embedded'],2)
        table=E.fromstring(out.entries['xl/tables/table1.xml'])
        self.assertEqual(len(table.find(tag('tableColumns'))),len(self.headers)+7)
        imageparts=[n for n in out.entries if n.startswith('xl/media/pp_')]
        self.assertEqual(len(imageparts),1) # identical test PNG bytes deduplicated
        anchors=E.fromstring(next(v for n,v in out.entries.items() if n.startswith('xl/drawings/pp_drawing') and n.endswith('.xml')))
        self.assertEqual(len(anchors),2)
        self.assertTrue((folder/'COMPLETED.txt').exists())
        # Theory is stored at full precision although six decimals are displayed.
        c=out.sheet('Known').find('.//'+tag('c')+'[@r="I2"]')
        self.assertEqual(Decimal(c.find(tag('v')).text),Decimal('180.06338810418'))
        ppm=out.sheet('Known').find('.//'+tag('c')+'[@r="J2"]')
        self.assertIn('I2',ppm.find(tag('f')).text)
        self.assertGreater(Decimal(ppm.find(tag('v')).text),0)

    def test_no_mass_column_still_images_names(self):
        _,r=execute(replace(self.k,mass_col=0),replace(self.u,mass_col=0),self.root/'out')
        self.assertEqual(r['tables']['100']['ppm_calculated'],0); self.assertEqual(r['tables']['100']['name_matches'],2)

    def test_two_books_equal_basename_audit(self):
        p=self.root/'sub';p.mkdir();other=fixture(p/'pair.xlsx',{'Unknown':self.urows})
        uc=autoconfig(other,'Unknown','1000')
        _,r=execute(self.k,uc,self.root/'out')
        self.assertEqual(len(r['outputs']),2)
        for path in r['outputs']:
            b=Book(path);h,rows=b.scan('Postprocess_Audit',1)
            self.assertEqual(len(rows),2)

    def test_reject_same_sheet_input_overwrite_and_protected(self):
        with self.assertRaises(ValueError):execute(self.k,self.k,self.root/'out')
        with self.assertRaises(ValueError):Book(self.path).save(self.path)
        b=Book(self.path);rt=b.sheet('Known');E.SubElement(rt,tag('sheetProtection'),sheet='1');b.changed[b.sheet_parts['Known']]=E.tostring(rt)
        with self.assertRaises(ValueError):b.append('Known',1,['New'],{})

    def test_reprocess_adds_no_colliding_headers_or_drawing_ids(self):
        d=self.root/'imgs';synthetic_png(d/'STD A.png')
        folder,r=execute(replace(self.k,image_dir=str(d)),self.u,self.root/'out')
        path=Path(r['outputs'][0]);kc=autoconfig(path,'Known','100');uc=autoconfig(path,'Unknown','1000')
        _,r2=execute(replace(kc,image_dir=str(d)),uc,self.root/'out')
        b=Book(r2['outputs'][0]);h,_=b.scan('Known',1)
        self.assertIn('Theoretical_mass_PP2',h.values())
        self.assertIn('Postprocess_Audit_2',b.sheet_parts)

    def test_duplicate_claims_no_reuse(self):
        krows=self.krows+[list(self.krows[1])]
        p=fixture(self.root/'dup.xlsx',{'Known':krows,'Unknown':self.urows})
        d=self.root/'imgs';synthetic_png(d/'STD A.png')
        _,r=execute(replace(autoconfig(p,'Known'),image_dir=str(d)),autoconfig(p,'Unknown'),self.root/'out')
        self.assertEqual(r['tables']['100']['images_embedded'],0)

    def test_variable_rows_94_and_1000(self):
        k=[self.headers]+[['S%s'%i,'C2',24,'K%s'%i,.5,'','',''] for i in range(94)]
        u=[self.headers]+[['P%s'%i,'C2',24,'K%s'%i,.5,'','',''] for i in range(1000)]
        path=fixture(self.root/'large.xlsx',{'Known':k,'Unknown':u})
        _,r=execute(autoconfig(path,'Known'),autoconfig(path,'Unknown'),self.root/'out')
        self.assertEqual(r['tables']['100']['rows'],94);self.assertEqual(r['tables']['1000']['rows'],1000)
        self.assertEqual(r['tables']['100']['name_matches'],94)


if __name__ == '__main__': unittest.main()
