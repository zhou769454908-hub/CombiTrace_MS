"""Numeric-only ppm display contract; synthetic fixtures, no experimental data."""
import csv
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from lxml import etree as E
from core.final_table_postprocess import (
    PPM_FIXED, PPM_SCI, ppm_number_format, format_ppm_display,
    mass_result, autoconfig, load_rows, execute, sha256, Options,
)
from core.final_table_selftest import fixture, run_numeric_ppm_selftest
from core.final_table_ooxml import Book, tag, colname


class NumericPPMTests(unittest.TestCase):
    def test_zero_format_is_numeric(self):
        self.assertEqual(PPM_FIXED, '+0.000000;-0.000000;0.000000')
        self.assertEqual(PPM_SCI.split(';')[2], '0.000000')
        for value in [0, 0.0, Decimal('-0'), Decimal('0E-26')]:
            self.assertEqual(ppm_number_format(value), PPM_FIXED)
            self.assertEqual(format_ppm_display(value), '0.000000')

    def test_signed_regular_values(self):
        for value, expected in [('1.234567', '+1.234567'), ('-2.5', '-2.500000'), ('3', '+3.000000')]:
            self.assertEqual(format_ppm_display(value), expected)
            self.assertEqual(ppm_number_format(value), PPM_FIXED)

    def test_tiny_values_and_boundary_do_not_display_zero(self):
        for x in ['5e-7', '-5e-7', '1e-18', '-1e-25', '4.99999e-7']:
            self.assertEqual(ppm_number_format(x), PPM_SCI)
            self.assertIn('E', format_ppm_display(x))
            self.assertNotEqual(Decimal(format_ppm_display(x)), 0)
        self.assertEqual(ppm_number_format('5.001e-7'), PPM_FIXED)
        self.assertNotEqual(Decimal(format_ppm_display('5.001e-7')), 0)

    def test_unavailable_preview_is_not_zero(self):
        for x in [None, '', 'NaN', '-Infinity', '#VALUE!']:
            self.assertEqual(format_ppm_display(x), '')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.header = ['Name','Formula','Exact mass','Combo']
        self.data = [
            ['S0', 'C2', Decimal('24'), 'K0'],
            ['Spos', 'C2', Decimal('24.000048'), 'K1'],
            ['Sneg', 'C2', Decimal('23.999976'), 'K2'],
            ['Stiny', 'C2', Decimal('24.000000000001'), 'K3'],
            ['Smissing', 'C2', None, 'K4'],
            ['Sbad', 'XYZ', Decimal('24'), 'K5'],
        ]
        self.path = fixture(self.root/'input.xlsx', {
            'Known': [self.header]+self.data,
            'Unknown': [self.header]+[[r[0].replace('S','T',1)]+r[1:] for r in self.data],
        })
        self.k = autoconfig(self.path, 'Known')
        self.u = autoconfig(self.path, 'Unknown')

    def tearDown(self):
        self.temp.cleanup()

    def export(self, mode='first'):
        return execute(self.k, self.u, self.root/'out', Options(xic_display_mode=mode))

    def test_mass_arithmetic_and_reference_qc_unchanged(self):
        h, rows = load_rows(Book(self.path), self.k)
        for row, expected in zip(rows[:3], [0,2,-1]):
            result = mass_result(row, self.k, h, 5)
            self.assertEqual(result['ppm'], Decimal(expected))
            self.assertIn('REFERENCE_ONLY_NOT_MEASUREMENT', result['status'])
            self.assertEqual(result['theory'], 24)
        self.assertEqual(mass_result(rows[0], replace(self.k, mass_mode='observed_neutral'), h, 5)['ppm'],0)

    def test_exported_both_tables_are_numeric_formulas(self):
        folder, result = self.export()
        out = Book(result['outputs'][0])
        for sheet in ['Known','Unknown']:
            h,_ = out.scan(sheet,1)
            col = next(k for k,v in h.items() if v == 'Mass_difference_ppm')
            for rn, expected in [(2,0),(3,2),(4,-1)]:
                c = out.sheet(sheet).find('.//'+tag('c')+'[@r="'+colname(col)+str(rn)+'"]')
                self.assertNotIn(c.get('t'), ['inlineStr','str','s'])
                self.assertIsNotNone(c.find(tag('f')))
                self.assertEqual(Decimal(c.find(tag('v')).text),expected)
                fmt_id = out.styles.find(tag('cellXfs'))[int(c.get('s'))].get('numFmtId')
                fmt = next(f.get('formatCode') for f in out.styles.find(tag('numFmts')) if f.get('numFmtId') == fmt_id)
                self.assertEqual(fmt,PPM_FIXED)
        self.assertTrue((folder/'COMPLETED.txt').exists())

    def test_missing_invalid_are_blank_with_reason(self):
        _, result = self.export()
        b=Book(result['outputs'][0])
        for sheet in ['Known','Unknown']:
            h,rs=b.scan(sheet,1)
            col=next(k for k,v in h.items() if v=='Mass_difference_ppm')
            qc=next(k for k,v in h.items() if v=='Mass_QC')
            for rn,vs in rs[-2:]:
                self.assertEqual(vs.get(col,''),'')
                self.assertTrue(vs.get(qc))
        self.assertEqual(result['tables']['100']['ppm_calculated'],4)

    def test_csv_precision_and_original_preserved(self):
        before=sha256(self.path)
        folder,_=self.export()
        with (folder/'postprocess_audit.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(Decimal(rows[0]['PPM_full_precision']),0)
        self.assertEqual(Decimal(rows[1]['PPM_full_precision']),2)
        self.assertNotEqual(Decimal(rows[3]['PPM_full_precision']),0)
        self.assertEqual(rows[4]['PPM_full_precision'],'')
        self.assertEqual(sha256(self.path),before)

    def test_first_and_all_have_same_numeric_ppm(self):
        outputs=[]
        for mode in ['first','all']:
            _,r=self.export(mode)
            b=Book(r['outputs'][0]);h,rows=b.scan('Known',1)
            col=next(k for k,v in h.items() if v=='Mass_difference_ppm')
            outputs.append([v.get(col) for _,v in rows])
        self.assertEqual(outputs[0],outputs[1])

    def test_gui_uses_same_formatter(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'core/final_table_lab.py').read_text(encoding='utf-8')
        self.assertIn('format_ppm_display(p)',source)
        self.assertNotIn("'一致（输入精度内）' if p == 0",source)

    def test_workbook_notes_no_old_zero_claim(self):
        _,r=self.export()
        b=Book(r['outputs'][0]);_,rs=b.scan('Postprocess_Readme',1)
        note=next(v[2] for _,v in rs if v.get(1)=='Zero display')
        self.assertIn('0.000000',note)
        self.assertNotIn('formatted as 一致',note)

    def test_frozen_selftest_contract(self):
        self.assertIn('numeric zero',run_numeric_ppm_selftest())


if __name__ == '__main__':
    unittest.main()
