"""Display-only regression tests; fixtures contain synthetic images, not measurements."""
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw
from core.final_table_ooxml import Book, tag
from core.final_table_selftest import fixture
from core.final_table_postprocess import Options, autoconfig, prepare_context, execute, sha256
from core.final_table_display import ordered_entries, display_preview, render_xic, validate_display_mode


def make_image(path, rep):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (780, 220), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), 'SYNTHETIC - Rep %s - NOT EXPERIMENTAL' % rep, fill='black')
    draw.line([(25, 65), (25, 190), (740, 190)], fill='black', width=2)
    draw.line([(30, 185), (180, 185), (210, 160), (245, 90), (280, 160), (325, 185), (730, 185)], fill='black', width=2)
    image.save(path)
    return path


class DisplayChoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        header = ['Name', 'Formula', 'Exact_mass', 'Combo', 'Sheet']
        self.path = fixture(self.root / 'pair.xlsx', {
            'Known': [header, ['STD_0001', 'C2H6O', 46.0418648, 'shared-key', 'STD']],
            'Unknown': [header, ['LIB_0001', 'C2H6O', 46.0418648, 'shared-key', 'LIB']]})
        self.k = replace(autoconfig(self.path, 'Known'), image_dir=str(self.root/'xic'))
        self.u = replace(autoconfig(self.path, 'Unknown'), image_dir=str(self.root/'xic'))
        self.images = {}
        for group, name in [('STD', 'STD_0001'), ('LIB', 'LIB_0001')]:
            for rep in '123':
                p = self.root/'xic'/group/(group+'-'+rep+'__XIC_plots')/('001__'+name+'__mz46.04186.png')
                self.images[group, rep] = make_image(p, rep)

    def tearDown(self):
        self.tmp.cleanup()

    def matched(self):
        return prepare_context(self.k, self.u, Options())['100']['matches'][2]

    def audit_rows(self, folder):
        with (folder/'postprocess_audit.csv').open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    def test_default_is_first_and_replicate_matching_remains_on(self):
        self.assertEqual(Options().xic_display_mode, 'first')
        self.assertTrue(Options().triplicate_xic)
        self.assertEqual(len(self.matched()['paths']), 3)

    def test_first_mode_uses_rep1_not_file_discovery_order(self):
        match = self.matched()
        match['selected'].reverse()
        picture = render_xic(match, 520, 220, 'first')
        self.assertEqual([e['rep'] for e in match['used']], ['1'])
        self.assertEqual(match['status'], 'EMBEDDED_FIRST_REP1')
        self.assertIn('Rep 1 | STD-1', picture[3])
        self.assertEqual(Image.open(io.BytesIO(picture[0])).size, (1040, 440))
        self.assertEqual(len(match['selected']), 3)  # Undisplayed evidence not removed.

    def test_all_mode_preserves_triplicate_layout(self):
        match = self.matched()
        picture = render_xic(match, 520, 480, 'all')
        self.assertEqual([e['rep'] for e in match['used']], list('123'))
        self.assertEqual(match['status'], 'EMBEDDED_REPLICATES_3/3')
        self.assertEqual(picture[1:3], (520, 480))

    def test_missing_first_falls_back_to_rep2_and_labels(self):
        self.images['STD', '1'].unlink()
        match = self.matched()
        picture = render_xic(match, 520, 220, 'first')
        self.assertEqual(match['used'][0]['rep'], '2')
        self.assertIn('REP1_NOT_AVAILABLE', match['status'])
        self.assertIn('Rep 2 |', picture[3])

    def test_only_third_available_keeps_rep3_label(self):
        self.images['STD', '1'].unlink()
        self.images['STD', '2'].unlink()
        match = self.matched()
        render_xic(match, 480, 190, 'first')
        self.assertEqual(match['used'][0]['rep'], '3')
        self.assertIn('REP2_NOT_AVAILABLE', match['status'])

    def test_same_rep_conflict_not_first_ambiguous_file(self):
        p = self.images['STD', '1']
        make_image(p.parent/'002__STD_0001__mz46.04186.png', 'DIFFERENT')
        match = self.matched()
        self.assertIn('1', match['conflicts'])
        render_xic(match, 480, 190, 'first')
        self.assertEqual(match['used'][0]['rep'], '2')
        self.assertIn('REP1_CONFLICT_NOT_SELECTED', match['status'])

    def test_corrupt_rep1_falls_back_read_error_recorded(self):
        self.images['STD', '1'].write_bytes(b'corrupt-not-an-image')
        match = self.matched()
        render_xic(match, 480, 190, 'first')
        self.assertEqual(match['used'][0]['rep'], '2')
        self.assertIn('REP1_READ_FAILED', match['status'])
        self.assertEqual(len(match['read_errors']), 1)

    def test_all_mode_keeps_other_panels_on_read_failure(self):
        self.images['STD', '1'].write_bytes(b'corrupt-not-an-image')
        match = self.matched()
        render_xic(match, 520, 480, 'all')
        self.assertEqual([e['rep'] for e in match['used']], ['2', '3'])
        self.assertTrue(match['status'].startswith('EMBEDDED_REPLICATES_2/3'))

    def test_all_corrupt_records_no_embedded_source(self):
        for rep in '123':
            self.images['STD', rep].write_bytes(b'bad-image')
        folder, result = execute(self.k, self.u, self.root/'out')
        self.assertEqual(result['tables']['100']['images_embedded'], 0)
        row = self.audit_rows(folder)[0]
        self.assertEqual(row['Source_image_count'], '0')
        self.assertEqual(len(json.loads(row['Image_read_errors'])), 3)

    def test_conflicting_families_remain_unresolved(self):
        # Two copies of a full output tree are not a replicate choice.
        for rep in '123':
            make_image(self.root/'xic'/'backup'/'STD'/('STD-'+rep+'__XIC_plots')/
                       '001__STD_0001__mz46.04186.png', rep)
        match = self.matched()
        self.assertFalse(match['paths'])
        with self.assertRaises(ValueError):
            render_xic(match, 480, 190, 'first')
        self.assertEqual(display_preview(match, 'first'), 'No image can be displayed safely')

    def test_preview_distinguishes_matched_from_displayed(self):
        match = self.matched()
        self.assertIn('Rep 1', display_preview(match, 'first'))
        self.assertIn('3/3', display_preview(match, 'all'))
        self.assertEqual(len(match['selected']), 3)

    def test_both_tables_export_counts_and_mode_provenance(self):
        for mode, expected in [('first', 1), ('all', 3)]:
            folder, result = execute(self.k, self.u, self.root/'out',
                                     Options(xic_display_mode=mode, image_height=480 if mode=='all' else 220))
            self.assertEqual(result['options']['xic_display_mode'], mode)
            for label in ['100', '1000']:
                summary = result['tables'][label]
                self.assertEqual(summary['images_embedded'], 1)
                self.assertEqual(summary['source_images_embedded'], expected)
                self.assertEqual(summary['single_image_rows'], int(mode=='first'))
                self.assertEqual(summary['three_replicate_rows'], int(mode=='all'))
            for row in self.audit_rows(folder):
                self.assertEqual(row['XIC_display_mode'], mode)
                self.assertEqual(row['Matched_image_count'], '3')
                self.assertEqual(row['Source_image_count'], str(expected))
                self.assertTrue(row['Rep2_matched_image'])
                self.assertEqual(bool(row['Rep2_image']), mode=='all')
                self.assertEqual(row['Displayed_replicates'], '1' if mode=='first' else '1|2|3')
            self.assertEqual(result['tables']['100']['name_matches'], 1)

    def test_switching_display_does_not_change_mass_names_or_input(self):
        paths = [self.path]+list(self.images.values())
        hashes = {p: sha256(p) for p in paths}
        outrows = []
        for mode in ['first', 'all']:
            folder, result = execute(self.k, self.u, self.root/'out', Options(xic_display_mode=mode))
            rows = self.audit_rows(folder)
            keys = ['Table', 'Original_name', 'Theoretical_full_precision', 'PPM_full_precision',
                    'Mass_QC', 'Name_in_1000', 'Name_match_status', 'Name_match_method']
            outrows.append([{k: r[k] for k in keys} for r in rows])
        self.assertEqual(outrows[0], outrows[1])
        self.assertEqual(hashes, {p: sha256(p) for p in paths})

    def test_generic_single_image_still_works(self):
        root = self.root/'plain'/'STD'; make_image(root/'STD_0001.png', 'generic')
        cfg = replace(self.k, image_dir=str(root))
        match = prepare_context(cfg, self.u, Options())['100']['matches'][2]
        self.assertFalse(match['is_triplicate'])
        for mode in ['first', 'all']:
            picture = render_xic(match, 480, 190, mode)
            self.assertEqual(len(match['used']), 1)
            self.assertEqual(match['status'], 'EMBEDDED')
            self.assertTrue(picture[0])

    def test_exported_drawing_dimensions_and_labels_match_display_mode(self):
        from lxml import etree as E
        from core.final_table_ooxml import D
        for mode, height in [('first', 220), ('all', 480)]:
            folder, result = execute(self.k, self.u, self.root/'out',
                                     Options(xic_display_mode=mode, image_width=520, image_height=height))
            book = Book(result['outputs'][0])
            drawings = [E.fromstring(v) for n, v in book.entries.items()
                        if n.startswith('xl/drawings/pp_drawing') and n.endswith('.xml')]
            self.assertEqual(len(drawings), 2)
            for drawing in drawings:
                extent = drawing.find('.//{%s}ext' % D)
                self.assertEqual(int(extent.get('cy')), height*9525)
                description = drawing.find('.//{%s}cNvPr' % D).get('descr')
                self.assertIn('Rep 1' if mode=='first' else 'Replicate XICs', description)
                self.assertEqual(len(drawing), 1)

    def test_undisplayed_conflict_still_present_in_first_mode_audit(self):
        p = self.images['STD', '2']
        make_image(p.parent/'002__STD_0001__mz46.04186.png', 'different')
        folder, result = execute(self.k, self.u, self.root/'out')
        row = self.audit_rows(folder)[0]
        self.assertEqual(row['Displayed_replicates'], '1')
        self.assertTrue(row['Rep2_conflicts'])
        self.assertEqual(row['Rep2_image'], '')
        self.assertEqual(row['Source_image_count'], '1')

    def test_invalid_mode_rejected_before_output_created(self):
        out = self.root/'invalid'
        with self.assertRaises(ValueError):
            execute(self.k, self.u, out, Options(xic_display_mode='random'))
        self.assertFalse(out.exists())

    def test_old_strict_matching_option_still_available_in_api(self):
        ctx = prepare_context(self.k, self.u, Options(triplicate_xic=False))
        self.assertEqual(ctx['100']['matches'][2]['status'], 'AMBIGUOUS_IMAGES')
        self.assertEqual(display_preview(ctx['100']['matches'][2], 'first'), 'No image can be displayed safely')


if __name__ == '__main__':
    unittest.main()
