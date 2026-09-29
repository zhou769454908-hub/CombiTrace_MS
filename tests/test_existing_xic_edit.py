"""Synthetic workbook tests: editing existing drawings, not experiment data."""
import copy
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from lxml import etree as E
from PIL import Image, ImageDraw, ImageChops
from core.final_table_ooxml import Book, N, D, A, R, tag, _bytes, _xml, relfile, resolve
from core.final_table_selftest import fixture
from core.final_table_postprocess import Options, autoconfig, execute as create_result
from core.final_table_captions import CaptionSettings
from core.existing_xic_edit import (EditOptions, execute, file_hash, table_records, numbered_sheets,
                                   _assert_data_preserved, _from_old_audit, digest, SourceResolver, edit_book)


def pictures(book):
    result=[]
    for name,p in book.sheet_parts.items():
        root=book.sheet(name);de=root.find(tag('drawing'))
        if de is None:continue
        rels=_xml(book.part(relfile(p)));rel=next(x for x in rels if x.get('Id')==de.get('{%s}id'%R))
        dp=resolve(p,rel.get('Target'));dre=_xml(book.part(relfile(dp)));drawing=_xml(book.part(dp))
        for anchor in drawing:
            pic=anchor.find('{%s}pic'%D)
            if pic is None:continue
            nv=pic.find('{%s}nvPicPr/{%s}cNvPr'%(D,D))
            if not nv.get('name','').startswith('Postprocess XIC '):continue
            rid=pic.find('.//{%s}blip'%A).get('{%s}embed'%R)
            media=resolve(dp,next(x.get('Target') for x in dre if x.get('Id')==rid))
            result.append((name,media,book.part(media),nv,anchor))
    return result


class ExistingXICTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.src=self.root/'images'
        header=['Name','Formula','Exact_mass','Combo','Sheet','Concentration','Observed_mz','Mass_difference_ppm']
        self.input=fixture(self.root/'pair.xlsx',{
            'Known':[header,['SYNTHETIC_known','C2H6O',46.041865,'KEY','STD',0.123456,45.034645,-1.234567]],
            'Unknown':[header,['SYNTHETIC_unknown','C2H6O',46.041865,'KEY','LIB',0.987654,45.0347,2.5]],
            'Other':[['Original formula','Value'],['Untouched',{'f':'1+2','v':3}]]})
        for group,name in [('STD','SYNTHETIC_known'),('LIB','SYNTHETIC_unknown')]:
            for rep in '123':
                folder=self.src/(group+'-'+rep+'__XIC_plots');folder.mkdir(parents=True,exist_ok=True)
                image=Image.new('RGB',(300,120),'white');d=ImageDraw.Draw(image)
                d.rectangle((0,0,299,119),outline='black',width=2)
                d.text((10,10),'ORIGINAL SYNTHETIC '+rep,fill='black');d.line((10,100,110,20+int(rep),290,100),fill='black',width=2)
                image.save(folder/(name+'.png'))
        self.k=replace(autoconfig(self.input,'Known'),image_dir=str(self.src))
        self.u=replace(autoconfig(self.input,'Unknown'),image_dir=str(self.src))

    def tearDown(self):self.temp.cleanup()

    def create(self,mode='all'):
        _,r=create_result(self.k,self.u,self.root/'initial',Options(image_width=520,image_height=480 if mode=='all' else 220,
            xic_display_mode=mode,xic_caption_replicate=True,xic_caption_raw=True))
        return Path(r['outputs'][0])

    def edit(self,path,options=None):
        folder,r=execute([path],self.root/'edited',options)
        return Path(r['outputs'][0]),r

    def test_default_keep_layout_and_no_captions(self):
        o=EditOptions();self.assertEqual(o.mode,'keep');self.assertFalse(o.captions.enabled)

    def test_invalid_args_fail_before_output(self):
        with self.assertRaises(ValueError):execute([],self.root/'bad')
        with self.assertRaises(ValueError):execute([self.input],self.root/'bad',EditOptions(mode='x'))
        self.assertFalse((self.root/'bad').exists())

    def test_two_sheets_replace_without_new_data_columns(self):
        p=self.create();out,r=self.edit(p);self.assertEqual(r['updated'],2)
        a,b=Book(p),Book(out)
        for s in ('Known','Unknown'):self.assertEqual(a.scan(s)[0],b.scan(s)[0])
        self.assertEqual(len(pictures(a)),len(pictures(b)))
        self.assertEqual(len(pictures(b)),2)

    def test_raw_and_mass_functions_never_called(self):
        p=self.create()
        with patch('core.observed_mass.extract_table',side_effect=AssertionError('RAW called')),patch('core.final_table_postprocess.mass_result',side_effect=AssertionError('mass called')):
            out,r=self.edit(p)
        self.assertEqual(r['updated'],2)

    def test_measurement_cells_and_cached_formulas_unchanged(self):
        p=self.create();out,r=self.edit(p)
        _assert_data_preserved(Book(p),Book(out),r['events'])
        for s in ('Known','Unknown'):
            a,b=Book(p).grid(s)[2],Book(out).grid(s)[2]
            self.assertEqual([a[c] for c in range(1,9)],[b[c] for c in range(1,9)])

    def test_other_sheet_and_existing_audits_byte_identical(self):
        p=self.create();out,r=self.edit(p);a,b=Book(p),Book(out)
        for s in ('Other','Postprocess_Audit','Postprocess_Readme'):
            self.assertEqual(a.part(a.sheet_parts[s]),b.part(b.sheet_parts[s]))

    def test_input_not_overwritten(self):
        p=self.create();h=file_hash(p);out,r=self.edit(p)
        self.assertEqual(file_hash(p),h);self.assertNotEqual(p,out)
        self.assertTrue((out.parent/'COMPLETED.txt').exists())

    def test_preview_writes_no_files(self):
        p=self.create();folder,r=execute([p],self.root/'preview',preview=True)
        self.assertIsNone(folder);self.assertEqual(r['ready'],2);self.assertFalse((self.root/'preview').exists())

    def test_no_original_images_embedded_all_recovery(self):
        p=self.create();shutil.rmtree(self.src);out,r=self.edit(p)
        self.assertEqual(r['updated'],2)
        self.assertTrue(all('EMBEDDED_PANEL_RECOVERY' in e['Source_kinds'] for e in r['events']))
        self.assertTrue(all(e['Displayed_replicates']=='1|2|3' for e in r['events']))
        for e in r['events']:
            self.assertTrue(all(x['full_caption']=='' for x in json.loads(e['Caption_records'])))

    def test_no_original_single_recovery(self):
        p=self.create('first');shutil.rmtree(self.src);out,r=self.edit(p)
        self.assertEqual(r['updated'],2)
        self.assertTrue(all(e['Displayed_replicates']=='1' for e in r['events']))

    def test_all_to_first_to_all_without_originals(self):
        p=self.create();shutil.rmtree(self.src)
        a,r=self.edit(p,EditOptions(mode='first'))
        self.assertTrue(all(e['Displayed_replicates']=='1' for e in r['events']))
        b,t=self.edit(a,EditOptions(mode='all',captions=CaptionSettings(show_replicate=True)))
        self.assertEqual(t['updated'],2)
        self.assertTrue(all(e['Displayed_replicates']=='1|2|3' for e in t['events']))
        self.assertEqual(len(pictures(Book(b))),2)

    def test_first_to_all_does_not_invent_absent_replicas(self):
        p=self.create('first');shutil.rmtree(self.src);out,r=self.edit(p,EditOptions(mode='all'))
        self.assertTrue(all(e['Displayed_replicates']=='1' for e in r['events']))
        self.assertTrue(all('EMBEDDED_REPLICATES_1/3' in e['Reason'] for e in r['events']))

    def test_first_to_all_uses_originals_when_present(self):
        p=self.create('first');out,r=self.edit(p,EditOptions(mode='all'))
        self.assertTrue(all(e['Displayed_replicates']=='1|2|3' for e in r['events']))

    def test_keep_does_not_change_displayed_rep2_to_new_rep1(self):
        for d in self.src.glob('*-1__XIC_plots'):shutil.rmtree(d)
        p=self.create('first')
        for group,name in [('STD','SYNTHETIC_known'),('LIB','SYNTHETIC_unknown')]:
            d=self.src/(group+'-1__XIC_plots');d.mkdir();Image.new('RGB',(300,120),'white').save(d/(name+'.png'))
        out,r=self.edit(p)
        self.assertTrue(all(e['Displayed_replicates']=='2' for e in r['events']))

    def test_added_labels_are_opt_in(self):
        p=self.create();out,r=self.edit(p,EditOptions(captions=CaptionSettings(True,False,False,'Review')))
        for e in r['events']:
            values=json.loads(e['Caption_records'])
            self.assertEqual([x['full_caption'] for x in values],['Rep '+rep+' | Review' for rep in '123'])

    def test_repeated_caption_edits_do_not_accumulate_padding(self):
        p=self.create();shutil.rmtree(self.src);a,r=self.edit(p);b,t=self.edit(a)
        self.assertEqual([x[2] for x in pictures(Book(a))],[x[2] for x in pictures(Book(b))])

    def test_moved_original_folder_uniquely_relocated(self):
        p=self.create();dest=self.root/'moved';shutil.move(str(self.src),dest)
        out,r=self.edit(p,EditOptions(image_directory=str(dest),allow_embedded_recovery=False))
        self.assertEqual(r['updated'],2);self.assertTrue(all('ORIGINAL_IMAGE' in e['Source_kinds'] for e in r['events']))

    def test_missing_sources_keep_old_image(self):
        p=self.create();shutil.rmtree(self.src);out,r=self.edit(p,EditOptions(allow_embedded_recovery=False))
        self.assertEqual(r['updated'],0);self.assertEqual(r['unchanged'],2)
        self.assertEqual([x[2] for x in pictures(Book(p))],[x[2] for x in pictures(Book(out))])

    def test_changed_name_is_not_misassigned(self):
        p=self.create();book=Book(p);root=book.sheet('Known')
        root.find('.//s:c[@r="A2"]/s:is/s:t',N).text='other compound'
        book.changed[book.sheet_parts['Known']]=_bytes(root);p2=self.root/'changed.xlsx';book.save(p2)
        out,r=self.edit(p2);self.assertEqual(r['updated'],1)
        self.assertTrue(any('ROW_IDENTITY_CHANGED' in e['Reason'] for e in r['events']))

    def test_changed_formula_is_not_misassigned(self):
        p=self.create();book=Book(p);root=book.sheet('Known')
        root.find('.//s:c[@r="B2"]/s:is/s:t',N).text='C2H4O'
        book.changed[book.sheet_parts['Known']]=_bytes(root);p2=self.root/'changed.xlsx';book.save(p2)
        out,r=self.edit(p2);self.assertEqual(r['updated'],1)
        self.assertTrue(any('ROW_FORMULA_CHANGED' in e['Reason'] for e in r['events']))

    def test_protected_sheet_not_bypassed(self):
        p=self.create();book=Book(p);root=book.sheet('Known');E.SubElement(root,tag('sheetProtection'),sheet='1')
        book.changed[book.sheet_parts['Known']]=_bytes(root);p2=self.root/'protected.xlsx';book.save(p2)
        out,r=self.edit(p2);self.assertEqual(r['updated'],1)
        self.assertTrue(any('PROTECTED' in e['Reason'] for e in r['events']))

    def test_no_postprocess_images_reports_no_supported_xic(self):
        out,r=self.edit(self.input);self.assertEqual(r['updated'],0)
        self.assertTrue(any('NO_SUPPORTED_XIC' in e['Reason'] for e in r['events']))

    def test_keeps_size_by_default(self):
        p=self.create('first');out,r=self.edit(p)
        self.assertTrue(all(e['Width_px']==520 and e['Height_px']==220 for e in r['events']))

    def test_custom_size_updates_anchor_not_values(self):
        p=self.create('first');out,r=self.edit(p,EditOptions(keep_size=False,width=600,height=250))
        self.assertTrue(all(e['Width_px']==600 and e['Height_px']==250 for e in r['events']))
        _assert_data_preserved(Book(p),Book(out),r['events'])

    def test_two_separate_result_files_batch(self):
        p=self.create('first');q=self.root/'other.xlsx';shutil.copyfile(p,q)
        folder,r=execute([p,q],self.root/'batch');self.assertEqual(r['updated'],4);self.assertEqual(len(r['outputs']),2)

    def test_no_extra_cells_on_second_edit(self):
        p=self.create();a,r=self.edit(p);b,t=self.edit(a)
        self.assertEqual(Book(p).scan('Known')[0],Book(b).scan('Known')[0])
        self.assertEqual(len(pictures(Book(b))),2)
        self.assertEqual(len(numbered_sheets(Book(b),'XIC_Edit_Audit')),2)

    def test_embedded_crop_does_not_cut_original_title_or_border(self):
        p=self.create('first');b=Book(p);e=table_records(b,'Postprocess_Audit')[0];pic=pictures(b)[0]
        sources,mode,triple=_from_old_audit(b,e,'18.48.1',pic[1],520,220,pic[3].get('descr'))
        s=next(x for x in sources if x['was_displayed']);shutil.rmtree(self.src)
        data,kind,loc=SourceResolver(b,EditOptions()).get(s)
        im=Image.open(io.BytesIO(data)).convert('RGB')
        # Original 300x120 bordered image remains in full at native pixels.
        box=ImageChops.difference(im,Image.new('RGB',im.size,'white')).getbbox()
        self.assertEqual((box[2]-box[0],box[3]-box[1]),(300,120))
        self.assertEqual(im.getpixel((box[0],box[1])),(0,0,0))
        self.assertEqual(im.getpixel((box[2]-1,box[3]-1)),(0,0,0))

    def test_relocated_ambiguous_sources_never_first_file(self):
        p=self.create('first');dest=self.root/'ambiguous';
        for branch in ('a','b'):shutil.copytree(self.src,dest/branch)
        shutil.rmtree(self.src)
        out,r=self.edit(p,EditOptions(image_directory=str(dest),allow_embedded_recovery=False))
        self.assertEqual(r['updated'],0)

    def test_legacy_generated_header_bounds(self):
        p=self.create('first');b=Book(p);e=table_records(b,'Postprocess_Audit')[0];pic=pictures(b)[0]
        e.pop('XIC_caption_records',None)
        s,m,t=_from_old_audit(b,e,'18.48',pic[1],520,220,pic[3].get('descr'))
        self.assertEqual(next(x for x in s if x['was_displayed'])['embedded']['bounds'],[10,30,1030,432])

    def test_unknown_version_does_not_crop(self):
        p=self.create('first');b=Book(p);e=table_records(b,'Postprocess_Audit')[0];pic=pictures(b)[0]
        s,m,t=_from_old_audit(b,e,'unknown',pic[1],520,220,pic[3].get('descr'))
        self.assertFalse(any('embedded' in x for x in s))

    def test_latest_xic_column_only_default(self):
        p=self.create('first')
        k=replace(autoconfig(p,'Known'),image_dir=str(self.src))
        u=replace(autoconfig(p,'Unknown'),image_dir=str(self.src))
        _,rr=create_result(k,u,self.root/'again',Options(image_width=520,image_height=220))
        twice=Path(rr['outputs'][0]);out,r=self.edit(twice)
        self.assertEqual(r['updated'],2)
        self.assertEqual(len(pictures(Book(out))),4)
        out2,r2=self.edit(twice,EditOptions(latest_column_only=False))
        self.assertEqual(r2['updated'],4)

    def test_custom_graphics_not_replaced(self):
        p=self.create('first');b=Book(p)
        pic=pictures(b)[0];root=b.sheet('Known');sp=b.sheet_parts['Known']
        rels=_xml(b.part(relfile(sp)));dp=resolve(sp,next(x.get('Target') for x in rels if x.get('Id')==root.find(tag('drawing')).get('{%s}id'%R)))
        dr=_xml(b.part(dp));anchor=copy.deepcopy(dr[0])
        nv=anchor.find('.//{%s}cNvPr'%D);nv.set('id','999');nv.set('name','KEEP LOGO')
        anchor.find('{%s}from/{%s}row'%(D,D)).text='0'
        dr.append(anchor);b.changed[dp]=_bytes(dr);p2=self.root/'logo.xlsx';b.save(p2)
        before=E.tostring(anchor,method='c14n');out,r=self.edit(p2)
        after=_xml(Book(out).part(dp))[-1]
        self.assertEqual(before,E.tostring(after,method='c14n'))

    def test_frozen_smoke_contract(self):
        from core.existing_xic_edit import run_existing_xic_selftest
        self.assertIn('no RAW/model calls',run_existing_xic_selftest())

    def test_overwritten_source_image_does_not_change_existing_trace(self):
        p=self.create('first');_,baseline=self.edit(p)
        original_media=[x[2] for x in pictures(Book(baseline['outputs'][0]))]
        for path in self.src.rglob('*.png'):Image.new('RGB',(300,120),'black').save(path)
        out,r=self.edit(p)
        self.assertEqual(original_media,[x[2] for x in pictures(Book(out))])
        self.assertTrue(all('EMBEDDED_PANEL_RECOVERY' in e['Source_kinds'] for e in r['events']))

    def test_xlsm_extension_preserved(self):
        p=self.create('first');q=self.root/'macro.xlsm';shutil.copyfile(p,q)
        out,r=self.edit(q);self.assertEqual(out.suffix,'.xlsm')

if __name__=='__main__':unittest.main()
