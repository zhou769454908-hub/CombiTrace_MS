"""Presentation-only contract: captions are opt-in, source and mass data unchanged."""
import copy
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageChops, ImageDraw
from core.final_table_captions import (CaptionSettings, build_caption, clean_text, from_options,
                                       draw_caption, fit_text, load_caption_font)
from core.final_table_display import render_xic
from core.final_table_postprocess import Options, autoconfig, prepare_context, execute, sha256
from core.final_table_ooxml import Book
from core.final_table_selftest import fixture


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.entry = {'rep': '1', 'raw': 'RAW-private-1', 'path': self.root/'raw.png'}
        Image.new('RGB', (900, 250), 'white').save(self.entry['path'])

    def tearDown(self):
        self.tmp.cleanup()

    def result(self, triplicate=False):
        entries=[]
        for rep in ('123' if triplicate else '1'):
            path=self.root/('raw-'+rep+'.png'); Image.new('RGB',(900,250),'white').save(path)
            entries.append(dict(self.entry, rep=rep, raw='RAW-private-'+rep, path=path))
        return {'paths': [x['path'] for x in entries], 'selected': entries,
                'conflicts': {}, 'is_triplicate': triplicate}

    def test_default_has_no_text_fields(self):
        self.assertEqual(from_options(Options()), CaptionSettings())
        self.assertFalse(CaptionSettings().enabled)
        self.assertEqual(CaptionSettings().field_names(), [])
        self.assertEqual(build_caption(self.entry, compound_name='秘密样品'), '')

    def test_replicate_only_does_not_include_any_name(self):
        self.assertEqual(build_caption(self.entry, CaptionSettings(show_replicate=True), 'compound'), 'Rep 1')

    def test_raw_only(self):
        self.assertEqual(build_caption(self.entry, CaptionSettings(show_raw=True), 'compound'), 'RAW-private-1')

    def test_name_only_is_current_table_name(self):
        self.assertEqual(build_caption(self.entry, CaptionSettings(show_name=True), '当前表名称'), '当前表名称')

    def test_custom_only_is_literal_not_template(self):
        text = "{raw} =1+1 <xml>"
        self.assertEqual(build_caption(self.entry, CaptionSettings(custom_text=text), 'other'), text)

    def test_all_fields_in_fixed_order(self):
        settings=CaptionSettings(True, True, True, 'Review')
        self.assertEqual(build_caption(self.entry, settings, 'Cpd'), 'Rep 1 | RAW-private-1 | Cpd | Review')
        self.assertEqual(settings.field_names(), ['replicate','raw','compound_name','custom_text'])

    def test_unknown_rep_does_not_invent_one(self):
        self.assertEqual(build_caption({'raw':'','rep':''}, CaptionSettings(True, True)), '')

    def test_no_raw_fallback_from_image_filename(self):
        self.assertEqual(build_caption({'path':'Private.png','rep':'2'}, CaptionSettings(show_raw=True)), '')

    def test_clean_controls_and_whitespace(self):
        self.assertEqual(clean_text('  α\x00\n名字\t '), 'α 名字')

    def test_custom_length_rejected(self):
        with self.assertRaises(ValueError): CaptionSettings(custom_text='a'*201).validate()
        CaptionSettings(custom_text='a'*200).validate()

    def test_invalid_field_types_rejected(self):
        with self.assertRaises(ValueError): CaptionSettings(show_raw='false').validate()
        with self.assertRaises(ValueError): CaptionSettings(custom_text=123).validate()

    def test_default_single_blank_source_stays_blank_pixels(self):
        result=self.result()
        pic=render_xic(result,520,220,'first')
        im=Image.open(io.BytesIO(pic[0])).convert('RGB')
        self.assertIsNone(ImageChops.difference(im,Image.new('RGB',im.size,'white')).getbbox())
        self.assertEqual(result['caption_records'][0]['header_pixels'],0)
        self.assertIn('RAW-private-1', pic[3])  # provenance stays, visible names do not

    def test_default_triple_has_no_headers_in_any_slot(self):
        result=self.result(True);pic=render_xic(result,520,480,'all')
        im=Image.open(io.BytesIO(pic[0])).convert('RGB')
        for top,bottom in ((0,319),(320,639),(640,960)):
            crop=im.crop((0,top,1040,bottom))
            self.assertIsNone(ImageChops.difference(crop,Image.new('RGB',crop.size,'white')).getbbox())
        self.assertTrue(all(x['full_caption']=='' and x['header_pixels']==0 for x in result['caption_records']))

    def test_enabled_label_changes_only_presentation(self):
        a=self.result(True);b=copy.deepcopy(a)
        off=render_xic(a,520,220,'first');on=render_xic(b,520,220,'first', CaptionSettings(show_replicate=True))
        self.assertNotEqual(off[0],on[0]);self.assertEqual(a['used'],b['used'])
        self.assertEqual(a['status'],b['status']);self.assertEqual(off[3],on[3])
        self.assertEqual(b['caption_records'][0]['full_caption'],'Rep 1')
        self.assertNotIn('RAW-private',b['caption_records'][0]['visible_caption'])

    def test_all_three_use_correct_per_replicate_labels(self):
        result=self.result(True)
        render_xic(result,520,480,'all',CaptionSettings(True,True,True),compound_name='CPD')
        self.assertEqual([x['full_caption'] for x in result['caption_records']],
                         ['Rep '+r+' | RAW-private-'+r+' | CPD' for r in '123'])

    def test_first_fallback_label_reflects_actual_replicate(self):
        result=self.result(True);result['selected'][0]['path'].write_bytes(b'bad')
        render_xic(result,520,220,'first',CaptionSettings(True,False))
        self.assertEqual(result['used'][0]['rep'],'2')
        self.assertEqual(result['caption_records'][0]['full_caption'],'Rep 2')
        self.assertIn('REP1_READ_FAILED',result['status'])

    def test_missing_slot_stays_marked_even_with_no_names(self):
        result=self.result(True);result['selected']=result['selected'][1:];result['paths']=result['paths'][1:]
        pic=render_xic(result,520,480,'all')
        im=Image.open(io.BytesIO(pic[0])).convert('RGB');crop=im.crop((0,0,1040,319))
        self.assertIsNotNone(ImageChops.difference(crop,Image.new('RGB',crop.size,'white')).getbbox())
        self.assertEqual(result['caption_records'][0]['full_caption'],'')
        self.assertEqual([x['rep'] for x in result['used']],['2','3'])

    def test_no_header_space_is_reserved_when_disabled(self):
        canvas=Image.new('RGB',(1040,440),'white')
        off,d0=draw_caption(canvas,(0,0,1040,440),self.entry)
        on,d1=draw_caption(canvas,(0,0,1040,440),self.entry,CaptionSettings(True))
        self.assertLess(off[1],on[1]);self.assertEqual(d0['header_pixels'],0)
        self.assertGreater(d1['header_pixels'],0)

    def test_long_text_fits_and_audit_retains_full_text(self):
        canvas=Image.new('RGB',(240,440),'white')
        setting=CaptionSettings(True,True,True,'EN'*80)
        area, detail=draw_caption(canvas,(0,0,240,440),self.entry,setting,'Name'*40)
        self.assertGreater(area[3]-area[1],0)
        self.assertTrue(detail['caption_truncated'])
        self.assertIn('EN'*80,detail['full_caption'])
        font=load_caption_font(20);draw=ImageDraw.Draw(canvas)
        for line in detail['visible_caption'].split('\n'):
            box=draw.textbbox((0,0),line,font=font)
            self.assertLessEqual(box[2]-box[0],220)

    def test_no_cropping_original_content(self):
        result=self.result()
        path=result['selected'][0]['path']
        image=Image.new('RGB',(100,40),'white');d=ImageDraw.Draw(image)
        d.rectangle((0,0,99,39),outline='black',width=2);d.text((6,10),'ORIGINAL',fill='black');image.save(path)
        before=sha256(path)
        for settings in (CaptionSettings(), CaptionSettings(True,True)):
            pic=render_xic(result,520,220,'first',settings)
            composed=Image.open(io.BytesIO(pic[0])).convert('RGB')
            # Four black border corners survive at natural resolution (thumbnail never upsizes).
            box=ImageChops.difference(composed,Image.new('RGB',composed.size,'white')).getbbox()
            self.assertIsNotNone(box)
            blank=Image.new('RGB',(1040,440),'white')
            area,_=draw_caption(blank,(0,0,1040,440),result['selected'][0],settings)
            x0,y0,x1,y1=area
            left=x0+(x1-x0-100)//2;top=y0+(y1-y0-40)//2
            self.assertIsNone(ImageChops.difference(composed.crop((left,top,left+100,top+40)),image).getbbox())
        self.assertEqual(before,sha256(path))

    def test_generic_single_custom_caption_supported(self):
        result=self.result();result['selected'][0]['rep']='';result['selected'][0]['raw']=''
        render_xic(result,520,220,'all',CaptionSettings(custom_text='Review'))
        self.assertEqual(result['caption_records'][0]['full_caption'],'Review')
        self.assertEqual(result['status'],'EMBEDDED')

    def test_invalid_setting_rejected_before_output_created(self):
        from core.final_table_postprocess import TableConfig
        output=self.root/'invalid'
        with self.assertRaises(ValueError):
            execute(TableConfig('missing.xlsx','Known'),TableConfig('other.xlsx','Unknown'),output,
                    Options(xic_caption_custom='x'*201))
        self.assertFalse(output.exists())


class CaptionExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        header=['Name','Formula','Exact_mass','Combo','Sheet','Concentration']
        self.path=fixture(self.root/'pair.xlsx',{
            'Known':[header,['Known-private','C2H6O',46.041865,'KEY','STD',0.123]],
            'Unknown':[header,['Target-private','C2H6O',46.041865,'KEY','LIB',0.456]]})
        self.k=replace(autoconfig(self.path,'Known'),image_dir=str(self.root/'images'))
        self.u=replace(autoconfig(self.path,'Unknown'),image_dir=str(self.root/'images'))
        for group,name in [('STD','Known-private'),('LIB','Target-private')]:
            for rep in '123':
                p=self.root/'images'/group/(group+'-'+rep+'__XIC_plots')/('001__'+name+'__mz46.04186.png')
                p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(200,80),'white').save(p)

    def tearDown(self):self.tmp.cleanup()

    def rows(self,folder):
        with (folder/'postprocess_audit.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

    def test_export_defaults_suppresses_headers_both_tables(self):
        folder,summary=execute(self.k,self.u,self.root/'out')
        for row in self.rows(folder):
            self.assertEqual(row['XIC_caption_fields'],'')
            self.assertEqual(json.loads(row['XIC_caption_records'])[0]['visible_caption'],'')
            self.assertTrue(row['Rep2_matched_image'])
            self.assertEqual(row['Source_image_count'],'1')
        self.assertFalse(summary['options']['xic_caption_raw'])

    def test_each_table_uses_its_own_name_not_cross_linked_name(self):
        folder,summary=execute(self.k,self.u,self.root/'out',Options(xic_caption_name=True))
        for row in self.rows(folder):
            self.assertEqual(json.loads(row['XIC_caption_records'])[0]['full_caption'],row['Original_name'])
        self.assertEqual(self.rows(folder)[0]['Name_in_1000'],'Target-private')

    def test_mode_and_caption_fields_exported_to_summary_and_readme(self):
        opts=Options(xic_display_mode='all',xic_caption_replicate=True,xic_caption_custom='仅用于内部核对')
        folder,summary=execute(self.k,self.u,self.root/'out',opts)
        self.assertTrue(summary['options']['xic_caption_replicate'])
        for row in self.rows(folder):
            labels=json.loads(row['XIC_caption_records'])
            self.assertEqual(len(labels),3)
            self.assertEqual(row['XIC_caption_fields'],'replicate|custom_text')
            self.assertTrue(all('private' not in x['visible_caption'] for x in labels))
        book=Book(summary['outputs'][0]);_,notes=book.scan('Postprocess_Readme',1)
        self.assertTrue(any('XIC added caption' in v.values() for _,v in notes))

    def test_switching_captions_does_not_change_mass_names_or_original_files(self):
        files=[self.path]+list((self.root/'images').rglob('*.png'))
        before={str(p):sha256(p) for p in files};results=[]
        for mode in ('first','all'):
            for settings in ({},{'xic_caption_raw':True,'xic_caption_name':True,'xic_caption_custom':'Report'}):
                folder,_=execute(self.k,self.u,self.root/'out',Options(xic_display_mode=mode,**settings))
                keys=('Table','Original_name','Theoretical_full_precision','PPM_full_precision','Mass_QC',
                      'Name_in_1000','Name_match_method','Matched_image_files','Source_metadata_trace')
                results.append([{k:r[k] for k in keys} for r in self.rows(folder)])
        self.assertTrue(all(x==results[0] for x in results))
        self.assertEqual(before,{str(p):sha256(p) for p in files})

    def test_custom_text_cannot_be_excel_formula(self):
        text='=1+1'
        folder,summary=execute(self.k,self.u,self.root/'out',Options(xic_caption_custom=text))
        book=Book(summary['outputs'][0]);headers,rows=book.scan('Postprocess_Audit',1)
        column=next(c for c,n in headers.items() if n=='XIC_caption_custom')
        self.assertEqual(rows[0][1][column],text)
        self.assertEqual(self.rows(folder)[0]['XIC_caption_custom'],text)


if __name__=='__main__':unittest.main()
