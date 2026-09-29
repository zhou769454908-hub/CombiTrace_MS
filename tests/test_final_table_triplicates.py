"""Synthetic exporter-contract tests. NO real patient/sample/chemical measurements."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from PIL import Image
from lxml import etree as E
from core.final_table_ooxml import Book, tag, D
from core.final_table_selftest import fixture
from core.final_table_postprocess import (Options, autoconfig, load_rows, match_names, Images, prepare_context, execute, sha256)
from core.final_table_triplicates import (SourceCatalog, split_raw, parts_from_combo, abc_formula_token, compose_triplicates, resolve_claims)


def png(p, value=250):
    p.parent.mkdir(parents=True, exist_ok=True)
    im=Image.new('RGB', (900, 270), (value, value, value))
    from PIL import ImageDraw
    d=ImageDraw.Draw(im); d.text((20,20), 'SYNTHETIC XIC - NOT EXPERIMENTAL', fill='black')
    d.line((30,70,30,240,850,240), fill='black', width=2)
    d.line([(40,238),(190,232),(225,210),(245,90),(267,208),(300,236),(820,236)],fill='black',width=2)
    im.save(p)
    return p


def combo(indices=(1,2,3), labels=('A-reagent','B-reagent','C-reagent'), forms=('C4H7N','C8H7N3O','C6H7NO')):
    return ' | '.join('%s#%s:%s:%s' % (r,n,l,f) for r,n,l,f in zip('ABC', indices, labels, forms))


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return path


class TriplicateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.form='C18H20N3O3P'
        self.headers=['Name','Formula','Exact_mass','Combo','Sheet']
        self.path=fixture(self.root/'final.xlsx', {
            'Known':[self.headers,['STD_0001',self.form,357.12423,combo(),'STD']],
            'Unknown':[self.headers,['LIB_0017',self.form,357.12423,combo((5,7,9)),'LIB']]})
        self.k=autoconfig(self.path,'Known');self.u=autoconfig(self.path,'Unknown',label='1000')

    def tearDown(self):self.tmp.cleanup()
    def row(self,cfg=None):
        cfg=cfg or self.k
        return load_rows(Book(cfg.path),cfg)[1][0]
    def plots(self, root=None, name='STD_0001', group='STD', reps='123', value=250):
        root=root or self.root/'export'
        return [png(root/group/(group+'-'+r+'__XIC_plots')/('001__'+name+'__mz356.11695.png'),value) for r in reps]

    def test_actual_exporter_triple_is_not_duplicate_conflict(self):
        ps=self.plots(); cfg=replace(self.k,image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options())
        m=c['100']['matches'][2]
        self.assertEqual(m['status'],'MATCHED_TRIPLICATES');self.assertEqual(m['paths'],ps)

    def test_partial_reps_preserve_labels(self):
        self.plots(reps='13');cfg=replace(self.k,image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options());m=c['100']['matches'][2]
        self.assertEqual([x['rep'] for x in m['selected']],['1','3'])
        result=compose_triplicates(m,520,480)
        self.assertEqual(result[1:3],(520,480));self.assertEqual(len(m['used']),2)

    def test_strict_mode_remains_available(self):
        self.plots();cfg=replace(self.k,image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options(triplicate_xic=False))
        self.assertEqual(c['100']['matches'][2]['status'],'AMBIGUOUS_IMAGES')

    def test_date_prefixed_raws_group(self):
        root=self.root/'export'
        for r in '123':png(root/'STD'/('20250101-STD-'+r+'__XIC_plots')/'001__STD_0001__mz356.11695.png')
        c=prepare_context(replace(self.k,image_dir=str(root)),self.u,Options())
        self.assertEqual(len(c['100']['matches'][2]['paths']),3)

    def test_multi_sample_group_not_merged(self):
        self.plots(group='S1');self.plots(group='S2')
        m=Images(self.root/'export').find(self.row(),self.k)
        self.assertEqual(m['status'],'AMBIGUOUS_IMAGES')

    def test_backup_run_not_merged_even_if_images_identical(self):
        self.plots(root=self.root/'export'/'old');self.plots(root=self.root/'export'/'new')
        c=prepare_context(replace(self.k,image_dir=str(self.root/'export')),self.u,Options())
        self.assertEqual(c['100']['matches'][2]['status'],'AMBIGUOUS_IMAGES')

    def test_same_replicate_conflict_keeps_other_two(self):
        self.plots();png(self.root/'export'/'STD'/'STD-2__XIC_plots'/'002__STD_0001__mz400.00000.png',120)
        c=prepare_context(replace(self.k,image_dir=str(self.root/'export')),self.u,Options())
        m=c['100']['matches'][2];self.assertEqual([x['rep'] for x in m['selected']],['1','3']);self.assertIn('2',m['conflicts'])

    def test_same_rep_identical_copy_dedup_only_same_namespace(self):
        ps=self.plots();(ps[1].parent/'002__STD_0001__mz356.11695.png').write_bytes(ps[1].read_bytes())
        c=prepare_context(replace(self.k,image_dir=str(self.root/'export')),self.u,Options())
        self.assertEqual(c['100']['matches'][2]['status'],'MATCHED_TRIPLICATES')

    def test_raw_columns_disambiguate(self):
        self.plots(group='S1');self.plots(group='S2')
        p=fixture(self.root/'raws.xlsx',{'Known':[['Name','Formula','Rep1_RAW','Rep2_RAW','Rep3_RAW'],
                ['STD_0001',self.form,'S2-1.raw','S2-2.raw','S2-3.raw']]})
        cfg=replace(autoconfig(p,'Known'),image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options());m=c['100']['matches'][2]
        self.assertEqual(len(m['paths']),3);self.assertTrue(all('/S2/' in str(p) for p in m['paths']))

    def test_renumbered_components_match(self):
        b=Book(self.path);hk,kr=load_rows(b,self.k);hu,ur=load_rows(b,self.u)
        m=match_names(kr,ur,hk,hu,self.k,self.u,Options())[2]
        self.assertEqual(m['name'],'LIB_0017');self.assertEqual(m['method'],'ABC_reagent_labels_and_formulas_ignore_local_indices')

    def test_different_labels_unique_abc_formula_is_review(self):
        b=Book(self.path);hk,kr=load_rows(b,self.k);hu,ur=load_rows(b,self.u)
        ur[0]['values'][4]=combo(labels=('X','Y','Z'))
        m=match_names(kr,ur,hk,hu,self.k,self.u,Options())[2]
        self.assertEqual(m['status'],'MATCHED_REVIEW_ABC_FORMULAS')
        self.assertEqual(match_names(kr,ur,hk,hu,self.k,self.u,Options(allow_abc_formula=False))[2]['status'],'NOT_MATCHED')

    def test_abc_formula_ambiguity_is_not_guessed(self):
        b=Book(self.path);hk,kr=load_rows(b,self.k);hu,ur=load_rows(b,self.u)
        ur[0]['values'][4]=combo(labels=('X','Y','Z'))
        ur.append(dict(ur[0], name='Other',row=3))
        self.assertEqual(match_names(kr,ur,hk,hu,self.k,self.u,Options())[2]['status'],'AMBIGUOUS')

    def test_component_order_preserved(self):
        self.assertNotEqual(parts_from_combo(combo())['A'][1],parts_from_combo(combo())['B'][1])
        b=Book(self.path);hk,kr=load_rows(b,self.k);hu,ur=load_rows(b,self.u)
        ur[0]['values'][4]=combo(labels=('X','Y','Z'),forms=('C8H7N3O','C4H7N','C6H7NO'))
        self.assertEqual(match_names(kr,ur,hk,hu,self.k,self.u,Options())[2]['status'],'NOT_MATCHED')

    def test_component_formula_key_format(self):
        x=abc_formula_token('abc_formula|a=C=4,H=7,N=1|b=C=8,H=7,N=3,O=1|c=C=6,H=7,N=1,O=1')
        self.assertEqual(x,tuple(parts_from_combo(combo())[r][1] for r in 'ABC'))

    def test_final_formula_conflict_blocks_renumber_match(self):
        b=Book(self.path);hk,kr=load_rows(b,self.k);hu,ur=load_rows(b,self.u);ur[0]['formula']='C2H6'
        self.assertEqual(match_names(kr,ur,hk,hu,self.k,self.u,Options())[2]['status'],'FORMULA_CONFLICT')

    def test_renamed_final_table_recovers_original_image_from_enumerated(self):
        ps=self.plots()
        write_csv(self.root/'export'/'STD'/'STD__enumerated.csv',[dict(name='STD_0001',formula=self.form,combo=combo(),sheet='STD',raw_1='STD-1.raw',raw_2='STD-2.raw',raw_3='STD-3.raw')])
        p=fixture(self.root/'renamed.xlsx',{'Known':[self.headers,['NEW_NAME',self.form,357.12423,combo(),'STD']]})
        cfg=replace(autoconfig(p,'Known'),image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options())
        self.assertEqual(c['100']['rows'][0]['_source_names'],['STD_0001'])
        self.assertEqual(c['100']['matches'][2]['paths'],ps)
        self.assertEqual(c['100']['rows'][0]['name'],'NEW_NAME')

    def test_moved_xic_quant_absolute_path_and_replicates(self):
        ps=self.plots()
        for i,p in enumerate(ps,1):
            write_csv(p.parent.parent/('STD-%s__XIC_quant.csv'%i),[dict(Name='STD_0001',Formula=self.form,Combo=combo(),XIC_PNG='Z:\\old\\STD-%s__XIC_plots\\%s'%(i,p.name))])
        cfg=replace(self.k,image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options());self.assertEqual(len(c['100']['matches'][2]['paths']),3)

    def test_legacy_x_replacement_metadata_mapping(self):
        root=self.root/'export'
        for i in '123':
            p=png(root/'1-12-x'/('1-12-'+i+'__XIC_plots')/'001__1-12-x_0001__mz356.11695.png')
            write_csv(p.parent.parent/('1-12-'+i+'__XIC_quant.csv'),[dict(Name='1-12-x_0001',Formula=self.form,Combo=combo())])
        write_csv(root/'1-12-x'/'1-12-x__enumerated.csv',[dict(name='1-12-x_0001',formula=self.form,combo=combo(),sheet='1-12-x',raw_1='1-12-1.raw',raw_2='1-12-2.raw',raw_3='1-12-3.raw')])
        p=fixture(self.root/'x.xlsx',{'Known':[self.headers,['1-12-x_0001',self.form,357.12423,combo(),'1-12-x']]})
        c=prepare_context(replace(autoconfig(p,'Known'),image_dir=str(root)),self.u,Options())
        self.assertEqual(len(c['100']['matches'][2]['paths']),3)

    def test_custom_suffix_via_enumerated_raw_columns(self):
        root=self.root/'export'
        for r in 'abc':png(root/'STD'/('STD-'+r+'__XIC_plots')/'001__STD_0001__mz356.11695.png')
        write_csv(root/'STD'/'STD__enumerated.csv',[dict(name='STD_0001',formula=self.form,combo=combo(),sheet='STD',raw_1='STD-a.raw',raw_2='STD-b.raw',raw_3='STD-c.raw')])
        c=prepare_context(replace(self.k,image_dir=str(root)),self.u,Options())
        self.assertEqual([x['rep'] for x in c['100']['matches'][2]['selected']],['1','2','3'])

    def test_formula_metadata_mismatch_does_not_embed(self):
        ps=self.plots()
        for i,p in enumerate(ps,1):
            write_csv(p.parent.parent/('STD-%s__XIC_quant.csv'%i),[dict(Name='STD_0001',Formula='C2H6',Combo=combo())])
        c=prepare_context(replace(self.k,image_dir=str(self.root/'export')),self.u,Options())
        self.assertEqual(c['100']['matches'][2]['status'],'IMAGE_METADATA_FORMULA_CONFLICT')

    def test_source_workbook_alias_and_unchanged_hash(self):
        ps=self.plots()
        src=fixture(self.root/'source.xlsx',{'Triplicate Average':[self.headers,['STD_0001',self.form,357.12423,combo(),'STD']]})
        p=fixture(self.root/'renamed.xlsx',{'Known':[self.headers,['RENAMED',self.form,357.12423,combo(),'STD']]})
        cfg=replace(autoconfig(p,'Known'),image_dir=str(self.root/'export'),metadata_path=str(src))
        before=sha256(src);folder,result=execute(cfg,self.u,self.root/'out',Options(image_height=480, xic_display_mode='all'))
        self.assertEqual(sha256(src),before);self.assertEqual(result['tables']['100']['source_images_embedded'],3)

    def test_hidden_summary_used_without_path(self):
        self.plots()
        p=fixture(self.root/'hidden.xlsx',{'Known':[self.headers,['RENAMED',self.form,357.12423,combo(),'STD']],
                                        'Triplicate Average':[self.headers,['STD_0001',self.form,357.12423,combo(),'STD']]})
        cfg=replace(autoconfig(p,'Known'),image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options());self.assertEqual(len(c['100']['matches'][2]['paths']),3)

    def test_same_image_cannot_be_claimed_by_two_rows(self):
        self.plots();cfg=replace(self.k,image_dir=str(self.root/'export'))
        row=self.row();matcher=Images(self.root/'export')
        matches=resolve_claims({2:matcher.find(row,cfg),3:matcher.find(dict(row,row=3),cfg)})
        self.assertEqual(matches[2]['status'],'IMAGE_CLAIMED_BY_MULTIPLE_ROWS');self.assertFalse(matches[3]['paths'])

    def test_export_counts_rows_vs_images_and_no_unknown_name_zero(self):
        self.plots();self.plots(name='LIB_0017',group='LIB')
        kc=replace(self.k,image_dir=str(self.root/'export'));uc=replace(self.u,image_dir=str(self.root/'export'))
        before=sha256(self.path);folder,r=execute(kc,uc,self.root/'out',Options(image_height=480, xic_display_mode='all'))
        self.assertEqual(sha256(self.path),before)
        self.assertEqual(r['tables']['100']['images_embedded'],1);self.assertEqual(r['tables']['100']['source_images_embedded'],3)
        self.assertEqual(r['tables']['100']['name_matches'],1);self.assertFalse(r['tables']['1000']['name_matching_applicable'])
        b=Book(r['outputs'][0]);d=[E.fromstring(v) for n,v in b.entries.items() if n.startswith('xl/drawings/pp_drawing') and n.endswith('.xml')]
        self.assertEqual(sum(len(x) for x in d),2)
        self.assertTrue((folder/'postprocess_diagnostics.txt').exists());self.assertTrue((folder/'xic_file_index.csv').exists())

    def test_plain_name_suffix_not_treated_as_replicate(self):
        p=png(self.root/'images'/'Compound-1.png')
        row=dict(self.row(),name='Compound')
        self.assertEqual(Images(p.parent).find(row,self.k)['status'],'NOT_FOUND')

    def test_explicit_rep_file_suffix_supported(self):
        for i in '123':png(self.root/'images'/('STD_0001__Rep'+i+'.png'))
        self.assertEqual(len(Images(self.root/'images').find(self.row(),self.k)['paths']),3)

    def test_corrupt_panel_is_not_counted(self):
        ps=self.plots();ps[1].write_bytes(b'not png')
        c=prepare_context(replace(self.k,image_dir=str(self.root/'export')),self.u,Options())
        m=c['100']['matches'][2];compose_triplicates(m,520,480)
        self.assertEqual(len(m['used']),2);self.assertEqual(m['read_errors'][0][0],'2')

    def test_only_formula_without_identity_not_fabricated(self):
        b=Book(self.path);hk,kr=load_rows(b,self.k);hu,ur=load_rows(b,self.u)
        kr[0]['values'][4]='';ur[0]['values'][4]=''
        self.assertEqual(match_names(kr,ur,hk,hu,self.k,self.u,Options())[2]['status'],'NOT_MATCHED')

    def test_source_paths_disambiguate_sanitized_names(self):
        root=self.root/'export'
        for r in '123':
            d=root/'STD'/('STD-'+r+'__XIC_plots')
            pa=png(d/'001__item__mz356.11695.png',240)
            pb=png(d/'002__item__mz356.11695.png',220)
            write_csv(d.parent/('STD-'+r+'__XIC_quant.csv'),[
                dict(Name='甲',Formula=self.form,Combo=combo(),XIC_PNG='Z:\\old\\'+d.name+'\\'+pa.name),
                dict(Name='乙',Formula=self.form,Combo=combo(labels=('X','Y','Z')),XIC_PNG='Z:\\old\\'+d.name+'\\'+pb.name)])
        p=fixture(self.root/'cjk.xlsx',{'Known':[self.headers,['甲',self.form,357.12423,combo(),'STD']]})
        c=prepare_context(replace(autoconfig(p,'Known'),image_dir=str(root)),self.u,Options())
        m=c['100']['matches'][2]
        self.assertEqual(len(m['paths']),3);self.assertTrue(all(p.name.startswith('001__') for p in m['paths']))
        self.assertIn('source_XIC_PNG',m['method'])

    def test_metadata_path_prefers_selected_image_tree(self):
        ps=self.plots()
        stale=self.root/'stale'/'STD-1__XIC_plots'/ps[0].name;png(stale,10)
        matcher=Images(self.root/'export')
        self.assertEqual(matcher._path_matches(str(stale),prefer_root=True),[ps[0]])
        self.assertEqual(matcher._path_matches(str(stale)),[stale])

    def test_group_mismatch_is_reported_separately(self):
        self.plots(group='Wrong')
        cfg=replace(self.k,image_dir=str(self.root/'export'))
        c=prepare_context(cfg,self.u,Options())
        self.assertEqual(c['100']['matches'][2]['status'],'SOURCE_GROUP_MISMATCH')

    def test_metadata_can_be_disabled(self):
        self.plots()
        write_csv(self.root/'export'/'STD'/'STD__enumerated.csv',[dict(name='STD_0001',formula=self.form,combo=combo(),sheet='STD')])
        p=fixture(self.root/'rename.xlsx',{'Known':[self.headers,['NEW',self.form,357.12423,combo(),'STD']]})
        c=prepare_context(replace(autoconfig(p,'Known'),image_dir=str(self.root/'export')),self.u,Options(use_source_index=False))
        self.assertEqual(c['100']['matches'][2]['status'],'NOT_FOUND')

if __name__=='__main__':unittest.main()
