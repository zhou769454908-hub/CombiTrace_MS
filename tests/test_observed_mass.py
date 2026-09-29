"""Measurement provenance and postprocessor regression using explicitly synthetic scans."""
import csv
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from core.observed_mass import (
    pick_centroid, canonical_adduct, event_matches, neutral_from_mz, measure_request,
    row_requests, extract_table, mass_cells, FisherSource, RawBackendUnavailable,
)
from core.final_table_postprocess import (
    Options, TableConfig, autoconfig, theoretical_value, execute, prepare_context, mass_result, sha256,
)
from core.final_table_selftest import fixture
from core.observed_mass_selftest import SyntheticSource, run_observed_mass_selftest
from core.final_table_ooxml import Book
from core.final_table_triplicates import hn, SourceCatalog


class SelectionTests(unittest.TestCase):
    def test_signed_measurement(self):
        self.assertEqual(pick_centroid([100.0012], [100], 100, 20)['mz'],100.0012)
        self.assertEqual(pick_centroid([99.9998], [100], 100, 20)['mz'],99.9998)
    def test_no_peak_not_theory(self):
        self.assertIsNone(pick_centroid([101], [100], 100, 20)['mz'])
    def test_nonfinite_zero_negative_intensity_excluded(self):
        self.assertIsNone(pick_centroid([100,np.nan,100], [0,100,-1],100,20)['mz'])
    def test_ambiguous_not_average(self):
        r=pick_centroid([99.999,100.001],[100,70],100,20)
        self.assertEqual(r['status'],'AMBIGUOUS_CENTROIDS'); self.assertIsNone(r['mz'])
    def test_not_nearest_to_theory(self):
        r=pick_centroid([100.000001,100.0015],[19,100],100,20)
        self.assertEqual(r['mz'],100.0015)
    def test_preserve_more_than_six_decimals(self):
        mz=356.123456789012
        self.assertEqual(pick_centroid([mz],[123],mz,20)['mz'],mz)
    def test_invalid_shape(self):
        self.assertEqual(pick_centroid([100,101],[1],100,20)['status'],'INVALID_CENTROID_ARRAYS')
    def test_ms1_polarity(self):
        self.assertTrue(event_matches('FTMS - p ESI Full ms [100-900]','[M-H]-'))
        for ev in ['FTMS + p ESI Full ms [100-900]','FTMS - p ESI Full ms2 200@hcd','FTMS - p ESI SIM [100-900]']:
            self.assertFalse(event_matches(ev,'[M-H]-'))
    def test_neutral_conversion_two_charges(self):
        f,a='C10H15NO2','[M-2H]2-'; t=theoretical_value(f,True,a); m=theoretical_value(f)
        self.assertEqual(neutral_from_mz(f,a,t+Decimal('0.001')),m+Decimal('0.002'))
    def test_neutral_conversion_dimer(self):
        f,a='C10H15NO2','[2M-H]-'; t=theoretical_value(f,True,a); m=theoretical_value(f)
        self.assertEqual(neutral_from_mz(f,a,t+Decimal('0.002')),m+Decimal('0.001'))
    def test_adduct_combination_rejected(self):
        with self.assertRaises(ValueError): canonical_adduct('[M-H]- + optional [M+HCOO]-')
    def test_adduct_alias(self): self.assertEqual(canonical_adduct('M-H'),'[M-H]-')


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.q={'apex_rt':1.,'rt_start':.9,'rt_end':1.1,'adduct':'[M-H]-','theory':100.,'status':'READY'}
        self.options=Options()
        self.s=SimpleNamespace(index=[(1,.99),(2,1.),(3,1.01)], version='synthetic',
            event=lambda s:'FTMS - p ESI Full ms [10-200]',
            spectrum=lambda s:([100.0012],[1000], 'SYNTHETIC'))
    def test_outside_qc_retained(self):
        r=measure_request(self.s,self.q,self.options)
        self.assertEqual(r['observed_mz'],100.0012); self.assertIn('OUTSIDE_PPM_LIMIT',r['status'])
    def test_apex_by_rt_not_mass_error(self):
        self.s.spectrum=lambda s:([100.0012 if s==2 else 100.0],[1000], 'SYNTHETIC')
        r=measure_request(self.s,self.q,self.options)
        self.assertEqual(r['scan'],2); self.assertEqual(r['observed_mz'],100.0012)
    def test_no_apex_no_neighbour_substitution(self):
        self.s.spectrum=lambda s:([101 if s==2 else 100.0],[1000], 'SYNTHETIC')
        r=measure_request(self.s,self.q,self.options)
        self.assertIsNone(r['observed_mz']); self.assertEqual(r['status'],'APEX_NO_PEAK_IN_SEARCH_WINDOW')
    def test_no_full_run_rt_fallback(self):
        self.q['apex_rt']=15.
        r=measure_request(self.s,self.q,self.options)
        self.assertIsNone(r['observed_mz'])
    def test_profile_error_not_fake_measurement(self):
        def fail(s): raise ValueError('NO_CENTROID_STREAM_PROFILE_NOT_CONVERTED')
        self.s.spectrum=fail
        self.assertIsNone(measure_request(self.s,self.q,self.options)['observed_mz'])
    def test_neighbour_not_used_as_primary(self):
        r=measure_request(self.s,self.q,self.options)
        self.assertEqual(r['support_scan_count'],3); self.assertEqual(r['scan'],2)


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.p=Path(self.tmp.name)
        self.raw=self.p/'raw'; self.raw.mkdir()
        for r in '123': (self.raw/('sample-'+r+'.raw')).write_text('SYNTHETIC_STUB_NOT_A_REAL_RAW')
        self.headers=['Name','Formula','Exact_mass','RAW','Apex_RT','Found','Combo']
        self.data=['A','C10H15NO2',str(theoretical_value('C10H15NO2')),'sample-1.raw',1.,True,'KEY_A']
        path=fixture(self.p/'input.xlsx', {'Known':[self.headers,self.data],
            'Unknown':[self.headers,['B']+self.data[1:]]})
        self.k=replace(autoconfig(path,'Known'), mass_mode='raw_mz', raw_dir=str(self.raw))
        self.u=replace(autoconfig(path,'Unknown'), mass_mode='raw_mz', raw_dir=str(self.raw))
    def tearDown(self): self.tmp.cleanup()
    def context(self): return prepare_context(self.k,self.u,Options())
    def test_end_to_end(self):
        r=run_observed_mass_selftest(); self.assertEqual(r['observed_rows'],2)
    def test_preview_never_uses_exactmass_as_observation(self):
        v=self.context()['100']; r=mass_result(v['rows'][0],self.k,v['headers'],5)
        self.assertIsNone(r['ppm']); self.assertIsNone(r['observed'])
    def test_missing_raw_keeps_blank(self):
        (self.raw/'sample-1.raw').unlink()
        r=extract_table(self.context()['100'],Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('RAW_FILE_NOT_FOUND',r['status'])
    def test_ambiguous_raw_no_first(self):
        d=self.raw/'other'; d.mkdir(); (d/'sample-1.RAW').write_text('duplicate name')
        r=extract_table(self.context()['100'],Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('AMBIGUOUS_RAW_FILENAME',r['status'])
    def test_source_found_false(self):
        v=self.context()['100']; v['rows'][0]['values'][6]=False
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('SOURCE_PEAK_NOT_FOUND',r['status'])
    def test_missing_rt_not_inferred_from_table_order(self):
        v=self.context()['100']; v['rows'][0]['values'][5]=''
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('MISSING_PER_RAW_APEX_RT',r['status'])
    def test_source_formula_conflict(self):
        v=self.context()['100']; v['rows'][0]['_source_status']='METADATA_FORMULA_CONFLICT'
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('UNRESOLVED_SOURCE_IDENTITY',r['status'])
    def test_primary_first_replicate_not_smallest_ppm(self):
        v=self.context()['100']; row=v['rows'][0]
        row['values']={1:'A',2:'C10H15NO2',7:'KEY_A'}
        row['_source_records']=[]
        for r in '123':
            si={hn(k):val for k,val in {'RAW':'sample-'+r+'.raw','Apex_RT':1.,'Found':'True'}.items()}
            row['_source_records'].append({'_source_info':si,'_source_trace':'x'+r+'__XIC_quant.csv'})
        results=extract_table(v,Options(),reader_factory=SyntheticSource)
        self.assertEqual(results[2]['chosen']['rep'],'1')
        m,c=mass_cells(results[2]); self.assertAlmostEqual(float(c['Rep3_Mass_ppm']['value']),8.,places=6)
    def test_raw2_fallback_recorded(self):
        v=self.context()['100']; v['rows'][0]['values'][4]='sample-2.raw'
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIn('MASS_REPLICATE_FALLBACK',r['status']); self.assertEqual(r['chosen']['rep'],'2')
    def test_two_rows_same_centroid_rejected(self):
        v=self.context()['100']; row=dict(v['rows'][0]); row['row']=3; row['name']='Another'; v['rows'].append(row)
        r=extract_table(v,Options(),reader_factory=SyntheticSource)
        self.assertIsNone(r[2]['observed']); self.assertIn('CENTROID_CLAIMED',r[2]['status'])
    def test_input_values_unchanged_and_export_audit(self):
        before=sha256(self.k.path)
        folder,r=execute(self.k,self.u,self.p/'out',Options(),reader_factory=SyntheticSource)
        self.assertEqual(sha256(self.k.path),before)
        b=Book(r['outputs'][0]); h,rows=b.scan('Known'); vals={h[c]:v for c,v in rows[0][1].items()}
        self.assertAlmostEqual(float(vals['Exact_mass']),float(self.data[2]))
        self.assertIn('Observed_Mass_Audit',b.sheet_parts)
        self.assertAlmostEqual(float(vals['Mass_difference_ppm']),1.25,places=6)
    def test_no_theory_fill_when_no_measurements(self):
        class NoneSource(SyntheticSource):
            def spectrum(self,sn): return [999],[1000],'SYNTHETIC'
        folder,r=execute(self.k,self.u,self.p/'out',Options(),reader_factory=NoneSource)
        b=Book(r['outputs'][0]); h,rows=b.scan('Known'); vals={h[c]:v for c,v in rows[0][1].items()}
        self.assertIn(vals.get('Observed_mz'),(None,'')); self.assertIn(vals.get('Mass_difference_ppm'),(None,''))
    def test_windows_backend_missing_raises_not_completed(self):
        def fail(p): raise RawBackendUnavailable('SYNTHETIC missing bridge')
        with self.assertRaises(RawBackendUnavailable): execute(self.k,self.u,self.p/'out',Options(),reader_factory=fail)
        self.assertFalse(list((self.p/'out').rglob('COMPLETED.txt')))
    def test_mz_mode_independent_of_image_display(self):
        a=extract_table(self.context()['100'],Options(xic_display_mode='first'),reader_factory=SyntheticSource)[2]
        b=extract_table(self.context()['100'],Options(xic_display_mode='all'),reader_factory=SyntheticSource)[2]
        self.assertEqual(a['observed'],b['observed'])
    def test_wrong_source_adduct_not_reused(self):
        v=self.context()['100']; row=v['rows'][0]; row['values']={1:'A',2:'C10H15NO2',7:'KEY_A'}
        row['_source_records']=[{'_source_trace':'x__XIC_quant.csv','_source_info':{hn(k):v for k,v in {'RAW':'sample-1.raw','Apex_RT':1,'Found':'True','Matched_Adduct':'[M+HCOO]-'}.items()}}]
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('SOURCE_ION_MISMATCH',r['status'])
    def test_wide_triplicate_no_shared_mean_rt(self):
        v=self.context()['100']; row=v['rows'][0]; row['values']={1:'A',2:'C10H15NO2',7:'KEY_A'}
        info={hn(k):v for k,v in {'Rep1_RAW':'sample-1.raw','Rep1_Apex_RT':1,'Rep1_Found':'True',
              'Rep2_RAW':'sample-2.raw','Apex_RT_mean(found)':1}.items()}
        row['_source_records']=[{'_source_trace':'summary','_source_info':info}]
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNotNone(r['observed'])
        self.assertIn('MISSING_PER_RAW_APEX_RT',[q['status'] for q in r['measurements']])
    def test_qc_window_not_larger_than_search(self):
        with self.assertRaises(ValueError): extract_table(self.context()['100'],Options(observed_search_ppm=2),reader_factory=SyntheticSource)


class MetadataIntegrationTests(unittest.TestCase):
    setUp = PostprocessTests.setUp
    tearDown = PostprocessTests.tearDown
    context = PostprocessTests.context
    def test_original_quant_csv_triplicates(self):
        export=self.p/'xic'; export.mkdir()
        headers=['Name','Formula','Combo','Found','Apex_RT','RT_start','RT_end','Matched_Adduct','MH_Found','MH_Apex_RT']
        for rep in '123':
            with (export/('sample-'+rep+'__XIC_quant.csv')).open('w',newline='') as f:
                w=csv.writer(f); w.writerow(headers)
                w.writerow(['Original_A','C10H15NO2','KEY_A',True,1.,.9,1.1,'[M-H]-',True,1.])
        # Final names differ. Combo links them without positional matching.
        k=replace(self.k,image_dir=str(export)); u=replace(self.u,image_dir=str(export))
        v=prepare_context(k,u,Options())['100']
        v['rows'][0]['values']={1:'Final_A',2:'C10H15NO2',7:'KEY_A'}
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertEqual(len(r['measurements']),3)
        self.assertEqual([q['rep'] for q in r['measurements']],['1','2','3'])
        self.assertIsNotNone(r['observed'])
    def test_conflicting_same_raw_quant_records_rejected(self):
        v=self.context()['100']; row=v['rows'][0]; row['values']={1:'A',2:'C10H15NO2',7:'KEY_A'}
        row['_source_records']=[{'_source_trace':x+'__XIC_quant.csv','_source_info':{hn('RAW'):'sample-1.raw',hn('Apex_RT'):rt,hn('Found'):'True'}} for x,rt in [('a',1.),('b',2.)]]
        r=extract_table(v,Options(),reader_factory=SyntheticSource)[2]
        self.assertIsNone(r['observed']); self.assertIn('CONFLICTING_PEAK_METADATA',r['status'])
    def test_zero_observed_shift_not_jittered(self):
        f='C10H15NO2'; t=theoretical_value(f,True,'[M-H]-')
        meta={'observed':t,'theory':t,'neutral':theoretical_value(f),'adduct':'[M-H]-',
              'chosen':{'formula':f,'raw_name':'synthetic','rep':'1','observed_mz':float(t)},
              'measurements':[],'status':'OBSERVED_RAW_MS1'}
        m,_=mass_cells(meta)
        self.assertEqual(m['ppm'],Decimal(0))

class NativeAdapterContractTests(unittest.TestCase):
    def backend(self, centroid=True, profile=False):
        from types import ModuleType
        module=ModuleType('fisher_py')
        class Access:
            in_acquisition=False
            disposed=False
            def get_scan_event_string_for_scan_number(self,sn): return 'FTMS - p ESI Full ms [50-500]'
            def get_centroid_stream(self,sn,noise):
                return SimpleNamespace(masses=[200.123456789] if centroid else [],intensities=[100] if centroid else [])
            def get_scan_stats_for_scan_number(self,sn): return SimpleNamespace(is_centroid_scan=not profile)
            def get_segmented_scan_from_scan_number(self,sn,stats): return SimpleNamespace(positions=[200.123456789],intensities=[100])
            def dispose(self): self.disposed=True
        class Raw:
            def __init__(self,path):
                self._raw_file_access=Access(); self._ms1_scan_numbers=[1]; self._ms1_retention_times=[1.]
        module.RawFile=Raw
        return module
    def test_native_centroid_preserved_and_disposed(self):
        with patch.dict('sys.modules',{'fisher_py':self.backend()}):
            source=FisherSource('SYNTHETIC_ONLY'); a=source.access
            mz,ints,kind=source.spectrum(1)
            self.assertEqual(mz[0],200.123456789); self.assertEqual(kind,'RAW_NATIVE_CENTROID_STREAM')
            source.close(); self.assertTrue(a.disposed)
    def test_profile_not_treated_as_centroid(self):
        with patch.dict('sys.modules',{'fisher_py':self.backend(False,True)}):
            source=FisherSource('SYNTHETIC_ONLY')
            with self.assertRaises(ValueError): source.spectrum(1)
            source.close()
    def test_confirmed_centroid_segment_fallback(self):
        with patch.dict('sys.modules',{'fisher_py':self.backend(False,False)}):
            source=FisherSource('SYNTHETIC_ONLY')
            self.assertEqual(source.spectrum(1)[2],'RAW_CONFIRMED_CENTROID_SEGMENT'); source.close()


if __name__=="__main__": unittest.main()
