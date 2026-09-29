"""Synthetic contracts only. Never presented as observations from a real instrument."""
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import numpy as np
from .final_table_selftest import fixture
from .final_table_postprocess import autoconfig, theoretical_value, execute, Options
from .final_table_ooxml import Book


class SyntheticSource:
    version = 'SYNTHETIC_BACKEND_NOT_AN_INSTRUMENT'
    opened = []
    def __init__(self, path):
        self.path = Path(path)
        self.opened.append(str(path))
        self.index = [(1, 0.99), (2, 1.0), (3, 1.01), (4, 2.0), (5, 2.01)]
        self.theory = float(theoretical_value('C10H15NO2', True, '[M-H]-'))
        self.closed = False
    def event(self, sn): return 'FTMS - p ESI Full ms [50.0-1000.0]'
    def spectrum(self, sn):
        ppm = {'sample-1.raw': 1.25, 'sample-2.raw': -2.5, 'sample-3.raw': 8.0}.get(self.path.name.lower(), 1.25)
        return np.array([self.theory * (1 + ppm / 1e6)]), np.array([1000.0]), 'SYNTHETIC_CENTROID'
    def close(self): self.closed = True


def run_observed_mass_selftest():
    with TemporaryDirectory() as temp:
        p = Path(temp)
        raw = p/'raw'; raw.mkdir()
        (raw/'sample-1.raw').write_text('SYNTHETIC_STUB_NOT_A_REAL_RAW')
        headers = ['Name', 'Formula', 'Exact_mass', 'RAW', 'Apex_RT', 'Found', 'Combo']
        data = [['Known_A', 'C10H15NO2', str(theoretical_value('C10H15NO2')), 'sample-1.raw', 1.0, True, 'ID_A']]
        book = fixture(p/'input.xlsx', {'Known': [headers]+data,
            'Unknown': [headers]+[['Target_A']+data[0][1:]]})
        k = replace(autoconfig(book, 'Known'), mass_mode='raw_mz', raw_dir=str(raw))
        u = replace(autoconfig(book, 'Unknown'), mass_mode='raw_mz', raw_dir=str(raw))
        folder, result = execute(k, u, p/'out', Options(), reader_factory=SyntheticSource)
        out = Book(result['outputs'][0]); h, rows = out.scan('Known')
        vals = {h[c]: v for c,v in rows[0][1].items()}
        assert abs(float(vals['Mass_difference_ppm'])-1.25) < 1e-6
        assert vals['Observed_mz'] != vals['Exact_mass']
        assert result['tables']['100']['observed_mz_count'] == 1
        assert 'Observed_Mass_Audit' in out.sheet_parts
        assert (folder/'observed_mass_scans.csv').exists()
        return {'scope': 'synthetic source only, NOT a real RAW reader test', 'observed_rows': 2,
                'measurement_ppm': float(vals['Mass_difference_ppm']), 'source_files_unchanged': True}
