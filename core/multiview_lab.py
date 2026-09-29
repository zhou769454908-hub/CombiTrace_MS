"""Small independent Tkinter window; heavy modelling runs off the UI thread."""
from __future__ import annotations
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class MultiviewLab(tk.Toplevel):
    def __init__(self, master, cache_path='', training_path='', output_parent=''):
        super().__init__(master)
        self.title('CombiTrace-MS | Multi-view response models')
        self.geometry('1080x790')
        self.minsize(850, 650)
        self.events = queue.Queue()
        self.running = False
        self.last_folder = None
        self.column_maps = {'training': {}, 'target': {}}
        self.vars = {name: tk.StringVar(value=value) for name, value in {
            'cache': cache_path, 'training': training_path, 'target': '',
            'training_sheet': '', 'target_sheet': '', 'seed': '42', 'folds': '5',
            'unit': 'Same unit as the calibration table',
            'output': output_parent or (str(Path(cache_path).parent / 'v18_46_runs') if cache_path
                       else str(Path.cwd() / 'v18_46_runs')),
        }.items()}
        body = ttk.Frame(self, padding=14)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        intro = ('Multi-view response models using existing descriptors. Compare structural, observed-ion and component information with training-only model selection. Inputs and previous outputs remain unchanged.')
        ttk.Label(body, text=intro, wraplength=960).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 12))
        fields = [('cache', 'ESI descriptor workbook (.xlsx; required)'),
                  ('training', 'Current calibration table (optional)'),
                  ('target', 'Current target table (optional)'),
                  ('output', 'Output directory')]
        for row, (name, label) in enumerate(fields, start=1):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=5)
            ttk.Entry(body, textvariable=self.vars[name]).grid(row=row, column=1, sticky='ew', pady=5)
            ttk.Button(body, text='Browse...', command=lambda key=name: self.browse(key)).grid(row=row, column=2, padx=8)
        row = 5
        for kind, label in [('training', 'Calibration table'), ('target', 'Target table')]:
            frame = ttk.Frame(body)
            frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)
            ttk.Label(frame, text=label + 'Worksheet (blank for CSV):').pack(side='left')
            ttk.Entry(frame, textvariable=self.vars[kind + '_sheet'], width=25).pack(side='left', padx=5)
            ttk.Button(frame, text='View / set column mapping...', command=lambda k=kind: self.choose_columns(k)).pack(side='left', padx=8)
            row += 1
        settings = ttk.Frame(body)
        settings.grid(row=7, column=0, columnspan=3, sticky='ew', pady=8)
        for key, label, width in [('folds', 'Outer folds', 5), ('seed', 'Random seed', 8), ('unit', 'Concentration unit', 30)]:
            ttk.Label(settings, text=label).pack(side='left', padx=(0, 5))
            ttk.Entry(settings, textvariable=self.vars[key], width=width).pack(side='left', padx=(0, 16))
        self.compare_legacy = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, variable=self.compare_legacy,
            text='Run the continuous-audit control on the same folds').pack(side='left', padx=5)
        self.confirm = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, variable=self.confirm,
            text='Calibration and target Area/IS values use the same scale (fraction or percentage), and known concentrations use consistent units.').grid(
            row=8, column=0, columnspan=3, sticky='w', pady=5)
        ttk.Label(body, text=(
            'Without a current calibration table, cached ESI_Features rows and values are used. Selecting a current table makes its row list authoritative; descriptors are matched by Combo, with RT and identity checks. Historical residual-based exclusions are not carried over automatically. Use a reviewed calibration table.'),
            wraplength=940).grid(row=9, column=0, columnspan=3, sticky='w', pady=8)
        controls = ttk.Frame(body)
        controls.grid(row=10, column=0, columnspan=3, sticky='ew', pady=8)
        self.run_button = ttk.Button(controls, text='Run multi-view comparison', command=self.run)
        self.run_button.pack(side='left')
        ttk.Button(controls, text='Open results', command=self.open_folder).pack(side='left', padx=10)
        self.status = ttk.Label(controls, text='Ready')
        self.status.pack(side='left', padx=12)
        self.log = tk.Text(body, height=12, wrap='word')
        self.log.grid(row=11, column=0, columnspan=3, sticky='nsew', pady=8)
        body.rowconfigure(11, weight=1)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self._poll_after = self.after(100, self.poll)

    def destroy(self):
        pending = getattr(self, '_poll_after', None)
        if pending:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
            self._poll_after = None
        super().destroy()

    def browse(self, key):
        if key == 'output':
            selected = filedialog.askdirectory(parent=self)
        else:
            selected = filedialog.askopenfilename(parent=self, filetypes=[
                ('Excel / CSV', '*.xlsx *.xlsm *.csv'), ('All files', '*.*')])
        if selected:
            self.vars[key].set(selected)
            if key in self.column_maps:
                self.column_maps[key] = {}

    def choose_columns(self, kind):
        path = self.vars[kind].get().strip()
        if not path:
            messagebox.showinfo('Column mapping', 'Select an input table first.', parent=self)
            return
        try:
            from .response_predictor import read_table, guess_columns
            from .continuous_io import pick_header
            table = read_table(Path(path), sheet_name=self.vars[kind + '_sheet'].get().strip())
            cfg = guess_columns(table, need_concentration=(kind == 'training'))
            guessed = {'combo': cfg.combo_col, 'formula': cfg.formula_col, 'name': cfg.name_col,
                       'ratio': pick_header(table.headers, ['Measured_ratio']) or cfg.ratio_col,
                       'concentration': cfg.concentration_col,
                       'group': pick_header(table.headers, ['Injection_Group', 'Injection group']),
                       'rt': pick_header(table.headers, ['Apex_RT_min', 'Apex_RT', 'RT'])}
        except Exception as exc:
            messagebox.showerror('Could not read headers', str(exc), parent=self)
            return
        dialog = tk.Toplevel(self)
        dialog.title('Map input columns; do not use predicted concentration as a training label')
        dialog.transient(self)
        dialog.columnconfigure(1, weight=1)
        fields = [('combo', 'Combo (required)'), ('formula', 'Formula'), ('name', 'Compound name'),
                  ('ratio', 'Measured Area/IS ratio (required)'), ('group', 'Injection group'), ('rt', 'RT (cache consistency check)')]
        if kind == 'training':
            fields.insert(4, ('concentration', 'Known concentration (required)'))
        variables = {}
        for row, (key, label) in enumerate(fields):
            ttk.Label(dialog, text=label).grid(row=row, column=0, sticky='w', padx=10, pady=6)
            variable = tk.StringVar(value=self.column_maps[kind].get(key, guessed.get(key, '')))
            variables[key] = variable
            ttk.Combobox(dialog, textvariable=variable, values=[''] + table.headers,
                         width=48, state='readonly').grid(row=row, column=1, padx=10, pady=6)

        def save():
            self.column_maps[kind] = {key: var.get() for key, var in variables.items()}
            dialog.destroy()
        ttk.Button(dialog, text='Save mapping', command=save).grid(row=len(fields), column=0, columnspan=2, pady=12)

    def run(self):
        if self.running:
            return
        if not self.confirm.get():
            messagebox.showwarning('Confirm input scales', 'Check the ratio scale and concentration unit, then confirm.', parent=self)
            return
        values = {key: var.get().strip() for key, var in self.vars.items()}
        try:
            if not Path(values['cache']).is_file():
                raise ValueError('Select a descriptor workbook containing ESI_Features.')
            if not values['output']:
                raise ValueError('Select an output directory.')
            seed, folds = int(values['seed']), int(values['folds'])
            if not 3 <= folds <= 10:
                raise ValueError('Outer folds must be 3-10; the available independent groups may limit the actual count.')
            for key in ('training', 'target'):
                if values[key] and not Path(values[key]).is_file():
                    raise ValueError('Input table not found: ' + values[key])
        except Exception as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)
            return
        maps = {key: dict(value) for key, value in self.column_maps.items()}
        compare_legacy = bool(self.compare_legacy.get())
        self.running = True
        self.run_button.configure(state='disabled')
        self.status.configure(text='Running')

        def worker():
            try:
                from .multiview_io import execute
                folder, result = execute(values['cache'], values['output'],
                    training_path=values['training'], target_path=values['target'],
                    training_sheet=values['training_sheet'], target_sheet=values['target_sheet'],
                    training_columns=maps['training'], target_columns=maps['target'],
                    concentration_unit=values['unit'], seed=seed, folds=folds, compare_legacy=compare_legacy,
                    progress=lambda s: self.events.put(('log', s)))
                self.events.put(('done', (folder, result['summary']['Validation_status'])))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log':
                    self.log.insert('end', str(value) + '\n')
                    self.log.see('end')
                else:
                    self.running = False
                    self.run_button.configure(state='normal')
                    if kind == 'done':
                        self.last_folder, status = value
                        self.status.configure(text=status)
                        messagebox.showinfo('Run complete', 'Validation status: ' + status + '\n\n' + str(self.last_folder), parent=self)
                    else:
                        self.status.configure(text='Failed; input files unchanged')
                        self.log.insert('end', 'ERROR: ' + value + '\n')
                        messagebox.showerror('Run incomplete', value, parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self._poll_after = self.after(150, self.poll)

    def open_folder(self):
        if not self.last_folder:
            messagebox.showinfo('Results', 'No completed run is available in this window.', parent=self)
            return
        if sys.platform == 'win32':
            os.startfile(str(self.last_folder))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(self.last_folder)])
        else:
            subprocess.Popen(['xdg-open', str(self.last_folder)])

    def close(self):
        if self.running:
            messagebox.showinfo('Running', 'Keep this window open until processing finishes. Progress is shown below.', parent=self)
            return
        self.destroy()


def main():
    root = tk.Tk()
    root.withdraw()
    window = MultiviewLab(root)
    def close():
        if window.running:
            window.close()
        else:
            root.destroy()
    window.protocol('WM_DELETE_WINDOW', close)
    root.mainloop()
