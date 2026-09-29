"""Final-table postprocess tab and stand-alone Tkinter entry."""
from __future__ import annotations
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .final_table_display import DISPLAY_LABELS, display_preview
from .final_table_postprocess import format_ppm_display

MODE_LABELS = {
    'Read observed m/z from RAW': 'raw_mz',
    'Automatic (Exact mass treated as theoretical)': 'auto',
    'Observed neutral mass (source verified)': 'observed_neutral',
    'Observed m/z (use the specified ion)': 'observed_mz',
    'Theoretical neutral mass: numerical comparison only': 'reference_neutral',
}
COLUMN_FIELDS = [('name_col', 'Name column'), ('formula_col', 'Neutral formula column'),
                 ('mass_col', 'Exact mass / measured mass column'), ('key_col', 'Shared identity key'),
                 ('image_col', 'Image path column (optional)'), ('group_col', 'XIC subfolder group column (optional)'),
                 ('adduct_col', 'Ion column (m/z mode)')]


class InputPanel(ttk.LabelFrame):
    def __init__(self, parent, label):
        super().__init__(parent, text=label + (' table (actual row count is used)' if label == '100' else ' table (all original rows retained)'), padding=8)
        self.label = label
        self.columnconfigure(1, weight=1)
        self.vars = {k: tk.StringVar(value='') for k in ['path', 'sheet', 'image_dir', 'metadata_path', 'raw_dir', 'raw_col', 'rt_col'] + [x[0] for x in COLUMN_FIELDS]}
        self.vars['header_row'] = tk.StringVar(value='1')
        self.vars['mass_mode'] = tk.StringVar(value=next(iter(MODE_LABELS)))
        self.vars['fixed_adduct'] = tk.StringVar(value='[M-H]-')
        ttk.Label(self, text='Input result workbook').grid(row=0, column=0, sticky='w')
        ttk.Entry(self, textvariable=self.vars['path'], width=28).grid(row=0, column=1, sticky='ew', padx=4)
        ttk.Button(self, text='Browse', command=self.browse).grid(row=0, column=2)
        ttk.Label(self, text='Worksheet').grid(row=1, column=0, sticky='w', pady=4)
        self.sheet_box = ttk.Combobox(self, textvariable=self.vars['sheet'], state='readonly', width=27)
        self.sheet_box.grid(row=1, column=1, sticky='ew', padx=4)
        self.sheet_box.bind('<<ComboboxSelected>>', lambda e: self.read_headers(guess_row=True))
        self.header_frame = ttk.Frame(self)
        self.header_frame.grid(row=2, column=0, columnspan=3, sticky='ew')
        ttk.Label(self.header_frame, text='Header row').pack(side='left')
        ttk.Spinbox(self.header_frame, from_=1, to=1000, textvariable=self.vars['header_row'], width=5).pack(side='left', padx=6)
        ttk.Button(self.header_frame, text='Read / refresh columns', command=self.read_headers).pack(side='left', padx=6)
        self.count = ttk.Label(self.header_frame, text='Not loaded')
        self.count.pack(side='left', padx=8)
        self.boxes = {}
        for r, (key, title) in enumerate(COLUMN_FIELDS, 3):
            ttk.Label(self, text=title).grid(row=r, column=0, sticky='w', pady=3)
            box = ttk.Combobox(self, textvariable=self.vars[key], state='readonly', width=29)
            box.grid(row=r, column=1, columnspan=2, sticky='ew', padx=4)
            self.boxes[key] = box
        ttk.Label(self, text='Mass source').grid(row=10, column=0, sticky='w', pady=4)
        ttk.Combobox(self, textvariable=self.vars['mass_mode'], values=list(MODE_LABELS), state='readonly', width=32).grid(row=10, column=1, columnspan=2, sticky='ew', padx=4)
        ttk.Label(self, text='Default ion (no ion column)').grid(row=11, column=0, sticky='w', pady=4)
        ttk.Combobox(self, textvariable=self.vars['fixed_adduct'], values=['[M-H]-', '[M+H]+', '[M+HCOO]-', '[M+Na]+', '[M+NH4]+', '[M+Cl]-', '[M-2H]2-'], width=29).grid(row=11, column=1, columnspan=2, sticky='ew', padx=4)
        ttk.Label(self, text='XIC output directory').grid(row=12, column=0, sticky='w', pady=4)
        ttk.Entry(self, textvariable=self.vars['image_dir'], width=27).grid(row=12, column=1, sticky='ew', padx=4)
        ttk.Button(self, text='Browse', command=self.browse_images).grid(row=12, column=2)
        ttk.Label(self, text='Original summary / mapping').grid(row=13, column=0, sticky='w', pady=4)
        ttk.Entry(self, textvariable=self.vars['metadata_path'], width=27).grid(row=13, column=1, sticky='ew', padx=4)
        ttk.Button(self, text='Browse', command=self.browse_metadata).grid(row=13, column=2)
        ttk.Label(self, text='Select the directory containing all replicate subdirectories. Original __enumerated and __XIC_quant CSV indexes are used when available.', wraplength=470).grid(row=14, column=0, columnspan=3, sticky='w')
        ttk.Label(self, text='Thermo RAW directory').grid(row=15, column=0, sticky='w', pady=4)
        ttk.Entry(self, textvariable=self.vars['raw_dir'], width=27).grid(row=15, column=1, sticky='ew', padx=4)
        ttk.Button(self, text='Browse', command=self.browse_raw).grid(row=15, column=2)
        for rr, key, label_text in [(16, 'raw_col', 'RAW column (optional)'), (17, 'rt_col', 'Apex RT column (min; optional)')]:
            ttk.Label(self, text=label_text).grid(row=rr, column=0, sticky='w', pady=3)
            box = ttk.Combobox(self, textvariable=self.vars[key], state='readonly', width=29)
            box.grid(row=rr, column=1, columnspan=2, sticky='ew', padx=4)
            self.boxes[key] = box
        ttk.Label(self, text='Use per-replicate RAW names, apex RT and peak bounds from the original quantification files. A mean RT is not a per-RAW apex time.', wraplength=470).grid(row=18, column=0, columnspan=3, sticky='w')
        from .ui_support import wrap_labels
        wrap_labels(self, max_single=215)
        self.headers = {}
        self.book = None

    def browse_raw(self):
        path = filedialog.askdirectory(parent=self)
        if path: self.vars['raw_dir'].set(path)

    def browse_metadata(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[('Original summary or mapping', '*.xlsx *.xlsm *.csv')])
        if path: self.vars['metadata_path'].set(path)

    def browse_images(self):
        path = filedialog.askdirectory(parent=self)
        if path:
            self.vars['image_dir'].set(path)

    def browse(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[('Excel', '*.xlsx *.xlsm')])
        if path:
            self.vars['path'].set(path)
            self.load_file()

    def load_file(self):
        try:
            from .final_table_postprocess import autoconfig
            from .final_table_ooxml import Book
            path = self.vars['path'].get().strip()
            self.book = Book(path)
            self.sheet_box['values'] = list(self.book.sheet_parts)
            cfg = autoconfig(path, label=self.label)
            self.vars['sheet'].set(cfg.sheet)
            self.read_headers(guess_row=True)
        except Exception as exc:
            messagebox.showerror('Could not read workbook', str(exc), parent=self)

    def read_headers(self, guess_row=False):
        try:
            from .final_table_postprocess import autoconfig, pick, FORMULA_ALIASES, NAME_ALIASES, MASS_ALIASES, IMAGE_ALIASES
            from .final_table_ooxml import Book, colname
            self.book = Book(self.vars['path'].get().strip())
            sheet = self.vars['sheet'].get()
            if guess_row:
                self.vars['header_row'].set(str(autoconfig(self.vars['path'].get(), sheet).header_row))
            self.headers, rows = self.book.scan(sheet, int(self.vars['header_row'].get()))
            items = ['(Automatic / not specified)'] + ['%s | %s' % (colname(c), h) for c, h in sorted(self.headers.items())]
            defaults = {'name_col': pick(self.headers, NAME_ALIASES), 'formula_col': pick(self.headers, FORMULA_ALIASES),
                        'mass_col': pick(self.headers, MASS_ALIASES), 'image_col': pick(self.headers, IMAGE_ALIASES),
                        'adduct_col': pick(self.headers, ('Adduct', 'Ion', 'Ion form', 'Adduct'))}
            for key, box in self.boxes.items():
                box['values'] = items
                col = defaults.get(key, 0)
                self.vars[key].set('%s | %s' % (colname(col), self.headers[col]) if col else items[0])
            self.count.config(text='%s nonempty rows' % len(rows))
        except Exception as exc:
            messagebox.showerror('Could not read headers', str(exc), parent=self)

    def config(self):
        from .final_table_postprocess import TableConfig
        from .final_table_ooxml import colindex
        def column(key):
            text = self.vars[key].get()
            return colindex(text.split(' | ', 1)[0]) if ' | ' in text else 0
        return TableConfig(path=self.vars['path'].get().strip(), sheet=self.vars['sheet'].get(),
                           header_row=int(self.vars['header_row'].get()),
                           mass_mode=MODE_LABELS[self.vars['mass_mode'].get()], fixed_adduct=self.vars['fixed_adduct'].get(),
                           image_dir=self.vars['image_dir'].get().strip(),
                           metadata_path=self.vars['metadata_path'].get().strip(),
                           raw_dir=self.vars['raw_dir'].get().strip(), raw_col=column('raw_col'), rt_col=column('rt_col'),
                           **{key: column(key) for key, _ in COLUMN_FIELDS})


class FinalTablePanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.events = queue.Queue()
        self.running = False
        self.last_folder = None
        self._poll_after = None
        canvas = tk.Canvas(self, highlightthickness=0, background='white')
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y'); canvas.pack(side='left', fill='both', expand=True)
        body = ttk.Frame(canvas, padding=10)
        win = canvas.create_window((0, 0), window=body, anchor='nw')
        body.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(win, width=e.width))
        self.body = body
        self.canvas = canvas
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=1)
        ttk.Label(body, text='Final tables', font=('', 13, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 4))
        ttk.Label(body, text='Inputs are read-only. Results are saved in a new directory. Observed m/z can be extracted from RAW without refitting models or changing concentration, area or Exact_mass.', wraplength=1030).grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 8))
        self.edit_existing_btn = ttk.Button(body, text='Edit XIC images in existing results...', command=self.open_existing_xic)
        self.edit_existing_btn.grid(row=0, column=1, sticky='e')
        self.known = InputPanel(body, '100'); self.unknown = InputPanel(body, '1000')
        self.known.grid(row=2, column=0, sticky='nsew', padx=(0, 6))
        self.unknown.grid(row=2, column=1, sticky='nsew', padx=(6, 0))
        settings = ttk.Frame(body); settings.grid(row=3, column=0, columnspan=2, sticky='ew', pady=8)
        self.recursive = tk.BooleanVar(value=True)
        self.allow_formula = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text='Search XIC subdirectories', variable=self.recursive).pack(side='left')
        ttk.Checkbutton(settings, text='Allow unique-formula name matching (isomers require review)', variable=self.allow_formula).pack(side='left', padx=10)
        ttk.Label(settings, text='QC limit (ppm)').pack(side='left')
        self.ppm = tk.StringVar(value='5')
        ttk.Entry(settings, textvariable=self.ppm, width=5).pack(side='left', padx=4)
        imageopts = ttk.Frame(body); imageopts.grid(row=4, column=0, columnspan=2, sticky='ew')
        self.width = tk.StringVar(value='520'); self.height = tk.StringVar(value='220')
        ttk.Label(imageopts, text='XIC size: width').pack(side='left')
        ttk.Entry(imageopts, textvariable=self.width, width=5).pack(side='left', padx=4)
        ttk.Label(imageopts, text='height').pack(side='left')
        ttk.Entry(imageopts, textvariable=self.height, width=5).pack(side='left', padx=4)
        ttk.Label(imageopts, text='px. Images retain their original aspect ratio, channels, axes and legends.').pack(side='left', padx=4)
        self.xic_display_mode = tk.StringVar(value='first')
        displayopts = ttk.LabelFrame(body, text='XIC layout (applies to both tables)', padding=6)
        displayopts.grid(row=5, column=0, columnspan=2, sticky='ew', pady=6)
        self.display_buttons = {}
        for value, title in DISPLAY_LABELS.items():
            button = ttk.Radiobutton(displayopts, text=title + (' (default)' if value == 'first' else ''),
                                     variable=self.xic_display_mode, value=value, command=self.display_changed)
            button.pack(side='left', padx=(0, 18))
            self.display_buttons[value] = button
        ttk.Label(displayopts, text='Single-image mode uses the first readable, unambiguous image in Rep 1, 2, 3 order. Sources are recorded.').pack(side='left')
        captions = ttk.LabelFrame(body, text='Additional XIC captions (none by default)', padding=6)
        captions.grid(row=6, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        captions.columnconfigure(1, weight=1)
        self.caption_replicate = tk.BooleanVar(value=False)
        self.caption_raw = tk.BooleanVar(value=False)
        self.caption_name = tk.BooleanVar(value=False)
        self.caption_custom = tk.StringVar(value='')
        self.caption_preview = tk.StringVar(value='')
        choices = ttk.Frame(captions); choices.grid(row=0, column=0, columnspan=3, sticky='w')
        self.caption_buttons = {}
        for key, title, variable in (('replicate', 'Replicate number (Rep 1/2/3)', self.caption_replicate),
                                      ('raw', 'RAW filename', self.caption_raw),
                                      ('compound_name', 'Compound name in this table', self.caption_name)):
            button = ttk.Checkbutton(choices, text=title, variable=variable, command=self.caption_changed)
            button.pack(side='left', padx=(0, 18)); self.caption_buttons[key] = button
        ttk.Button(choices, text='Clear captions', command=self.clear_captions).pack(side='left')
        ttk.Label(captions, text='Custom caption').grid(row=1, column=0, sticky='w', pady=3)
        self.caption_entry = ttk.Entry(captions, textvariable=self.caption_custom)
        self.caption_entry.grid(row=1, column=1, sticky='ew', padx=6)
        ttk.Label(captions, text='Up to 200 characters; literal text, not an expression.').grid(row=1, column=2, sticky='w')
        ttk.Label(captions, textvariable=self.caption_preview, wraplength=990).grid(row=2, column=0, columnspan=3, sticky='w', pady=2)
        ttk.Label(captions, text='Only added captions are controlled here. Original titles, axes and legends are unchanged. Missing or conflicting replicates remain marked.',
                  wraplength=990).grid(row=3, column=0, columnspan=3, sticky='w')
        self.caption_custom.trace_add('write', self.caption_changed)
        self.caption_changed()
        self.source_index = tk.BooleanVar(value=True)
        self.abc_formula = tk.BooleanVar(value=True)
        newopts = ttk.Frame(body); newopts.grid(row=7, column=0, columnspan=2, sticky='ew', pady=6)
        ttk.Checkbutton(newopts, text='Read original export indexes', variable=self.source_index).pack(side='left', padx=8)
        ttk.Checkbutton(newopts, text='Match ordered A/B/C formula keys (review required)', variable=self.abc_formula).pack(side='left')
        massopts = ttk.Frame(body); massopts.grid(row=8, column=0, columnspan=2, sticky='ew', pady=8)
        self.mass_search = tk.StringVar(value='20'); self.mass_rt = tk.StringVar(value='0.05')
        ttk.Label(massopts, text='RAW search half-window (ppm)').pack(side='left')
        ttk.Entry(massopts, textvariable=self.mass_search, width=6).pack(side='left', padx=5)
        ttk.Label(massopts, text='Apex RT half-window (min)').pack(side='left')
        ttk.Entry(massopts, textvariable=self.mass_rt, width=6).pack(side='left', padx=5)
        ttk.Label(massopts, text='Use the nearest eligible MS1 centroid. Replicates remain separate; the smallest ppm error is not used for selection.').pack(side='left')
        out = ttk.Frame(body); out.grid(row=9, column=0, columnspan=2, sticky='ew', pady=8)
        out.columnconfigure(1, weight=1)
        ttk.Label(out, text='Output directory').grid(row=0, column=0)
        self.output = tk.StringVar(value=str(Path.cwd() / 'final_table_outputs'))
        ttk.Entry(out, textvariable=self.output).grid(row=0, column=1, sticky='ew', padx=8)
        ttk.Button(out, text='Browse', command=self.browse_output).grid(row=0, column=2)
        ttk.Label(body, text='Theoretical values are displayed to six decimals. ppm values retain their sign; zero is 0.000000 and very small nonzero values use scientific notation.\nIn RAW mode, observed and theoretical m/z refer to the same ion. Missing observations remain blank. Specify a shared key in both tables, or neither.', wraplength=1040).grid(row=10, column=0, columnspan=2, sticky='w', pady=3)
        ctl = ttk.Frame(body); ctl.grid(row=11, column=0, columnspan=2, sticky='ew', pady=8)
        self.preview_btn = ttk.Button(ctl, text='Check all matches', command=self.preview)
        self.preview_btn.pack(side='left')
        self.run_btn = ttk.Button(ctl, text='Process tables and embed XIC', command=self.run)
        self.run_btn.pack(side='left', padx=10)
        ttk.Button(ctl, text='Open results', command=self.open_folder).pack(side='left')
        self.status = ttk.Label(ctl, text='Ready'); self.status.pack(side='left', padx=12)
        self.log = tk.Text(body, height=7, wrap='word')
        self.log.grid(row=12, column=0, columnspan=2, sticky='nsew')
        self._poll_after = self.after(100, self.poll)

    def destroy(self):
        if self._poll_after:
            try: self.after_cancel(self._poll_after)
            except tk.TclError: pass
            self._poll_after = None
        super().destroy()

    def open_existing_xic(self):
        if self.running:
            messagebox.showinfo('Processing', 'Wait until the current export finishes before editing existing results.', parent=self)
            return
        from .existing_xic_lab import open_window
        return open_window(self)

    def browse_output(self):
        path = filedialog.askdirectory(parent=self)
        if path: self.output.set(path)

    def display_changed(self):
        # Resize only the default presets; preserve deliberately customized sizes.
        if self.height.get() in ('220', '480'):
            self.height.set('480' if self.xic_display_mode.get() == 'all' else '220')

    def caption_settings(self):
        from .final_table_captions import CaptionSettings
        return CaptionSettings(self.caption_replicate.get(), self.caption_raw.get(),
                               self.caption_name.get(), self.caption_custom.get()).validate()

    def caption_changed(self, *_):
        from .final_table_captions import build_caption
        try:
            settings = self.caption_settings()
            example = build_caption({'rep': '1', 'raw': 'Sample-1'}, settings, 'Compound_A')
            self.caption_preview.set('Caption preview: ' + (example or '(None; original image only)'))
        except ValueError as exc:
            self.caption_preview.set(str(exc))

    def clear_captions(self):
        self.caption_replicate.set(False); self.caption_raw.set(False); self.caption_name.set(False)
        self.caption_custom.set(''); self.caption_changed()

    def collect(self):
        from .final_table_postprocess import Options
        self.caption_settings()  # Validate before any RAW is opened or output directory is created.
        return self.known.config(), self.unknown.config(), Options(
            recursive=self.recursive.get(), allow_unique_formula=self.allow_formula.get(),
            image_width=int(self.width.get()), image_height=int(self.height.get()), ppm_limit=float(self.ppm.get()),
            triplicate_xic=True, use_source_index=self.source_index.get(), allow_abc_formula=self.abc_formula.get(),
            xic_display_mode=self.xic_display_mode.get(), observed_search_ppm=float(self.mass_search.get()),
            observed_rt_halfwindow=float(self.mass_rt.get()),
            xic_caption_replicate=self.caption_replicate.get(), xic_caption_raw=self.caption_raw.get(),
            xic_caption_name=self.caption_name.get(), xic_caption_custom=self.caption_custom.get())

    def preview(self):
        try:
            known, unknown, options = self.collect()
            from .final_table_postprocess import prepare_context, mass_result
            context = prepare_context(known, unknown, options)
            links = context['links']
            dialog = tk.Toplevel(self); dialog.title('All rows checked; first 12 rows of each table shown'); dialog.geometry('1240x530')
            cols = ('Table', 'Row', 'Name', 'Theory', 'PPM', 'XIC', 'Display', 'Name_in_1000', 'Source')
            tree = ttk.Treeview(dialog, columns=cols, show='headings')
            for c in cols:
                tree.heading(c, text=c); tree.column(c, width=110 if c not in ('XIC', 'Display', 'Source') else 260)
            for label, cfg in [('100', known), ('1000', unknown)]:
                view = context[label]
                for row in view['rows'][:12]:
                    m = mass_result(row, cfg, view['headers'], options.ppm_limit)
                    p = m['ppm']; link = links.get(row['row'], {}) if label == '100' else {}
                    img = view['matches'][row['row']]
                    tree.insert('', 'end', values=(label, row['row'], row['name'],
                        format(m['theory'], '.6f') if m['theory'] is not None else '',
                        format_ppm_display(p) if p is not None else 'Not computable',
                        img['status'] + ' (' + str(len(img['paths'])) + ' matched)',
                        display_preview(img, options.xic_display_mode), link.get('name', ''), row.get('_source_status', '')))
            xs = ttk.Scrollbar(dialog, orient='horizontal', command=tree.xview); tree.configure(xscrollcommand=xs.set)
            xs.pack(side='bottom', fill='x'); tree.pack(fill='both', expand=True)
            lines = ['Additional captions: ' + self.caption_settings().description(),
                     'XIC layout: ' + DISPLAY_LABELS[options.xic_display_mode], 'The preview does not read RAW files. Observed m/z and ppm are extracted during processing.']
            from collections import Counter
            for label in ('100', '1000'):
                v = context[label]
                counts = Counter(m['status'] for m in v['matches'].values())
                lines.append('%s table: %s image files; %s/%s rows matched; %s' % (label, len(v['images'].files),
                    sum(bool(m['paths']) for m in v['matches'].values()), len(v['rows']), dict(counts)))
            lines.append('Calibration-to-target name matches: %s. Duplicate image claims have been checked. Image readability is verified during export.' %
                         sum(v['status'].startswith('MATCHED') for v in links.values()))
            ttk.Label(dialog, text='\n'.join(lines), wraplength=1200).pack(fill='x')
        except Exception as exc:
            messagebox.showerror('Preview failed', str(exc), parent=self)

    def run(self):
        if self.running: return
        try:
            known, unknown, options = self.collect()
            output = self.output.get().strip()
            if not output: raise ValueError('Select an output directory.')
        except Exception as exc:
            messagebox.showerror('Invalid settings', str(exc), parent=self); return
        self.running = True; self.run_btn.config(state='disabled'); self.preview_btn.config(state='disabled')
        self.status.config(text='Processing...'); self.log.delete('1.0', 'end')
        def worker():
            try:
                from .final_table_postprocess import execute
                folder, result = execute(known, unknown, output, options, lambda msg: self.events.put(('log', msg)))
                self.events.put(('done', (folder, result)))
            except Exception:
                self.events.put(('error', traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log': self.log.insert('end', value + '\n')
                elif kind == 'done':
                    self.last_folder, result = value
                    self.running = False; self.run_btn.config(state='normal'); self.preview_btn.config(state='normal')
                    self.status.config(text='Complete; check match counts')
                    for label, summary in result['tables'].items():
                        name_text = str(summary['name_matches']) + ' rows' if summary['name_matching_applicable'] else 'Not applicable (target table has no reverse name mapping)'
                        self.log.insert('end', '%s table: %s rows; %s theoretical values; %s ppm values; XIC in %s rows from %s images; %s single-image rows; %s complete triplicates; %s incomplete triplicates; name matches %s.\n' % (
                            label, summary['rows'], summary['theory_calculated'], summary['ppm_calculated'], summary['images_embedded'],
                            summary['source_images_embedded'], summary['single_image_rows'], summary['three_replicate_rows'], summary['partial_replicate_rows'], name_text))
                        if summary.get('mass_mode') == 'raw_mz':
                            self.log.insert('end', 'Observed RAW m/z: %s rows. Missing or conflicting results remain blank; see Observed_Mass_Audit.\n' % summary.get('observed_mz_count', 0))
                    self.log.insert('end', 'See postprocess_diagnostics.txt for counts and reasons. Postprocess_Audit distinguishes matched sources from displayed images.\n')
                elif kind == 'error':
                    self.running = False; self.run_btn.config(state='normal'); self.preview_btn.config(state='normal')
                    self.status.config(text='Failed; log retained')
                    self.log.insert('end', value + '\n')
                    messagebox.showerror('Postprocessing failed', value.splitlines()[-1], parent=self)
                self.log.see('end')
        except queue.Empty: pass
        self._poll_after = self.after(100, self.poll)

    def open_folder(self):
        if not self.last_folder: return
        if sys.platform.startswith('win'): os.startfile(str(self.last_folder))
        else: subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(self.last_folder)])


def main():
    root = tk.Tk(); root.title('CombiTrace-MS | Final tables'); root.geometry('1220x960'); root.minsize(950, 600)
    panel = FinalTablePanel(root); panel.pack(fill='both', expand=True)
    def close():
        if panel.running and not messagebox.askyesno('Processing is still active', 'Closing will interrupt processing. Output without a completion marker is incomplete. Close anyway?', parent=root): return
        panel.destroy(); root.destroy()
    root.protocol('WM_DELETE_WINDOW', close); root.mainloop()
