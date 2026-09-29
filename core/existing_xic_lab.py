"""Local edit window for existing result workbooks; no RAW or training inputs."""
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
from .existing_xic_edit import EditOptions, MODE_LABELS
from .final_table_captions import CaptionSettings, build_caption


class ExistingXICPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12)
        self.running = False
        self.events = queue.Queue()
        self.last_folder = None
        self.paths = []
        self._after = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(7, weight=1)
        ttk.Label(self, text='Edit XIC images in existing results', font=('', 14, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(self, text='Replace existing images without adding another XIC column. Observed m/z, ppm, concentrations, areas and name links are retained. RAW files are not read.',
                  wraplength=1050).grid(row=1, column=0, sticky='w', pady=(4, 10))
        filebox = ttk.LabelFrame(self, text='1. Existing result workbooks', padding=6)
        filebox.grid(row=2, column=0, sticky='ew')
        filebox.columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(filebox, height=4, selectmode='extended', exportselection=False)
        self.listbox.grid(row=0, column=0, rowspan=3, sticky='ew')
        ttk.Button(filebox, text='Add workbooks', command=self.browse).grid(row=0, column=1, padx=6)
        ttk.Button(filebox, text='Remove selected', command=self.remove).grid(row=1, column=1, padx=6)
        ttk.Button(filebox, text='Clear list', command=self.clear_files).grid(row=2, column=1, padx=6)
        self.latest = tk.BooleanVar(value=True)
        ttk.Checkbutton(filebox, text='Edit only the rightmost XIC column in each data sheet',
                        variable=self.latest).grid(row=3, column=0, columnspan=2, sticky='w', pady=4)
        opts = ttk.LabelFrame(self, text='2. Image layout and captions', padding=8)
        opts.grid(row=3, column=0, sticky='ew', pady=8); opts.columnconfigure(1, weight=1)
        self.mode = tk.StringVar(value='keep')
        modes = ttk.Frame(opts); modes.grid(row=0, column=0, columnspan=3, sticky='w')
        for key, label in MODE_LABELS.items():
            ttk.Radiobutton(modes, text=label, variable=self.mode, value=key, command=self.mode_changed).pack(side='left', padx=(0, 12))
        flags = ttk.Frame(opts); flags.grid(row=1, column=0, columnspan=3, sticky='w', pady=6)
        self.rep = tk.BooleanVar(value=False); self.raw = tk.BooleanVar(value=False); self.name = tk.BooleanVar(value=False)
        for label, var in [('Replicate number', self.rep), ('RAW filename', self.raw), ('Compound name in this table', self.name)]:
            ttk.Checkbutton(flags, text=label, variable=var, command=self.caption_changed).pack(side='left', padx=(0, 18))
        ttk.Button(flags, text='Clear captions', command=self.clear_captions).pack(side='left')
        ttk.Label(opts, text='Custom caption').grid(row=2, column=0, sticky='w')
        self.custom = tk.StringVar(value='')
        ttk.Entry(opts, textvariable=self.custom).grid(row=2, column=1, sticky='ew', padx=6)
        ttk.Label(opts, text='Up to 200 characters; captions are off by default.').grid(row=2, column=2)
        self.caption_preview = tk.StringVar()
        ttk.Label(opts, textvariable=self.caption_preview, wraplength=1000).grid(row=3, column=0, columnspan=3, sticky='w', pady=4)
        self.custom.trace_add('write', self.caption_changed); self.caption_changed()
        self.keep_size = tk.BooleanVar(value=True)
        self.width = tk.StringVar(value='520'); self.height = tk.StringVar(value='220')
        sizes = ttk.Frame(opts); sizes.grid(row=4, column=0, columnspan=3, sticky='w')
        ttk.Checkbutton(sizes, text='Keep existing dimensions (use layout defaults when switching single/triplicate)', variable=self.keep_size).pack(side='left')
        ttk.Label(sizes, text='Width').pack(side='left', padx=(10, 3)); ttk.Entry(sizes, textvariable=self.width, width=5).pack(side='left')
        ttk.Label(sizes, text='height').pack(side='left', padx=3); ttk.Entry(sizes, textvariable=self.height, width=5).pack(side='left')
        ttk.Label(sizes, text='px; used when existing dimensions are not retained').pack(side='left', padx=3)
        sources = ttk.LabelFrame(self, text='3. Image sources', padding=8)
        sources.grid(row=4, column=0, sticky='ew'); sources.columnconfigure(1, weight=1)
        self.image_dir = tk.StringVar()
        ttk.Label(sources, text='Original XIC image directory (optional)').grid(row=0, column=0, sticky='w')
        ttk.Entry(sources, textvariable=self.image_dir).grid(row=0, column=1, sticky='ew', padx=6)
        ttk.Button(sources, text='Browse', command=self.browse_images).grid(row=0, column=2)
        self.recover = tk.BooleanVar(value=True)
        ttk.Checkbutton(sources, text='Prefer verified image blocks embedded in the workbook; retain original plot titles and axes', variable=self.recover).grid(row=1, column=0, columnspan=3, sticky='w', pady=5)
        ttk.Label(sources, text='Additional replicates cannot be restored unless their embedded blocks or original images are available. Unsupported layouts and identity conflicts are left unchanged.',
                  wraplength=1010).grid(row=2, column=0, columnspan=3, sticky='w')
        output = ttk.Frame(self); output.grid(row=5, column=0, sticky='ew', pady=8); output.columnconfigure(1, weight=1)
        ttk.Label(output, text='Output directory').grid(row=0, column=0)
        self.output = tk.StringVar(value=str(Path.cwd() / 'xic_edited_outputs'))
        ttk.Entry(output, textvariable=self.output).grid(row=0, column=1, sticky='ew', padx=6)
        ttk.Button(output, text='Browse', command=self.browse_output).grid(row=0, column=2)
        controls = ttk.Frame(self); controls.grid(row=6, column=0, sticky='ew', pady=4)
        self.preview_btn = ttk.Button(controls, text='Check editable images', command=lambda: self.run(True)); self.preview_btn.pack(side='left')
        self.run_btn = ttk.Button(controls, text='Replace images and save copies', command=self.run); self.run_btn.pack(side='left', padx=10)
        ttk.Button(controls, text='Open results', command=self.open_folder).pack(side='left')
        self.status = ttk.Label(controls, text='Ready'); self.status.pack(side='left', padx=14)
        self.log = tk.Text(self, height=9, wrap='word'); self.log.grid(row=7, column=0, sticky='nsew', pady=(5, 0))
        self._after = self.after(100, self.poll)

    def caption_changed(self, *_):
        try:
            text = build_caption({'rep': '1', 'raw': 'Sample-1'}, self.captions(), 'Compound_A')
            self.caption_preview.set('Caption preview: ' + (text or '(No added caption; original plot content is retained)'))
        except ValueError as exc:
            self.caption_preview.set(str(exc))

    def captions(self):
        return CaptionSettings(self.rep.get(), self.raw.get(), self.name.get(), self.custom.get()).validate()

    def clear_captions(self):
        self.rep.set(False); self.raw.set(False); self.name.set(False); self.custom.set(''); self.caption_changed()

    def mode_changed(self):
        if self.height.get() in ('220', '480'):
            self.height.set('480' if self.mode.get() == 'all' else '220')

    def browse(self):
        if self.running: return
        paths = filedialog.askopenfilenames(parent=self, title='Select existing postprocessed workbooks', filetypes=[('Excel result', '*.xlsx *.xlsm')])
        for p in paths:
            if p not in self.paths:
                self.paths.append(p); self.listbox.insert('end', p)
        if paths and self.output.get() == str(Path.cwd() / 'xic_edited_outputs'):
            self.output.set(str(Path(paths[0]).parent))

    def remove(self):
        if self.running: return
        for i in reversed(self.listbox.curselection()):
            self.paths.pop(i); self.listbox.delete(i)

    def clear_files(self):
        if self.running: return
        self.paths = []; self.listbox.delete(0, 'end')

    def browse_images(self):
        path = filedialog.askdirectory(parent=self, title='Select original XIC image directory (not RAW)')
        if path: self.image_dir.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(parent=self)
        if path: self.output.set(path)

    def run(self, preview=False):
        if self.running: return
        try:
            if not self.paths: raise ValueError('Add an existing result workbook.')
            options = EditOptions(self.mode.get(), self.captions(), self.keep_size.get(), int(self.width.get()),
                                  int(self.height.get()), self.image_dir.get().strip(), self.latest.get(), self.recover.get()).validate()
            paths = list(self.paths); output = self.output.get().strip()
            if not preview and not output: raise ValueError('Select an output directory.')
        except Exception as exc:
            messagebox.showerror('Invalid settings', str(exc), parent=self); return
        self.running = True; self.preview_btn.configure(state='disabled'); self.run_btn.configure(state='disabled')
        self.status.configure(text='Checking...' if preview else 'Updating images only...'); self.log.delete('1.0', 'end')
        def worker():
            try:
                from .existing_xic_edit import execute
                result = execute(paths, output, options, lambda msg: self.events.put(('log', msg)), preview)
                self.events.put(('done', result))
            except Exception:
                self.events.put(('error', traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'log': self.log.insert('end', value + '\n')
                else:
                    self.running = False; self.preview_btn.configure(state='normal'); self.run_btn.configure(state='normal')
                    if kind == 'error':
                        self.status.configure(text='Failed; log retained'); self.log.insert('end', value + '\n')
                        messagebox.showerror('Image update failed', value.splitlines()[-1], parent=self)
                    else:
                        folder, summary = value
                        if folder: self.last_folder = folder
                        self.status.configure(text='Check complete' if summary['preview'] else 'Complete; check counts')
                        self.log.insert('end', 'Editable / replaced: %s images; unchanged: %s items.\n' % (summary['ready'] if summary['preview'] else summary['updated'], summary['unchanged']))
                        for e in [x for x in summary['events'] if x['Status'] == 'UNCHANGED'][:20]:
                            self.log.insert('end', '%s %s row %s: %s\n' % (e['Workbook'], e.get('Sheet', ''), e.get('Excel_row', ''), e['Reason']))
                        if folder: self.log.insert('end', str(folder) + '\nSee XIC_Edit_Audit and xic_edit_diagnostics.txt. Input files were not overwritten.\n')
                self.log.see('end')
        except queue.Empty:
            pass
        self._after = self.after(100, self.poll)

    def open_folder(self):
        if not self.last_folder: return
        if sys.platform.startswith('win'): os.startfile(str(self.last_folder))
        else: subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(self.last_folder)])

    def destroy(self):
        if self._after:
            try: self.after_cancel(self._after)
            except tk.TclError: pass
            self._after = None
        super().destroy()


def open_window(parent=None):
    root = tk.Toplevel(parent) if parent is not None else tk.Tk()
    root.title('CombiTrace-MS | Edit existing XIC images')
    root.geometry('1160x820'); root.minsize(1050, 760)
    panel = ExistingXICPanel(root); panel.pack(fill='both', expand=True)
    def close():
        if panel.running and not messagebox.askyesno('Processing is still active', 'Closing may interrupt export. Output without a completion marker is incomplete. Close anyway?', parent=root):
            return
        panel.destroy(); root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    return root, panel


def main():
    root, _ = open_window(); root.mainloop()
