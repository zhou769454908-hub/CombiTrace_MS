"""Layout helpers for the English desktop interface."""
import tkinter as tk
from tkinter import ttk, font

def setup_style(root):
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background='#ffffff')
    style.configure('TFrame', background='#ffffff')
    style.configure('TLabel', background='#ffffff')
    style.configure('TButton', padding=(9, 5))
    style.configure('TNotebook', background='#f1f3f5', borderwidth=0)
    style.configure('TNotebook.Tab', padding=(10, 7))
    style.map('TNotebook.Tab', background=[('selected', '#ffffff')])
    root.configure(background='#ffffff')

def scroll_body(parent):
    canvas = tk.Canvas(parent, highlightthickness=0, background='white')
    bar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
    hbar = ttk.Scrollbar(parent, orient='horizontal', command=canvas.xview)
    parent.rowconfigure(0, weight=1); parent.columnconfigure(0, weight=1)
    canvas.grid(row=0,column=0,sticky='nsew');bar.grid(row=0,column=1,sticky='ns');hbar.grid(row=1,column=0,sticky='ew')
    canvas.configure(yscrollcommand=bar.set, xscrollcommand=hbar.set)
    body=ttk.Frame(canvas);item=canvas.create_window((0,0),window=body,anchor='nw')
    body.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>',lambda e:canvas.itemconfigure(item,width=max(e.width,1100)))
    # Do not install global mouse bindings: nested Text/Treeview widgets keep scrolling normally.
    canvas.bind('<MouseWheel>',lambda e:canvas.yview_scroll(-int(e.delta/120),'units'))
    canvas.bind('<Button-4>',lambda e:canvas.yview_scroll(-1,'units'))
    canvas.bind('<Button-5>',lambda e:canvas.yview_scroll(1,'units'))
    return body

def wrap_labels(parent,max_single=215):
    for w in parent.winfo_children():
        if isinstance(w,ttk.Label):
            try:
                value=str(w.cget('text'));info=w.grid_info()
                if info and len(value)>25:
                    span=int(info.get('columnspan',1))
                    w.configure(wraplength=min(1060,max_single*span),justify='left')
            except tk.TclError:
                pass
        wrap_labels(w,max_single)
