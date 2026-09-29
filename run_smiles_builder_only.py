"""Launch the main application directly on the standalone SMILES builder tab."""
from app import ThermoBatchReportApp

if __name__ == "__main__":
    app = ThermoBatchReportApp()
    app.title('CombiTrace-MS | Product SMILES builder')
    try:
        app.notebook.select(app.tab_smiles)
    except Exception:
        pass
    app.mainloop()
