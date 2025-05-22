import gradio as gr, subprocess, pathlib, shutil, uuid, os, pandas as pd
from prepare_data import prepare_data_structure

# Ensure Matplotlib writes its config to a writable directory. This is useful in
# read-only environments such as Docker containers.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
pathlib.Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

SRC = pathlib.Path(__file__).parent / "src"
# Directory used for intermediate results and generated CSVs. It defaults to
# ``/tmp/space`` but can be overridden via the ``SLDEA_WORKDIR`` environment
# variable to allow running in read-only locations.
TMP = pathlib.Path(os.getenv("SLDEA_WORKDIR", "/tmp/space"))
TMP.mkdir(parents=True, exist_ok=True)


# ---------- Hilfsfunktionen ----------
def _run_notebook(ipynb_path, out_dir, cwd=None):
    """Führt ein Notebook headless aus und legt Ergebnis-CSV in out_dir ab."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    output_name = out_dir / f"{uuid.uuid4()}.ipynb"

    cmd = ["papermill", str(ipynb_path), str(output_name)]

    subprocess.check_call(cmd, cwd=cwd)
    return output_name


# ---------- Gradio Callbacks ----------
def preprocess(upload_files):
    uploads_dir = TMP / "uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
    uploads_dir.mkdir(parents=True)
    for f in upload_files:
        shutil.copy(f.name, uploads_dir / pathlib.Path(f.name).name)

    prepare_data_structure(uploads_dir, force=True)
    ipynb = SRC / "dialogue_pred.ipynb"  # anpassen, falls dein Notebook anders heißt
    result = _run_notebook(ipynb, TMP, cwd=TMP)
    return gr.File(result)


def train(train_files):
    uploads_dir = TMP / "train_uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
    uploads_dir.mkdir(parents=True)
    for f in train_files:
        shutil.copy(f.name, uploads_dir / pathlib.Path(f.name).name)

    prepare_data_structure(uploads_dir, force=True)
    ipynb = SRC / "ESL_AddedExperinments.ipynb"
    result = _run_notebook(ipynb, TMP, cwd=TMP)
    return gr.File(result)


def infer(model_pkl, test_files):
    """Run predictions on one or more test CSV/XLSX files."""
    frames = []
    for f in test_files:
        path = pathlib.Path(f.name)
        if path.suffix.lower() == ".xlsx":
            frames.append(pd.read_excel(f.name))
        else:
            frames.append(pd.read_csv(f.name))
    merged = pd.concat(frames, ignore_index=True)

    merged_csv = TMP / f"test_{uuid.uuid4()}.csv"
    merged.to_csv(merged_csv, index=False)

    output_csv = TMP / f"preds_{uuid.uuid4()}.csv"
    subprocess.check_call(
        [
            "python",
            SRC / "dialogue_pred.py",
            "--model",
            model_pkl.name,
            "--test",
            merged_csv,
            "--out",
            output_csv,
        ],
        cwd=TMP,
    )
    return gr.File(output_csv)


# ---------- GUI ----------
with gr.Blocks() as demo:
    with gr.Tab("Pre-processing"):
        in_files = gr.Files(label="Excel/CSV hochladen")
        btn = gr.Button("Start")
        out_file = gr.File(label="Ergebnis-Notebook")
        btn.click(preprocess, in_files, out_file)

    with gr.Tab("Training"):
        train_files = gr.Files(label="Trainingsdaten (Excel/CSV)")
        btn2 = gr.Button("Trainieren")
        model_out = gr.File(label="Trainings-Notebook")
        btn2.click(train, train_files, model_out)

    with gr.Tab("Application"):
        model = gr.File(label="Modell-Datei (.pkl)")
        test = gr.Files(label="Test-CSV oder XLSX")
        btn3 = gr.Button("Vorhersagen")
        preds = gr.File(label="Ergebnis-CSV")
        btn3.click(infer, [model, test], preds)
# ---------- Start ----------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # ← Space-Health-Check erreicht die App
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,  # optional: Exceptions im UI anzeigen
    )
