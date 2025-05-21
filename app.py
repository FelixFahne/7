import gradio as gr, subprocess, pathlib, shutil, uuid, os

SRC = pathlib.Path(__file__).parent / "src"
TMP = pathlib.Path("/tmp/space")         # temporäre Arbeits­verzeichnisse
TMP.mkdir(parents=True, exist_ok=True)

# ---------- Hilfsfunktionen ----------
def _run_notebook(ipynb_path, out_dir):
    """Führt ein Notebook headless aus und legt Ergebnis-CSV in out_dir ab."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    # Name ohne Endung für Output
    output_name = out_dir / f"{uuid.uuid4()}.ipynb"
    subprocess.check_call(
        ["papermill", str(ipynb_path), str(output_name)],
    )
    return output_name

# ---------- Gradio Callbacks ----------
def preprocess(csv_file):
    ipynb = SRC / "dialogue_pred.ipynb"   # anpassen, falls dein Notebook anders heißt
    result = _run_notebook(ipynb, TMP)
    return gr.File(result)

def train(processed_csv):
    ipynb = SRC / "ESL_AddedExperinments.ipynb"
    result = _run_notebook(ipynb, TMP)
    return gr.File(result)

def infer(model_pkl, test_csv):
    # Beispiel: führe ein Python-Script aus dem Originalrepo aus
    output_csv = TMP / f"preds_{uuid.uuid4()}.csv"
    subprocess.check_call(
        ["python", SRC / "dialogue_pred.py",
         "--model", model_pkl.name,
         "--test", test_csv.name,
         "--out", output_csv],
    )
    return gr.File(output_csv)

# ---------- GUI ----------
with gr.Blocks() as demo:
    with gr.Tab("Pre-processing"):
        in_file = gr.File(label="Roh-CSV hochladen")
        btn = gr.Button("Start")
        out_file = gr.File(label="Ergebnis-Notebook")
        btn.click(preprocess, in_file, out_file)

    with gr.Tab("Training"):
        proc = gr.File(label="Vorverarbeitete CSV")
        btn2 = gr.Button("Trainieren")
        model_out = gr.File(label="Trainings-Notebook")
        btn2.click(train, proc, model_out)

    with gr.Tab("Application"):
        model = gr.File(label="Modell-Datei (.pkl)")
        test = gr.File(label="Test-CSV")
        btn3 = gr.Button("Vorhersagen")
        preds = gr.File(label="Ergebnis-CSV")
        btn3.click(infer, [model, test], preds)

demo.launch()
