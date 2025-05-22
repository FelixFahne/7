import gradio as gr, subprocess, pathlib, shutil, uuid, os
from prepare_data import prepare_data_structure

SRC = pathlib.Path(__file__).parent / "src"
# Directory used for intermediate results and generated CSVs. It defaults to
# ``/tmp/space`` but can be overridden via the ``SLDEA_WORKDIR`` environment
# variable to allow running in read-only locations.
TMP = pathlib.Path(os.getenv("SLDEA_WORKDIR", "/tmp/space"))
TMP.mkdir(parents=True, exist_ok=True)


# ---------- Hilfsfunktionen ----------
def _run_notebook(ipynb_path, out_dir, input_csv=None, cwd=None):
    """Führt ein Notebook headless aus und legt Ergebnis-CSV in out_dir ab.

    Wenn ``input_csv`` angegeben ist, wird der Pfad als Parameter ``input_csv``
    an ``papermill`` durchgereicht, sodass das Notebook die hochgeladene Datei
    nutzen kann.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    output_name = out_dir / f"{uuid.uuid4()}.ipynb"

    cmd = ["papermill", str(ipynb_path), str(output_name)]
    if input_csv:
        cmd += ["-p", "input_csv", str(input_csv)]

    subprocess.check_call(cmd, cwd=cwd)
    return output_name


# ---------- Gradio Callbacks ----------
def preprocess(csv_file):
    prepare_data_structure()
    ipynb = SRC / "dialogue_pred.ipynb"  # anpassen, falls dein Notebook anders heißt
    result = _run_notebook(ipynb, TMP, csv_file.name if csv_file else None, cwd=TMP)
    return gr.File(result)


def train(processed_csv):
    prepare_data_structure()
    ipynb = SRC / "ESL_AddedExperinments.ipynb"
    result = _run_notebook(ipynb, TMP, processed_csv.name if processed_csv else None, cwd=TMP)
    return gr.File(result)


def infer(model_pkl, test_csv):
    # Beispiel: führe ein Python-Script aus dem Originalrepo aus
    output_csv = TMP / f"preds_{uuid.uuid4()}.csv"
    subprocess.check_call(
        [
            "python",
            SRC / "dialogue_pred.py",
            "--model",
            model_pkl.name,
            "--test",
            test_csv.name,
            "--out",
            output_csv,
        ],
        cwd=TMP,
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
# ---------- Start ----------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # ← Space-Health-Check erreicht die App
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,  # optional: Exceptions im UI anzeigen
    )
