"""
app.py – Gradio-UI für das wissenschaftliche Tool-Set
----------------------------------------------------
Abbildet den vollständigen Workflow

  1. Upload externer Roh-Datei  ➜  preprocessing.py
  2. Download Bundle (feature_label.csv + alle Artefakte)
  3. Upload feature_label.csv   ➜  dialogue_pred.py
  4. Download Bundle (model.pkl + alle Artefakte)
  5. Upload benötigte Bundles   ➜  ESL_AddedExperinments.py
  6. Download finales Ergebnis-Bundle

Alle Zwischen-Artefakte werden automatisch in ZIP-Paketen
bereitgestellt, so dass Nutzer jederzeit Sicherungen erhalten.
Only *this* file touches the UI / Workflow – the three backbone
scripts remain untouched.
"""
import gradio as gr
import subprocess
import shutil
import uuid
import sys
import time
from pathlib import Path
from zipfile import ZipFile

# ---------------------------------------------------------------------------
# Allgemeine Utilities
# ---------------------------------------------------------------------------

ROOT_TMP = Path("/tmp/space")  # zentraler Arbeitsort


def _new_session_subdir(step_name: str) -> Path:
    """Erzeugt einen eindeutigen Unterordner für den laufenden User-Step."""
    session_id = str(uuid.uuid4())
    subdir = ROOT_TMP / session_id / step_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def _collect_new_artifacts(workdir: Path, start_ts: float) -> list[Path]:
    """Alle Dateien/Folder, die *nach* start_ts entstanden sind."""
    artifacts: list[Path] = []
    for p in workdir.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime >= start_ts:
                artifacts.append(p)
        except FileNotFoundError:  # Datei könnte im Lauf gelöscht worden sein
            pass
    return artifacts


def _zip_artifacts(artifacts: list[Path], bundle_name: str, base_dir: Path) -> Path:
    """Zippt alle übergebenen Pfade relativ zu base_dir."""
    bundle_path = base_dir / bundle_name
    with ZipFile(bundle_path, "w") as zf:
        for art in artifacts:
            # Pfad relativ zum Arbeitsordner, damit Struktur erhalten bleibt
            arcname = art.relative_to(base_dir)
            zf.write(art, arcname)
    return bundle_path


def _safe_copy(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    shutil.copy(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Schritt 1: Preprocessing
# ---------------------------------------------------------------------------

def run_preprocess(upload_file):
    workdir = _new_session_subdir("preprocess")
    uploaded_path = _safe_copy(Path(upload_file.name), workdir)

    start_ts = time.time()
    try:
        # Aufruf preprocessing.py im Arbeitsverzeichnis
        subprocess.run(
            [sys.executable, "preprocessing.py", str(uploaded_path)],
            cwd=workdir,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return gr.Error(f"Preprocessing fehlgeschlagen:\n{e.stderr.decode()}")

    # Artefakte einsammeln
    artifacts = _collect_new_artifacts(workdir, start_ts)
    # feature_label.csv identifizieren (Pfad merken für separate Rückgabe)
    try:
        feature_csv = next(p for p in artifacts if p.name == "feature_label.csv")
    except StopIteration:
        return gr.Error("feature_label.csv wurde nicht erzeugt.")

    # Zusätzliche Artefakte (Annotations-CSVs, data_csv_sample, …)
    bundle_artifacts = [
        p for p in artifacts if p.name != "feature_label.csv"
    ]
    bundle_file = _zip_artifacts(
        bundle_artifacts, f"preprocess_bundle_{workdir.parent.name}.zip", workdir
    )

    return feature_csv, bundle_file


# ---------------------------------------------------------------------------
# Schritt 3: Training / dialogue_pred.py
# ---------------------------------------------------------------------------

def run_train(feature_csv_file):
    workdir = _new_session_subdir("train")
    feature_csv = _safe_copy(Path(feature_csv_file.name), workdir)

    start_ts = time.time()
    try:
        subprocess.run(
            [sys.executable, "dialogue_pred.py", str(feature_csv)],
            cwd=workdir,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return gr.Error(f"dialogue_pred fehlgeschlagen:\n{e.stderr.decode()}")

    artifacts = _collect_new_artifacts(workdir, start_ts)

    try:
        model_pkl = next(p for p in artifacts if p.suffix == ".pkl")
    except StopIteration:
        return gr.Error("model.pkl wurde nicht erzeugt.")

    bundle_artifacts = [p for p in artifacts if p != model_pkl]
    bundle_file = _zip_artifacts(
        bundle_artifacts, f"training_bundle_{workdir.parent.name}.zip", workdir
    )

    return model_pkl, bundle_file


# ---------------------------------------------------------------------------
# Schritt 5: ESL-Experimente
# ---------------------------------------------------------------------------

def run_esl(upload_files: list[gr.File]):
    workdir = _new_session_subdir("esl")
    # Alle Uploads kopieren & ggf. entpacken
    for f in upload_files:
        fpath = _safe_copy(Path(f.name), workdir)
        if fpath.suffix == ".zip":
            with ZipFile(fpath, "r") as zf:
                zf.extractall(workdir)

    start_ts = time.time()
    try:
        subprocess.run(
            [sys.executable, "ESL_AddedExperinments.py"],
            cwd=workdir,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return gr.Error(f"ESL-Experimente fehlgeschlagen:\n{e.stderr.decode()}")

    artifacts = _collect_new_artifacts(workdir, start_ts)
    if not artifacts:
        return gr.Error("ESL-Experimente erzeugten keine neuen Ausgaben.")

    bundle_file = _zip_artifacts(
        artifacts, f"esl_results_{workdir.parent.name}.zip", workdir
    )
    return bundle_file


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Scientific Dialogue Tool") as demo:
    gr.Markdown("# 🧪 Scientific Dialogue Processing Workflow")

    with gr.Tabs():
        # --- Tab 1: Preprocess ------------------------------------------------
        with gr.Tab("1 · Preprocess"):
            gr.Markdown(
                "Lade eine Roh-Datei hoch. "
                "Nach Abschluss erhältst du **feature_label.csv** "
                "und ein ZIP-Bundle aller Artefakte."
            )
            preprocess_file = gr.File(label="Roh-Datei hochladen")
            preprocess_btn = gr.Button("Preprocess starten")
            preprocess_out_csv = gr.File(label="feature_label.csv")
            preprocess_out_zip = gr.File(label="Preprocess-Bundle (.zip)")

            preprocess_btn.click(
                run_preprocess,
                inputs=preprocess_file,
                outputs=[preprocess_out_csv, preprocess_out_zip],
            )

        # --- Tab 2: Training --------------------------------------------------
        with gr.Tab("2 · Training"):
            gr.Markdown(
                "Lade **feature_label.csv** (aus Schritt 1) hoch. "
                "Du erhältst **model.pkl** und ein ZIP-Bundle aller "
                "Trainings-Artefakte."
            )
            train_file = gr.File(label="feature_label.csv hochladen")
            train_btn = gr.Button("Training starten")
            train_out_model = gr.File(label="model.pkl")
            train_out_zip = gr.File(label="Training-Bundle (.zip)")

            train_btn.click(
                run_train,
                inputs=train_file,
                outputs=[train_out_model, train_out_zip],
            )

        # --- Tab 3: ESL Experiments ------------------------------------------
        with gr.Tab("3 · ESL-Experimente"):
            gr.Markdown(
                "Lade alle benötigten Bundles hoch "
                "(z. B. Preprocess- und Training-Bundle) **oder** einzelne "
                "Dateien wie `data_csv_sample/`, `model.pkl` usw. "
                "Das Ergebnis ist ein ZIP-Bundle mit allen ESL-Outputs."
            )
            esl_files = gr.File(
                label="ZIP-Bundles/Dateien hochladen",
                file_count="multiple",
            )
            esl_btn = gr.Button("ESL Experimente starten")
            esl_out_zip = gr.File(label="ESL-Ergebnis-Bundle (.zip)")

            esl_btn.click(
                run_esl,
                inputs=esl_files,
                outputs=esl_out_zip,
            )

    gr.Markdown(
        "ℹ️ Alle temporären Arbeitsordner werden in `/tmp/space/<session>` "
        "angelegt und bei Neustart des Containers automatisch entfernt."
    )

if __name__ == "__main__":
    demo.launch()
