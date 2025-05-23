# SLEDA Interaktive Metriken

SLEDA ist ein dreistufiges Framework zur Bewertung von englischen Dialogen im Bereich "English as a Second Language" (ESL). Dieses Repository enthält die Ursprungs-Notebooks, Beispiel-Daten und ein Docker-Setup, um das Werkzeug u.a. in einer HuggingFace Space auszuführen.

## Aufbau des Repository

- `SLDEA Data/` – Beispiel-Dialoge im XLSX-Format.
- `feature_label.csv` – aus den annotierten Datensätzen extrahierte Merkmale.
- `2024ACLESLMainCodes_Results/` – Beispielergebnisse aus der zugehörigen Publikation.
- `ESL_AddedExperinments.ipynb` – optionales Notebook für zusätzliche Experimente.
- `src/preprocessing.py` – wandelt hochgeladene Excel/CSV-Dateien in das vom Framework erwartete CSV-Format um und erzeugt `feature_label.csv`.

## Datensatz

Nur ein kleiner Teil des gesamten SLDEA-Datensatzes ist enthalten. Für vollständigen Zugriff bitte `rena.gao@unimelb.edu.au` kontaktieren. Dateien sollten unter `src/SLDEA Data/` abgelegt werden, damit die Notebooks sie finden können bzw. im Webinterface hochgeladen werden.

## Arbeitsverzeichnis

`app.py` und `src/preprocessing.py` schreiben Zwischenergebnisse sowie erzeugte CSV-Dateien in ein Verzeichnis, das über die Umgebungsvariable `SLDEA_WORKDIR` gesteuert wird. Standardmäßig wird `/tmp/space` verwendet. Beim Einsatz der Weboberfläche können beliebig viele Excel/CSV-Dateien hochgeladen werden; sie landen automatisch in diesem Ordner. Wird `src/preprocessing.py` manuell ausgeführt, sollten mindestens fünf Excel-Dateien in `src/SLDEA Data/` vorhanden sein, damit `annotations(1).csv`–`annotations(5).csv` erzeugt werden können.

```bash
python src/preprocessing.py
```

Der Aufruf kann bei Bedarf wiederholt werden, z.B. wenn neue Daten hinzugefügt wurden oder das Zielverzeichnis per `SLDEA_WORKDIR` gewechselt wurde.

## Ausführen der Notebooks

Die Notebooks in `src/` beschreiben die Vorverarbeitung und Experimente aus der Publikation. Sie erwarten CSV-Dateien namens `annotations(1).csv` bis `annotations(5).csv` sowie einen Ordner `data_csv_sample/`. `src/preprocessing.py` legt diese bei Bedarf an.

```bash
papermill ESL_AddedExperinments.ipynb out.ipynb
```

Die Python-Skripte `dialogue_pred.py` und `ESL_AddedExperinments.py` bieten dieselbe Funktionalität ohne Jupyter.

## Kommandozeilen-Beispiel

```bash
export SLDEA_WORKDIR=$HOME/sleda_workdir
python src/preprocessing.py        # erzeugt annotations(1).csv–annotations(5).csv und feature_label.csv
python src/dialogue_pred.py        # trainiert und speichert model.pkl
python src/dialogue_pred.py --model model.pkl --test test.csv --out preds.csv
python src/ESL_AddedExperinments.py experiments.csv  # optional
```

## Lokaler Docker-Build

```bash
docker build -t sleda-space .
docker run -p 7860:7860 sleda-space
```

Anschließend steht eine Gradio-Oberfläche auf http://localhost:7860 bereit.

## Bereitstellung auf HuggingFace Spaces

Eine neue **Docker**-Space auf HuggingFace erstellen und dieses Repository verknüpfen. Beim Build nutzt die Space das `Dockerfile` im Root, installiert Abhängigkeiten, konvertiert die Notebooks und startet `app.py`. Die Weboberfläche bietet danach drei Tabs sowie optional **Extra Experiments**:

1. **Pre-processing** – Excel/CSV hochladen und `preprocessing.py` ausführen, um `feature_label.csv` zu erzeugen.
2. **Training** – `feature_label.csv` hochladen und `dialogue_pred.py` zum Trainieren starten. Das resultierende `model.pkl` wird gespeichert.
3. **Application** – `model.pkl` zusammen mit Test-CSV/XLSX hochladen. `dialogue_pred.py` erzeugt Vorhersagen in `preds_<uuid>.csv`.
4. **Extra Experiments** *(optional)* – ein Experimente-CSV hochladen, um `ESL_AddedExperinments.ipynb` auszuführen.

Der Container veröffentlicht Port `7860`, der von Spaces automatisch weitergeleitet wird.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.
