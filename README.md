---
title: SLEDA Interactive Metrics
emoji: "\U0001F4CA"
colorFrom: green
colorTo: indigo
sdk: docker
sdk_version: "0.1"
app_file: app.py
pinned: false
---

# SLEDA Interactive Metrics

SLEDA is a three-level framework for evaluating English Second Language (ESL) conversation dialogues. This repository contains the original notebooks, supporting data and a small Docker setup for running the tool in a HuggingFace Space.

## Repository layout

- `SLDEA Data/` – example ESL dialogues in XLSX format.
- `feature_label.csv` – features extracted from the annotated datasets.
- `2024ACLESLMainCodes_Results/` – sample results from the accompanying paper.
- `ESL_AddedExperinments.ipynb` – optional notebook for extra experiments.
- `src/preprocessing.py` – converts uploaded Excel/CSV files to the CSV structure
  expected by the notebooks and produces `feature_label.csv`.

## Dataset

Only a small sample of the full SLDEA dataset is provided here. For complete access contact `rena.gao@unimelb.edu.au`. Place the files under `src/SLDEA Data/` so that the notebooks can locate them or upload your own Excel/CSV files in the web interface.

## Working directory

Both `app.py` and `src/preprocessing.py` write their intermediate results and the generated annotation CSV files to a directory controlled by the `SLDEA_WORKDIR` environment variable. If the variable is not set, `/tmp/space` is used by default. When running the web interface you may upload any number of Excel/CSV files which are placed in this directory automatically. If you run `src/preprocessing.py` manually, make sure at least five Excel files reside in `src/SLDEA Data/` so that the helper can build `annotations(1).csv`–`annotations(5).csv`.
`src/preprocessing.py` expects these `annotations` files to live in the directory referenced by `SLDEA_WORKDIR`.

```bash
python src/preprocessing.py
```

Running the command again is useful if you add more Excel files later or want to regenerate the CSVs in a different location by setting `SLDEA_WORKDIR`.

## Running the notebooks

The notebooks in `src/` contain the preprocessing steps and experiments described in the paper. They require CSV files named `annotations(1).csv` to `annotations(5).csv` and a folder `data_csv_sample/`. The script `src/preprocessing.py` creates these files from the Excel sheets under `SLDEA Data/` if they are missing.

You can execute `ESL_AddedExperinments.ipynb` from the command line with [Papermill](https://papermill.readthedocs.io/):

```bash
papermill ESL_AddedExperinments.ipynb out.ipynb
```

The Python scripts `dialogue_pred.py` and `ESL_AddedExperinments.py` implement the same functionality without requiring Jupyter.

## Command-line example

Running the pipeline manually might look as follows:

```bash
export SLDEA_WORKDIR=$HOME/sleda_workdir
python src/preprocessing.py        # creates annotations(1).csv–annotations(5).csv and feature_label.csv
python src/dialogue_pred.py        # trains and saves model.pkl
python src/dialogue_pred.py --model model.pkl --test test.csv --out preds.csv
python src/ESL_AddedExperinments.py experiments.csv  # optional
```

## Local Docker build

A minimal Docker setup is provided for testing the application locally and for deployment to HuggingFace Spaces. Build and run the image with:

```bash
docker build -t sleda-space .
docker run -p 7860:7860 sleda-space
```

This exposes a Gradio interface on http://localhost:7860.

## Deploying to HuggingFace Spaces

Create a new **Docker** Space on HuggingFace and link it to this repository. The Space builder uses `Dockerfile` in the repository root to install the dependencies, convert the notebooks to scripts and start `app.py`. Once built, the web interface offers three tabs and an optional **Extra Experiments** tab:

1. **Pre-processing** – upload your Excel or CSV files and run `preprocessing.py`
   to convert them into the internal format and produce `feature_label.csv`.
2. **Training** – upload `feature_label.csv` and run `dialogue_pred.py` to train
   a model. The script saves the resulting `model.pkl`.
3. **Application** – upload `model.pkl` together with one or more test CSV/XLSX
   files. `dialogue_pred.py` generates predictions in `preds_<uuid>.csv`.
4. **Extra Experiments** *(optional)* – upload an experiments CSV to run
   additional analysis in `ESL_AddedExperinments.ipynb`.

The container exposes port `7860`, which Spaces automatically forwards.

## License

This project is released under the MIT License.
