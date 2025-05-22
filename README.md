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
- `dialogue_pred.ipynb` and `ESL_AddedExperinments.ipynb` – main notebooks for preprocessing, training and evaluation.
- `prepare_data.py` – script that converts the Excel files under `SLDEA Data/` to the
  CSV structure expected by the notebooks.
  It is executed automatically when starting the app.

## Dataset

Only a small sample of the full SLDEA dataset is provided here. For complete access contact `rena.gao@unimelb.edu.au`. Place the files under `src/SLDEA Data/` so that the notebooks can locate them.

## Running the notebooks

The notebooks in `src/` contain the preprocessing steps and experiments described in the paper. They require CSV files named `annotations(1).csv` to `annotations(5).csv` and a folder `data_csv_sample/`. The helper script `prepare_data.py` generates these files from the Excel sheets under `SLDEA Data/` if they are missing.

You can run the notebooks directly in Jupyter or execute them from the command line with [Papermill](https://papermill.readthedocs.io/):

```bash
papermill dialogue_pred.ipynb dialogue_pred_out.ipynb
```

The Python scripts `dialogue_pred.py` and `ESL_AddedExperinments.py` are exports of these notebooks and can be invoked directly once the required packages are installed.

## Local Docker build

A minimal Docker setup is provided for testing the application locally and for deployment to HuggingFace Spaces. Build and run the image with:

```bash
docker build -t sleda-space .
docker run -p 7860:7860 sleda-space
```

This exposes a Gradio interface on http://localhost:7860.

## Deploying to HuggingFace Spaces

Create a new **Docker** Space on HuggingFace and link it to this repository. The Space builder uses `Dockerfile` in the repository root to install the dependencies, convert the notebooks to scripts and start `app.py`. Once built, the web interface offers three tabs:

1. **Pre-processing** – run `dialogue_pred.ipynb` on an uploaded CSV file.
2. **Training** – execute `ESL_AddedExperinments.ipynb`.
3. **Application** – apply a trained model to new data via `dialogue_pred.py`.

The container exposes port `7860`, which Spaces automatically forwards.

## License

This project is released under the MIT License.
