# SLEDA Interactive Metrics

SLEDA is a three-level framework for evaluating English Second Language (ESL) conversation dialogues. This repository contains the original notebooks, supporting data and a small Docker setup for running the tool in a HuggingFace Space.

## Repository layout

- `SLDEA Data/` – example ESL dialogues in XLSX format.
- `feature_label.csv` – features extracted from the annotated datasets.
- `2024ACLESLMainCodes_Results/` – sample results from the accompanying paper.
- `ESL_AddedExperinments.ipynb` – optional notebook for extra experiments.
- `Dockerfile`, `requirements.txt` and `app.py` – files used when deploying to HuggingFace Spaces.

## Dataset

Only a small sample of the full SLDEA dataset is provided here. For complete access contact `rena.gao@unimelb.edu.au`. Place the files under `src/SLDEA Data/` so that the notebooks can locate them.

## Running the notebooks

You can execute `ESL_AddedExperinments.ipynb` with [Papermill](https://papermill.readthedocs.io/):

```bash
papermill ESL_AddedExperinments.ipynb out.ipynb
```

The Python scripts `dialogue_pred.py` and `ESL_AddedExperinments.py` implement the same functionality without requiring Jupyter.

## Local Docker build

A minimal Docker setup is provided for testing the application locally and for deployment to HuggingFace Spaces. Build and run the image with:

```bash
docker build -t sleda-space .
docker run -p 7860:7860 sleda-space
```

This exposes a Gradio interface on http://localhost:7860.

## Deploying to HuggingFace Spaces

Create a new **Docker** Space on HuggingFace and link it to this repository. The Space builder uses the repository root `Dockerfile` to install the dependencies, convert the notebooks to scripts and start `app.py`. Once built, the web interface offers three tabs and an optional **Extra Experiments** tab:

1. **Pre-processing** – run `preprocessing.py` (or `notebooks/a.ipynb`) on an uploaded CSV file.
2. **Training** – run `dialogue_pred.py` on `feature_label.csv` to train a model.
3. **Application** – apply a trained model to new data via `dialogue_pred.py`.
4. **Extra Experiments** *(optional)* – explore additional analysis in `ESL_AddedExperinments.ipynb`.

The container exposes port `7860`, which Spaces automatically forwards.

## License

This project is released under the MIT License.
