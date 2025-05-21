# 2024InteractiveMetrics

SLEDA is a three-level framework for evaluating dialogue quality and dialogue annotations for English Second Language (ESL)conversation dialogue.

It was created for the SLEDA project:

For more details, please read: Interaction Matters: A Three-Level Multi-class English Second Language Conversation Dialogue through Interactive Based Metrics

Dialogues labelled for three levels (from the above paper) can be found in SLDEA data.

Features exacted from the annotated datasets can be found in feature_label.csv. 

Python script data/explore_data.py provides an example of interfacing with the data.


### Dataset

- `dataset/SLDEA` - Sample Dataset: Full Access Contact Via: rena.gao@unimelb.edu.au
- Dataset Viewing
- To run the notebooks for examining the datasets, please follow the procedures listed below:

- Download the dataset from the folder.
- Put the data into dataset/SLDEA and extract sample.zip.
- To view the data, one may use preprocessing.ipynb for viewing the examples.

### Notebooks

- `notebooks/a.ipynb` - Notebook for preprocessing
- `notebooks/b.ipynb` - Notebook for main experiments
- `notebooks/c.ipync` - Notebook for added experiments 

### Figures

- `figures/` - Contains all figures used for this project

### Utils

- `utils/` - Contains all utility functions for this project

### Reports

- `reports/` - Generated analysis for Arvix paper

### Running in a HuggingFace Space

The repository already contains a minimal `space/` folder with a `Dockerfile`,
`requirements.txt` and `app.py`. These files allow the project to be deployed as
an interactive Space on Hugging Face. The Docker image installs the Python
dependencies, converts all Jupyter notebooks inside `src/` to Python scripts and
starts the Gradio interface defined in `space/app.py` on port `7860`.

To test the Space locally you can build and run the Docker image:

```bash
cd space
docker build -t sleda-space .
docker run -p 7860:7860 sleda-space
```

This will launch the same application that Hugging Face uses when creating a
Space from this repository.



