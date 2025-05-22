import os
import shutil
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

SRC = Path(__file__).parent / "src"
DATA_DIR = SRC / "SLDEA Data"

# Directory used for generated CSV files. On platforms where the repository
# itself is read-only (e.g. HuggingFace Spaces), we fall back to a writable
# location under ``/tmp``. The path can be overridden via the ``SLDEA_WORKDIR``
# environment variable so that the notebooks and scripts can locate their
# input/output in a consistent place.
WORKDIR = Path(os.getenv("SLDEA_WORKDIR", "/tmp/space"))

ANNOTATIONS = [WORKDIR / f"annotations({i}).csv" for i in range(1, 6)]
SAMPLE_DIR = WORKDIR / "data_csv_sample"


def _convert_excel_to_csv(data_dir: Path):
    """Convert uploaded Excel/CSV files to the 5 ``annotations`` CSV files."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    xlsx_files = sorted(data_dir.glob("*.xlsx"))
    csv_files = sorted(data_dir.glob("*.csv"))
    frames = []
    for p in xlsx_files:
        frames.append(pd.read_excel(p))
    for p in csv_files:
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No .xlsx or .csv files found in {data_dir}")

    part_size = (len(frames) + 4) // 5
    for idx, ann_path in enumerate(ANNOTATIONS):
        start = idx * part_size
        end = start + part_size
        subset = frames[start:end]
        if not subset:
            break
        df = pd.concat(subset)
        ann_path.write_text(df.to_csv(index=False))


def _populate_sample_dir(data_dir: Path):
    """Copy CSV files (or converted XLSX files) into ``data_csv_sample``."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for csv in data_dir.glob("*.csv"):
        shutil.copy(csv, SAMPLE_DIR / csv.name)
    for xlsx in data_dir.glob("*.xlsx"):
        df = pd.read_excel(xlsx)
        (SAMPLE_DIR / f"{xlsx.stem}.csv").write_text(df.to_csv(index=False))


def _load_converted_csv() -> pd.DataFrame:
    """Return a DataFrame concatenating all annotation CSVs."""
    frames = []
    for p in ANNOTATIONS:
        if p.exists():
            frames.append(pd.read_csv(p))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def assign_tone(row: pd.Series) -> str:
    """Assign a conversational tone based on label counts."""
    if (
        row.get("backchannels", 0) > 0
        or row.get("code-switching for communicative purposes", 0) > 0
        or row.get("collaborative finishes", 0) > 0
    ):
        return "Informal"
    if (
        row.get("subordinate clauses", 0) > 0
        or row.get("impersonal subject + non-factive verb + NP", 0) > 0
    ):
        return "Formal"
    return "Neutral"


def _perform_analysis(df: pd.DataFrame, show_plot: bool = False) -> None:
    """Replicate analysis steps from the original tool if columns are present.

    Parameters
    ----------
    df:
        DataFrame containing the annotated dialogue data.
    show_plot:
        If ``True`` display the tone distribution plot instead of saving it
        to ``tone_distribution.png`` under ``WORKDIR``.
    """
    if df.empty:
        return

    summary = (
        df.groupby("dialogue_segment").sum(numeric_only=True)
        if "dialogue_segment" in df.columns
        else pd.DataFrame()
    )

    if not summary.empty:
        summary["Tone"] = summary.apply(assign_tone, axis=1)
        tone_assignments = summary["Tone"].value_counts()

        plt.figure(figsize=(8, 5))
        tone_assignments.plot(kind="bar")
        plt.title("Distribution of Assigned Tones Across Dialogue Segments")
        plt.xlabel("Tone")
        plt.ylabel("Number of Segments")
        plt.xticks(rotation=0)
        if show_plot:
            plt.show()
        else:
            fig_path = WORKDIR / "tone_distribution.png"
            plt.savefig(fig_path)
        plt.close()

    req_cols = {
        "dialogue_id",
        "token_label_type1",
        "token_label_type2",
        "OverallToneChoice",
        "TopicExtension",
    }
    if req_cols.issubset(df.columns):
        features = df.groupby("dialogue_id").agg(
            {
                "token_label_type1": "sum",
                "token_label_type2": "sum",
            }
        )

        dialogue_labels = df.groupby("dialogue_id").agg(
            {
                "OverallToneChoice": "first",
                "TopicExtension": "first",
            }
        )

        data_for_regression = features.join(dialogue_labels)
        X = data_for_regression.drop(["OverallToneChoice", "TopicExtension"], axis=1)
        y = data_for_regression[["OverallToneChoice", "TopicExtension"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        LinearRegression().fit(X_train, y_train["OverallToneChoice"])
        LinearRegression().fit(X_train, y_train["TopicExtension"])


def prepare_data_structure(
    data_dir: Path = DATA_DIR,
    force: bool = False,
    show_plot: bool = False,
):
    """Ensure the workspace contains the CSVs required by the notebooks.

    Parameters
    ----------
    data_dir:
        Directory containing uploaded Excel/CSV files.
    force:
        Recreate the CSVs even if they already exist.
    show_plot:
        Forwarded to :func:`_perform_analysis` to display the plot instead
        of saving it.
    """
    if not force and all(p.exists() for p in ANNOTATIONS) and SAMPLE_DIR.exists():
        return
    _convert_excel_to_csv(data_dir)
    _populate_sample_dir(data_dir)
    df = _load_converted_csv()
    _perform_analysis(df, show_plot=show_plot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert SLDEA data and perform analysis")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the tone distribution plot instead of saving it",
    )
    args = parser.parse_args()

    prepare_data_structure(show_plot=args.show_plot)
