import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Workspace for generated files
WORKDIR = Path(os.getenv("SLDEA_WORKDIR", "/tmp/space"))

# Paths to the five annotation CSV files and the sample directory
ANNOTATIONS = [WORKDIR / f"annotations({i}).csv" for i in range(1, 6)]
SAMPLE_DIR = WORKDIR / "data_csv_sample"

# Basic columns expected in the input files
COLS = ["id", "label", "labellevel", "span"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has the required columns in a normalised order."""
    lower_map = {c.lower(): c for c in df.columns}
    if all(c in lower_map for c in COLS):
        df = df[[lower_map[c] for c in COLS]]
        df.columns = COLS
        return df
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS[: df.shape[1]]
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS]


def _convert_excel_to_csv(data_dir: Path) -> None:
    """Convert uploaded Excel/CSV files into ``annotations`` CSVs."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    xlsx_files = sorted(data_dir.glob("*.xlsx"))
    csv_files = sorted(data_dir.glob("*.csv"))
    frames = []
    for p in xlsx_files:
        df = pd.read_excel(p)
        frames.append(_normalize_columns(df))
    for p in csv_files:
        df = pd.read_csv(p)
        frames.append(_normalize_columns(df))
    if not frames:
        raise FileNotFoundError(f"No .xlsx or .csv files found in {data_dir}")
    part_size = (len(frames) + 4) // 5
    for idx, ann_path in enumerate(ANNOTATIONS):
        start = idx * part_size
        end = start + part_size
        subset = frames[start:end]
        if not subset:
            break
        pd.concat(subset).to_csv(ann_path, index=False)


def _populate_sample_dir(data_dir: Path) -> None:
    """Copy uploaded files into ``data_csv_sample`` as CSVs."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for csv in data_dir.glob("*.csv"):
        df = pd.read_csv(csv)
        df = _normalize_columns(df)
        (SAMPLE_DIR / csv.name).write_text(df.to_csv(index=False))
    for xlsx in data_dir.glob("*.xlsx"):
        df = pd.read_excel(xlsx)
        df = _normalize_columns(df)
        (SAMPLE_DIR / f"{xlsx.stem}.csv").write_text(df.to_csv(index=False))


def _load_converted_csv() -> pd.DataFrame:
    """Return a DataFrame with all annotation CSVs merged."""
    frames = [pd.read_csv(p) for p in ANNOTATIONS if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assign_tone(row: pd.Series) -> str:
    """Assign a conversational tone label based on label counts."""
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


def _perform_analysis(df: pd.DataFrame) -> None:
    """Generate tone histogram and run minimal regression examples."""
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
        plt.savefig(WORKDIR / "tone_distribution.png")
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
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        LinearRegression().fit(X_train, y_train["OverallToneChoice"])
        LinearRegression().fit(X_train, y_train["TopicExtension"])


def prepare_data_structure(data_dir: Path, force: bool = False) -> Path:
    """Convert ``data_dir`` and produce ``feature_label.csv`` in ``WORKDIR``."""
    if force or not all(p.exists() for p in ANNOTATIONS) or not SAMPLE_DIR.exists():
        _convert_excel_to_csv(data_dir)
        _populate_sample_dir(data_dir)
    df = _load_converted_csv()
    _perform_analysis(df)
    out_csv = WORKDIR / "feature_label.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data and generate feature_label.csv")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Directory containing Excel or CSV files",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild annotations")
    args = parser.parse_args()
    prepare_data_structure(args.data_dir, args.force)
