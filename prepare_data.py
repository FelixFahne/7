import os
import shutil
from pathlib import Path
import pandas as pd

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


def prepare_data_structure(data_dir: Path = DATA_DIR, force: bool = False):
    """Ensure the workspace contains the CSVs required by the notebooks."""
    if not force and all(p.exists() for p in ANNOTATIONS) and SAMPLE_DIR.exists():
        return
    _convert_excel_to_csv(data_dir)
    _populate_sample_dir(data_dir)


if __name__ == "__main__":
    prepare_data_structure()
