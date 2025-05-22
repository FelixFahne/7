import os
import shutil
from pathlib import Path
import pandas as pd

SRC = Path(__file__).parent / "src"
DATA_DIR = SRC / "SLDEA Data"

ANNOTATIONS = [SRC / f"annotations({i}).csv" for i in range(1, 6)]
SAMPLE_DIR = SRC / "data_csv_sample"


def _convert_excel_to_csv():
    xlsx_files = sorted(DATA_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {DATA_DIR}")
    # Partition files into 5 roughly equal groups
    part_size = (len(xlsx_files) + 4) // 5
    for idx, ann_path in enumerate(ANNOTATIONS):
        start = idx * part_size
        end = start + part_size
        subset = xlsx_files[start:end]
        if not subset:
            break
        df = pd.concat(pd.read_excel(p) for p in subset)
        ann_path.write_text(df.to_csv(index=False))


def _populate_sample_dir():
    SAMPLE_DIR.mkdir(exist_ok=True)
    for csv in DATA_DIR.glob("*.csv"):
        shutil.copy(csv, SAMPLE_DIR / csv.name)


def prepare_data_structure():
    if all(p.exists() for p in ANNOTATIONS) and SAMPLE_DIR.exists():
        return
    _convert_excel_to_csv()
    _populate_sample_dir()


if __name__ == "__main__":
    prepare_data_structure()
