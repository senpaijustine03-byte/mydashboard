import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# Correct file path inside the Kaggle dataset
file_path = "Groceries_dataset.csv"

# Load dataset using the recommended method
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "heeraldedhia/groceries-dataset",
    path=file_path,
    pandas_kwargs={
        "encoding": "latin1",
        "sep": ",",
        "quotechar": '"',
        "on_bad_lines": "skip"  # skip malformed rows
    }
)

print("First 5 records:")
print(df.head())
