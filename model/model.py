"""
PTSD probability prediction model.

Signatures of following functions should NOT be changed:
    load(path: Path) -> model
    predict(model, subject_dir: Path) -> float

Do NOT modify following functions:
    run(model, input_dir: Path, output_path: Path) -> None

Other functions may be changed as you wish.
"""
import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent

def train(*args: Any, **kwargs: Any):
    """Trains model."""
    ...

def predict(model, subject_dir: Path) -> float:
    """Predicts PTSD probability of the subject."""
    ...

def run(model, input_dir: Path , output_path: Path) -> None:
    """Runs model over all subjects in `input_dir` and saves results to `output_path`."""
    predictions = []
    for subject_dir in input_dir.iterdir():
        subject_id = subject_dir.name
        ptsd_probability = predict(model, subject_dir)
        predictions.append((subject_id, ptsd_probability))

    predictions.sort(key=lambda x: x[0])
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "ptsd_probability"])
        writer.writerows(predictions)

def save(model, path: Path):
    """Saves model to disk."""
    ...

def load(path: Path):
    """Loads model from disk."""
    ...
