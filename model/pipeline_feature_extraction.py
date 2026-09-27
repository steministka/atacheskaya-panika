"""
```
Pipeline stage 2: preprocessed .fif -> feature vectors, per subject.

Вычисляет два набора признаков на человека и хранит их в ОТДЕЛЬНЫХ файлах
(никогда не объединяются в одну строку/вектор на диске), чтобы нижележащий
потребитель мог независимо от ЭЭГ-набора подключать или не подключать
поведенческий набор:

  - ЭЭГ-признаки (15-мерные): топ-15 расширенных спектральных признаков
    и признаков сложности из REPORT_top15_ext_features_l2_logreg.md --
    относительная мощность по диапазонам + сложность DFA/Хигучи. Использует
    список TOP15 и логику извлечения из feature_extraction.py, чтобы числа
    в точности совпадали с тем отчётом.
  - Поведенческие признаки (2-мерные): длительность активности (задания)
    и длительность покоя, в секундах -- использует логику из
    feature_extraction_behavioral.py.


Структура выходных файлов, внутри <output_dir>:
    eeg/feature_names.txt, eeg/features.csv, eeg/<group>/<person>.npy
    behavioral/feature_names.txt, behavioral/features.csv, behavioral/<group>/<person>.npy

Использование (пакетное извлечение по подготовленному дереву):
    python3 pipeline_feature_extraction.py <prepared_input_dir> <output_dir> [n_jobs]

Использование (оценка подготовленных субъектов напрямую против обученной модели):
    python3 pipeline_feature_extraction.py --predict <model.joblib> <prepared_input_dir> <output_csv>
```
"""

import argparse
import csv
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import mne
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

sys.path.insert(0, str(Path(__file__).parent))
import compute_extended_features as cef  # noqa: E402 -- reuse dfa()/higuchi_fd()
import feature_extraction as fe  # noqa: E402 -- reuse TOP15, extract_parts, load_full

REST_FILE = cef.REST_FILE
N_TASKS = cef.N_TASKS
MIN_S = fe.MIN_S

EEG_FEATURES = fe.TOP15
BEHAVIORAL_FEATURES = ["behavioral__activity_duration_s", "behavioral__rest_duration_s"]


def find_person_dirs(input_dir: Path) -> list[Path]:
    return [Path(p) for p in fe.find_person_dirs(str(input_dir))]


def extract_eeg_features(subject_dir: Path) -> np.ndarray:
    """15-dim TOP15 vector for one subject (NaN where a recording is missing)."""
    subject_dir = Path(subject_dir)
    feats = {}

    rest_p = subject_dir / REST_FILE
    if rest_p.exists():
        x, fs = fe.load_full(str(rest_p))
        if x.shape[1] > MIN_S * fs:
            for k, v in fe.extract_parts(x, fs, "rest").items():
                feats[f"rest__{k}"] = v

    task_parts = []
    for t in range(1, N_TASKS + 1):
        p = subject_dir / f"T-{t}_raw.fif"
        if not p.exists():
            continue
        x, fs = fe.load_full(str(p))
        if x.shape[1] > MIN_S * fs:
            task_parts.append(fe.extract_parts(x, fs, "task"))
    if task_parts:
        for k in task_parts[0]:
            feats[f"task__{k}"] = float(np.mean([tp[k] for tp in task_parts]))

    return np.array([feats.get(name, np.nan) for name in EEG_FEATURES], dtype=np.float64)


def extract_behavioral_features(subject_dir: Path) -> np.ndarray:
    """2-dim [activity_duration_s, rest_duration_s] vector for one subject."""
    subject_dir = Path(subject_dir)

    rest_dur = np.nan
    rest_p = subject_dir / REST_FILE
    if rest_p.exists():
        x, fs = fe.load_full(str(rest_p))
        rest_dur = x.shape[1] / fs

    task_durs = []
    for t in range(1, N_TASKS + 1):
        p = subject_dir / f"T-{t}_raw.fif"
        if not p.exists():
            continue
        x, fs = fe.load_full(str(p))
        task_durs.append(x.shape[1] / fs)

    activity_dur = float(np.mean(task_durs)) if task_durs else np.nan
    return np.array([activity_dur, rest_dur], dtype=np.float64)


def _process_person(sdir: Path):
    group = sdir.parent.name
    person = sdir.name
    eeg_vec = extract_eeg_features(sdir)
    behav_vec = extract_behavioral_features(sdir)
    return group, person, eeg_vec, behav_vec


def _write_feature_set(output_dir: Path, subdir: str, names: list[str], rows: list[tuple]):
    base = output_dir / subdir
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "feature_names.txt", "w") as f:
        f.write("\n".join(names) + "\n")
    with open(base / "features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "person"] + names)
        for group, person, vec in rows:
            person_dir = base / group
            person_dir.mkdir(parents=True, exist_ok=True)
            np.save(person_dir / f"{person}.npy", vec)
            w.writerow([group, person] + vec.tolist())


def extract_all(input_dir: Path, output_dir: Path, n_jobs: int = 10) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    dirs = find_person_dirs(input_dir)
    print(f"{len(dirs)} subjects found under {input_dir}, {n_jobs} workers")

    eeg_rows, behav_rows = [], []
    with ProcessPoolExecutor(n_jobs) as ex:
        results = ex.map(_process_person, dirs)
        for group, person, eeg_vec, behav_vec in tqdm(results, total=len(dirs), desc="extracting features", unit="subject"):
            eeg_rows.append((group, person, eeg_vec))
            behav_rows.append((group, person, behav_vec))

    _write_feature_set(output_dir, "eeg", EEG_FEATURES, eeg_rows)
    _write_feature_set(output_dir, "behavioral", BEHAVIORAL_FEATURES, behav_rows)
    print(f"saved {len(eeg_rows)} subjects: eeg -> {output_dir/'eeg'}, behavioral -> {output_dir/'behavioral'}")


def build_feature_row(model, subject_dir: Path) -> dict:
    """EEG features, plus behavioral features if the model was trained with
    them (model['use_behavioral']), keyed by feature name."""
    row = dict(zip(EEG_FEATURES, extract_eeg_features(subject_dir)))
    if model.get("use_behavioral"):
        row.update(dict(zip(BEHAVIORAL_FEATURES, extract_behavioral_features(subject_dir))))
    return row


def apply_model(model, row: dict) -> float:
    """Builds X in model['feature_names'] order and returns P(PTSD)."""
    x = np.array([[row.get(name, np.nan) for name in model["feature_names"]]], dtype=np.float64)
    return float(model["model"].predict_proba(x)[:, 1][0])


def predict(model, subject_dir: Path) -> float:
    """subject_dir holds a subject's prepared .fif recordings. Returns P(PTSD) in [0, 1]."""
    row = build_feature_row(model, Path(subject_dir))
    return apply_model(model, row)


def run(model, input_dir: Path, output_csv: Path) -> None:
    """Score every prepared-.fif subject directory under input_dir; writes
    output_csv with columns subject_id, ptsd_probability."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    dirs = find_person_dirs(input_dir)

    rows = []
    for sdir in tqdm(dirs, desc="scoring prepared subjects", unit="subject"):
        subject_id = sdir.name
        try:
            proba = predict(model, sdir)
        except Exception as e:
            print(f"[FAIL] {subject_id}: {e}", file=sys.stderr)
            traceback.print_exc(limit=1)
            continue
        rows.append((subject_id, proba))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "ptsd_probability"])
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions -> {output_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", metavar="MODEL_JOBLIB", help="score prepared subjects instead of extracting")
    ap.add_argument("input_dir")
    ap.add_argument("output")
    ap.add_argument("n_jobs", nargs="?", type=int, default=10)
    args = ap.parse_args()

    if args.predict:
        model = joblib.load(args.predict)
        run(model, Path(args.input_dir), Path(args.output))
        return

    extract_all(Path(args.input_dir), Path(args.output), args.n_jobs)


if __name__ == "__main__":
    main()
