"""

Pipeline stage 1: raw .edf -> preprocessed .fif, per subject.

Финализированная предобработка:
проверка gain-ratio по T3, ресемплинг до 125 Гц, фильтрация, интерполяция плохих
каналов, перезадание референса, ICA. 

Директории:
    директория сырого субъекта:      <person>/{T-П,T-1..T-5}.edf
    директория подготовленного субъекта: <person>/{T-П,T-1..T-5}_raw.fif

Использование:
    python3 pipeline_data_preparation.py <raw_input_dir> <prepared_output_dir> [--t3]

Использование (оценка сырых субъектов напрямую против обученной модели):
    python3 pipeline_data_preparation.py --predict <model.joblib> <raw_input_dir> <output_csv>

"""

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import joblib
import mne
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prep_v2 import CHANNELS, preprocess_raw  # noqa: E402

mne.set_log_level("ERROR")

RAW_FILENAMES = ["T-П.edf"] + [f"T-{t}.edf" for t in range(1, 6)]


def prepare_subject(raw_dir: Path, out_dir: Path, mark_t3_bad: bool = False) -> list[Path]:
    """Preprocess every *.edf recording found directly under raw_dir into
    out_dir/<stem>_raw.fif. Returns the list of .fif files written."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for src in sorted(raw_dir.glob("*.edf")):
        dst = out_dir / f"{src.stem}_raw.fif"
        raw = mne.io.read_raw_edf(src, preload=True, verbose=False)
        raw.pick(CHANNELS)
        raw_clean, _log = preprocess_raw(raw, mark_t3_bad=mark_t3_bad)
        raw_clean.save(dst, overwrite=True, verbose=False)
        written.append(dst)
    return written


def find_raw_person_dirs(input_dir: Path) -> list[Path]:
    """Any directory containing at least one *.edf file."""
    input_dir = Path(input_dir)
    return sorted({p.parent for p in input_dir.rglob("*.edf")})


def predict(model, subject_dir: Path) -> float:
    """subject_dir holds a subject's raw .edf recordings (T-П.edf, T-1..T-5.edf).
    Preprocesses them to a scratch directory, then scores via feature_extraction.predict."""
    import pipeline_feature_extraction as fe

    subject_dir = Path(subject_dir)
    with tempfile.TemporaryDirectory(prefix="ptsd_prep_") as tmp:
        prepare_subject(subject_dir, Path(tmp), mark_t3_bad=False)
        return fe.predict(model, Path(tmp))


def run(model, input_dir: Path, output_csv: Path) -> None:
    """Score every raw-.edf subject directory under input_dir; writes
    output_csv with columns subject_id, ptsd_probability."""
    import csv

    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    dirs = find_raw_person_dirs(input_dir)

    rows = []
    for sdir in tqdm(dirs, desc="scoring raw subjects", unit="subject"):
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
    ap.add_argument("--predict", metavar="MODEL_JOBLIB", help="score raw subjects instead of preprocessing")
    ap.add_argument("--t3", action="store_true", help="preprocess_raw(mark_t3_bad=True)")
    ap.add_argument("input_dir")
    ap.add_argument("output")
    args = ap.parse_args()

    if args.predict:
        model = joblib.load(args.predict)
        run(model, Path(args.input_dir), Path(args.output))
        return

    src_root = Path(args.input_dir)
    dst_root = Path(args.output)
    person_dirs = find_raw_person_dirs(src_root)
    print(f"{len(person_dirs)} subjects found under {src_root}")

    n_ok, n_fail = 0, 0
    pbar = tqdm(person_dirs, desc="preprocessing subjects", unit="subject")
    for sdir in pbar:
        rel = sdir.relative_to(src_root)
        out_dir = dst_root / rel
        try:
            prepare_subject(sdir, out_dir, mark_t3_bad=args.t3)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {rel}: {e}")
            traceback.print_exc(limit=1)
        pbar.set_postfix(ok=n_ok, fail=n_fail)

    print(f"Finished. ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
