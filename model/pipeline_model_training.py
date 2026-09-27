"""
```
Pipeline stage 3: feature vectors -> trained classifier.

Загружает два ОТДЕЛЬНЫХ набора признаков, записанных pipeline_feature_extraction.py
(eeg/features.csv, behavioral/features.csv), и обучает рецепт "топ-15 расширенных
признаков + L2 LogReg" из REPORT_top15_ext_features_l2_logreg.md
(Imputer(median) -> StandardScaler -> LogisticRegression(C=0.05, penalty="l2",
class_weight="balanced")).

Флаг --use-behavioral управляет тем, добавляются ли к набору ЭЭГ-признаков
2 поведенческих (не-ЭЭГ) признака -- длительность активности/покоя -- перед обучением.

Поведенческие признаки -- задокументированный конфаундер, связанный с группой
и завершением задания (записи задания останавливаются по завершении теста Шульте;
длительность покоя также коррелирует с группой). Флаг --use-behavioral существует,
чтобы явно измерить этот риск, а не чтобы рекомендовать его к использованию
в продакшене.

Валидация: leave-one-subject-out (LOSO) -- каждый человек ровно один раз исключается
из обучения (модель обучается на всех остальных), что даёт одну честную out-of-fold
вероятность на человека. Это самый строгий уровень кросс-валидации на уровне субъекта,
доступный при столь малом числе субъектов с ПТСР (см. CV_REPORT_kfold_generalization.md
о том, почему одно разбиение 80/20 здесь не заслуживает доверия). Функция train()
записывает эти LOSO-вероятности в CSV (одна строка на человека/фолд) перед обучением
финальной модели на полном датасете.

Использование:
    python3 pipeline_model_training.py <features_dir> <model_out.joblib> <loso_predictions.csv> [--use-behavioral]

<features_dir> -- это output_dir, переданный в pipeline_feature_extraction.py
(должен содержать подпапки eeg/ и behavioral/, в каждой -- feature_names.txt
и features.csv).

Использование (оценка подготовленных субъектов напрямую против обученной модели):
    python3 pipeline_model_training.py --predict <model.joblib> <prepared_input_dir> <output_csv>
```
"""

import argparse
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import pipeline_feature_extraction as fe  # noqa: E402

PTSD_GROUP = "ПТСР"
C = 0.05


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, penalty="l2", class_weight="balanced", max_iter=5000),
    )


def _load_feature_set(features_dir: Path, subdir: str):
    names_path = features_dir / subdir / "feature_names.txt"
    csv_path = features_dir / subdir / "features.csv"
    with open(names_path) as f:
        names = [line.strip() for line in f if line.strip()]
    df = pd.read_csv(csv_path)
    return df, names


def load_features(features_dir: Path, use_behavioral: bool):
    """Returns (df, feature_names) with group/person plus the selected
    columns, EEG always included; the 2 behavioral columns joined in (on
    group+person) only when use_behavioral is True."""
    features_dir = Path(features_dir)
    df, eeg_names = _load_feature_set(features_dir, "eeg")
    feature_names = list(eeg_names)

    if use_behavioral:
        behav_df, behav_names = _load_feature_set(features_dir, "behavioral")
        df = df.merge(behav_df, on=["group", "person"], how="left", validate="one_to_one")
        feature_names += behav_names

    return df, feature_names


def loso_proba(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    idx_all = np.arange(n)
    proba = np.zeros(n)
    for i in tqdm(range(n), desc="LOSO folds", unit="fold"):
        tr = idx_all[idx_all != i]
        m = clone(make_model()).fit(X[tr], y[tr])
        proba[i] = m.predict_proba(X[i : i + 1])[:, 1][0]
    return proba


def print_metrics(label, y, pred):
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    print(f"  {label:22s} acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")


def best_f1_threshold(y, proba):
    thresholds = np.unique(proba)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def train(features_dir: Path, model_out: Path, loso_csv_out: Path, use_behavioral: bool = False) -> None:
    features_dir = Path(features_dir)
    df, feature_names = load_features(features_dir, use_behavioral)

    X = df[feature_names].to_numpy(dtype=np.float64)
    y = (df["group"] == PTSD_GROUP).astype(int).to_numpy()
    persons = df["person"].to_numpy()
    groups = df["group"].to_numpy()
    n = len(y)
    print(f"{n} persons ({y.sum()} PTSD) from {features_dir}, {len(feature_names)} features "
          f"(use_behavioral={use_behavioral})")
    if use_behavioral:
        print("  CAVEAT: duration features are a documented group/task-completion confound -- see CLAUDE.md")

    # ---- LOSO cross-validation: one fold per person ----
    proba = loso_proba(X, y)
    auc = roc_auc_score(y, proba)
    print(f"\n=== LOSO cross-validation ({n} folds) ===")
    print(f"ROC-AUC = {auc:.3f}")
    print_metrics("@0.5 (default)", y, (proba >= 0.5).astype(int))

    thr = best_f1_threshold(y, proba)
    print(f"\nChosen decision threshold (max F1 on LOSO OOF probabilities): {thr:.3f}")
    print_metrics(f"@{thr:.3f} (chosen)", y, (proba >= thr).astype(int))

    print("\nper-group mean predicted P(PTSD), LOSO out-of-fold:")
    for g in np.unique(groups):
        m = groups == g
        print(f"  {g:15s} n={m.sum():3d} mean_p={proba[m].mean():.3f}")

    loso_csv_out = Path(loso_csv_out)
    loso_csv_out.parent.mkdir(parents=True, exist_ok=True)
    loso_df = pd.DataFrame({
        "fold": np.arange(n),          # each fold == the one person held out
        "subject_id": persons,
        "group": groups,
        "true_label": y,
        "ptsd_probability": proba,
    })
    loso_df.to_csv(loso_csv_out, index=False)
    print(f"\nwrote LOSO out-of-fold predictions ({n} rows, one per subject/fold) -> {loso_csv_out}")

    # ---- final model: fit on every available person ----
    final_model = make_model().fit(X, y)
    final_proba = final_model.predict_proba(X)[:, 1]
    final_pred = (final_proba >= thr).astype(int)
    print("\n=== Final model, fit on the FULL dataset ===")
    print("In-sample fit below is a sanity check only (NOT a generalization")
    print("estimate -- the model has already seen every one of these labels).")
    print("The LOSO numbers above are the only trustworthy performance estimate.")
    print_metrics("in-sample @chosen thr", y, final_pred)
    print(f"  in-sample ROC-AUC = {roc_auc_score(y, final_proba):.3f}")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "use_behavioral": use_behavioral,
            "threshold": thr,
            "loso_auc": float(auc),
            "ptsd_group_label": PTSD_GROUP,
            "n_train_persons": int(n),
        },
        model_out,
    )
    print(f"\nSaved final model -> {model_out}")


def predict(model, subject_dir: Path) -> float:
    """subject_dir holds a subject's prepared .fif recordings. Returns P(PTSD) in [0, 1]."""
    return fe.predict(model, Path(subject_dir))


def run(model, input_dir: Path, output_csv: Path) -> None:
    """Score every prepared-.fif subject directory under input_dir; writes
    output_csv with columns subject_id, ptsd_probability."""
    fe.run(model, Path(input_dir), Path(output_csv))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", metavar="MODEL_JOBLIB", help="score prepared subjects instead of training")
    ap.add_argument("--use-behavioral", action="store_true",
                     help="train: include the 2 behavioral (duration) features alongside the 15 EEG features")
    ap.add_argument("input_dir_or_features_dir")
    ap.add_argument("output_or_model_out")
    ap.add_argument("loso_csv_out", nargs="?", help="required when training: where to write LOSO OOF predictions")
    args = ap.parse_args()

    if args.predict:
        model = joblib.load(args.predict)
        try:
            run(model, Path(args.input_dir_or_features_dir), Path(args.output_or_model_out))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return

    if not args.loso_csv_out:
        ap.error("training requires <loso_predictions.csv> as the third positional argument")

    train(
        Path(args.input_dir_or_features_dir),
        Path(args.output_or_model_out),
        Path(args.loso_csv_out),
        use_behavioral=args.use_behavioral,
    )


if __name__ == "__main__":
    main()
