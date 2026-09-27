"""
Extract the top-15 features selected in REPORT_top15_ext_features_l2_logreg.md
for every person under a neurodiagnostics-train-prep-style directory tree.

Only the 15 columns from that report are computed (relative band power +
DFA/Higuchi complexity, a subset of the family in compute_extended_features.py)
using the FULL recording length -- no REST_WIN_S/TASK_WIN_S crop, matching
compute_extended_features_fullrec.py's convention rather than the fixed-window
one. Task recordings T-1..T-5 are averaged into a single "task" vector per
person, as in the training pipeline.

Reuses the DFA and Higuchi-FD implementations from compute_extended_features.py
so results match the report's numbers exactly; relative band power is
recomputed locally (same Welch/band definitions) since only a few of its many
outputs are needed here.

Usage:
    python3 feature_extraction.py <input_dir> <output_dir> [n_jobs]

<input_dir> must have the same <group>/<person>/{T-П,T-1..T-5}_raw.fif layout
as neurodiagnostics-train-prep/ (group folders are optional -- any directory
containing a T-П_raw.fif and/or T-1..T-5_raw.fif is treated as one person).

Output, under <output_dir>:
    feature_names.txt                       the 15 column names, in order
    features.csv                            one row per person: group,person,<15 cols>
    <group>/<person>.npy                    that person's 15-dim float64 vector (NaN where unavailable)
"""

import argparse
import csv
import glob
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
from scipy.signal import welch

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

import compute_extended_features as cef  # reuse dfa() / higuchi_fd() implementations

CHANNELS = cef.CHANNELS
CI = cef.CI
BANDS = cef.BANDS
TOTAL = cef.TOTAL
REST_FILE = cef.REST_FILE
N_TASKS = cef.N_TASKS
MIN_S = 5.0  # skip pathologically short recordings (DFA/Higuchi need some minimum)

TOP15 = [
    "rest__rp__beta_T4",
    "rest__nonlin__dfa_T4",
    "task__nonlin__dfa_Fp2",
    "task__nonlin__dfa_O1",
    "task__nonlin__dfa_T3",
    "rest__rp__beta_O1",
    "task__rp__beta_O2",
    "rest__rp__beta_Fp1",
    "rest__rp__delta_T4",
    "task__rp__theta_O2",
    "task__rp__delta_Fp2",
    "task__rp__delta_Fp1",
    "task__nonlin__higuchi_Fp1",
    "task__rp__gamma_O2",
    "task__nonlin__higuchi_Fp2",
]

# per-recording-type, which (band -> channels) / dfa channels / higuchi channels feed TOP15
RP_NEEDED = {
    "rest": {"beta": {"T4", "O1", "Fp1"}, "delta": {"T4"}},
    "task": {"beta": {"O2"}, "delta": {"Fp2", "Fp1"}, "theta": {"O2"}, "gamma": {"O2"}},
}
DFA_NEEDED = {"rest": {"T4"}, "task": {"Fp2", "O1", "T3"}}
HIGUCHI_NEEDED = {"rest": set(), "task": {"Fp1", "Fp2"}}


def relative_band_power(x, fs, bands):
    """rp[band][channel] = band power / total(1-40Hz) power, Welch PSD."""
    f, p = welch(x, fs=fs, nperseg=min(int(2 * fs), x.shape[-1]), axis=-1)
    tm = (f >= TOTAL[0]) & (f <= TOTAL[1])
    total = p[:, tm].sum(1) + 1e-12
    out = {}
    for b in bands:
        lo, hi = BANDS[b]
        out[b] = p[:, (f >= lo) & (f < hi)].sum(1) / total
    return out


def extract_parts(x, fs, kind):
    """Returns unprefixed feature dict (e.g. "rp__beta_T4") for one recording."""
    out = {}
    rp_bands = RP_NEEDED[kind]
    if rp_bands:
        rp = relative_band_power(x, fs, rp_bands.keys())
        for b, chans in rp_bands.items():
            for c in chans:
                out[f"rp__{b}_{c}"] = rp[b][CI[c]]
    for c in DFA_NEEDED[kind]:
        out[f"nonlin__dfa_{c}"] = cef.dfa(x[CI[c]])
    for c in HIGUCHI_NEEDED[kind]:
        out[f"nonlin__higuchi_{c}"] = cef.higuchi_fd(x[CI[c]])
    return out


def load_full(path):
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    fs = raw.info["sfreq"]
    x = raw.get_data(picks=CHANNELS) * 1e6
    return x, fs


def process_person(sdir):
    group = os.path.basename(os.path.dirname(sdir))
    person = os.path.basename(sdir)
    feats = {}

    rest_p = os.path.join(sdir, REST_FILE)
    if os.path.exists(rest_p):
        x, fs = load_full(rest_p)
        if x.shape[1] > MIN_S * fs:
            for k, v in extract_parts(x, fs, "rest").items():
                feats[f"rest__{k}"] = v

    task_parts = []
    for t in range(1, N_TASKS + 1):
        p = os.path.join(sdir, f"T-{t}_raw.fif")
        if not os.path.exists(p):
            continue
        x, fs = load_full(p)
        if x.shape[1] > MIN_S * fs:
            task_parts.append(extract_parts(x, fs, "task"))
    if task_parts:
        for k in task_parts[0]:
            feats[f"task__{k}"] = float(np.mean([tp[k] for tp in task_parts]))

    vec = np.array([feats.get(name, np.nan) for name in TOP15], dtype=np.float64)
    return group, person, vec


def find_person_dirs(input_dir):
    """Any directory (possibly under a group subfolder) with a rest/task fif file."""
    dirs = []
    for root, _, files in os.walk(input_dir):
        if any(f == REST_FILE or (f.startswith("T-") and f.endswith("_raw.fif")) for f in files):
            dirs.append(root)
    return sorted(dirs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="root of the <group>/<person>/*_raw.fif tree")
    ap.add_argument("output_dir", help="where to write feature_names.txt, features.csv, <group>/<person>.npy")
    ap.add_argument("n_jobs", nargs="?", type=int, default=10)
    args = ap.parse_args()

    dirs = find_person_dirs(args.input_dir)
    print(f"{len(dirs)} subjects found under {args.input_dir}, {args.n_jobs} workers")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(TOP15) + "\n")

    rows = []
    with ProcessPoolExecutor(args.n_jobs) as ex:
        for i, (group, person, vec) in enumerate(ex.map(process_person, dirs)):
            person_dir = os.path.join(args.output_dir, group)
            os.makedirs(person_dir, exist_ok=True)
            np.save(os.path.join(person_dir, f"{person}.npy"), vec)
            rows.append([group, person] + vec.tolist())
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(dirs)}", flush=True)

    with open(os.path.join(args.output_dir, "features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "person"] + TOP15)
        w.writerows(rows)

    print(f"saved {len(rows)} person feature vectors under {args.output_dir}")


if __name__ == "__main__":
    main()
