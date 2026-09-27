"""
Extended per-person feature set from table_eeg_features_dry.csv, plus
Riemannian covariance matrices and Schulte-task behavioral timing.

Reads neurodiagnostics-train-prep/<group>/<person>/{T-П,T-1..T-5}_raw.fif
(all 125 Hz). For each recording computes a flat vector of scalar features
(named "<family>__<detail>") and a stack of 6x6 channel covariance matrices
(one per band in COV_BANDS). Task recordings T-1..T-5 are averaged into one
"task" vector (covariances: log-Euclidean mean).

Per person, stored in reve_embs/ext_features.npz:
    persons, groups                       (n,)
    rest_feats, task_feats                (n, n_feat)   NaN where unavailable
    feat_names                            (n_feat,)
    rest_cov, task_cov                    (n, n_bands, 6, 6)
    rest_seg_feats                        (n, N_SEGMENTS, n_feat)
    rest_seg_cov                          (n, N_SEGMENTS, n_bands, 6, 6)

Recording duration is deliberately NOT used: task recordings stop when the
Schulte table is solved, so length is a behavioral proxy. Every task recording
is cropped to its first TASK_WIN_S seconds (shorter ones are skipped) and rest
to REST_WIN_S (rest length also correlates with group), and length-dependent estimators (DFA scales, SampEn, LZ) use
fixed windows, so no feature can encode how long a recording was.

Usage: python3 compute_extended_features.py [n_jobs]
"""

import glob
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
from scipy.signal import butter, csd, sosfiltfilt, welch

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# PREP_ROOT / EXT_PATH env vars select another preprocessing version
DATA_ROOT = os.environ.get("PREP_ROOT", "neurodiagnostics-train-prep")
OUT_PATH = os.environ.get("EXT_PATH", "reve_embs/ext_features.npz")
GROUPS = ["Норма", "ПТСР", "Соматоформные"]
CHANNELS = ["Fp1", "Fp2", "T3", "T4", "O1", "O2"]
CI = {c: i for i, c in enumerate(CHANNELS)}
PAIRS = [("Fp1", "Fp2"), ("T3", "T4"), ("O1", "O2")]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
TOTAL = (1, 40)
CONN_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
COV_BANDS = {"broad": (1, 40), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
N_SEGMENTS = 5
N_TASKS = 5
REST_FILE = "T-П_raw.fif"
TASK_WIN_S = 20.0
REST_WIN_S = 30.0  # shortest rest is 32 s; 5 of the 7 short (<55 s) rests are PTSD


# ---------------------------------------------------------------- spectral
def spectral_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=min(int(2 * fs), x.shape[-1]), axis=-1)
    tm = (f >= TOTAL[0]) & (f <= TOTAL[1])
    total = p[:, tm].sum(1) + 1e-12
    rp = {b: p[:, (f >= lo) & (f < hi)].sum(1) / total for b, (lo, hi) in BANDS.items()}
    out = {}
    for b in BANDS:
        for c in CHANNELS:
            out[f"rp__{b}_{c}"] = rp[b][CI[c]]
    eps = 1e-12
    tbr, tar = rp["theta"] / (rp["beta"] + eps), rp["theta"] / (rp["alpha"] + eps)
    sfa = (rp["delta"] + rp["theta"]) / (rp["alpha"] + eps)
    for c in CHANNELS:
        out[f"ratio__logTBR_{c}"] = np.log(tbr[CI[c]] + eps)
        out[f"ratio__logTAR_{c}"] = np.log(tar[CI[c]] + eps)
        out[f"ratio__logSFA_{c}"] = np.log(sfa[CI[c]] + eps)
    # spectral entropy (normalized to [0,1]) and SE95 on 1-40 Hz
    pt = p[:, tm]
    q = pt / pt.sum(1, keepdims=True)
    sen = -(q * np.log2(q + eps)).sum(1) / np.log2(q.shape[1])
    cdf = np.cumsum(pt, 1) / pt.sum(1, keepdims=True)
    se95 = f[tm][np.argmax(cdf >= 0.95, axis=1)]
    # IAF: argmax in 7-13 Hz, NaN if at edge (no peak)
    am = (f >= 7) & (f <= 13)
    k = np.argmax(p[:, am], 1)
    iaf = np.where((k > 0) & (k < am.sum() - 1), f[am][k], np.nan)
    # aperiodic fit 2-30 Hz excluding alpha peak
    fm = (f >= 2) & (f <= 30) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[fm])
    for c in CHANNELS:
        i = CI[c]
        slope, off = np.polyfit(lf, np.log10(p[i, fm] + 1e-24), 1)
        out[f"aper__chi_{c}"] = -slope
        out[f"aper__offset_{c}"] = off
        out[f"spec__entropy_{c}"] = sen[i]
        out[f"spec__se95_{c}"] = se95[i]
        out[f"spec__iaf_{c}"] = iaf[i]
    # ln asymmetries (right - left)
    for b in BANDS:
        for l, r in PAIRS:
            out[f"asym__{b}_{r}-{l}"] = np.log(rp[b][CI[r]] + eps) - np.log(rp[b][CI[l]] + eps)
    # DEPD profile summary per pair
    lm = (f >= 1) & (f <= 40)
    for l, r in PAIRS:
        d = np.log10(p[CI[r], lm] + 1e-24) - np.log10(p[CI[l], lm] + 1e-24)
        out[f"asym__depd_mean_{r}-{l}"] = d.mean()
        out[f"asym__depd_std_{r}-{l}"] = d.std()
    return out


def wpli_feats(x, fs):
    nper = min(int(2 * fs), x.shape[-1])
    out = {}
    for a in range(6):
        for b in range(a + 1, 6):
            f, cxy = csd(x[a], x[b], fs=fs, nperseg=nper)
            im = np.imag(cxy)
            for band, (lo, hi) in CONN_BANDS.items():
                m = (f >= lo) & (f <= hi)
                v = im[m]
                out[f"wpli__{band}_{CHANNELS[a]}-{CHANNELS[b]}"] = (
                    np.abs(np.sum(np.abs(v) * np.sign(v))) / (np.sum(np.abs(v)) + 1e-12))
    return out


# ---------------------------------------------------------------- nonlinear
def hjorth(x):
    dx, ddx = np.diff(x), np.diff(x, 2)
    mob = np.sqrt(np.var(dx) / (np.var(x) + 1e-12))
    com = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (mob + 1e-12)
    return mob, com


def perm_entropy(x, order=4, delay=1):
    n = len(x) - (order - 1) * delay
    pat = np.stack([x[i * delay:i * delay + n] for i in range(order)], 1)
    ranks = np.argsort(pat, 1)
    codes = (ranks * (order ** np.arange(order))).sum(1)
    _, cnt = np.unique(codes, return_counts=True)
    pr = cnt / cnt.sum()
    return -(pr * np.log2(pr)).sum() / np.log2(math.factorial(order))


def higuchi_fd(x, kmax=10):
    N = len(x)
    Lk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm.append(np.abs(np.diff(x[idx])).sum() * (N - 1) / ((len(idx) - 1) * k * k))
        Lk.append(np.mean(Lm))
    ks = np.arange(1, kmax + 1)
    return np.polyfit(np.log(1.0 / ks), np.log(np.array(Lk) + 1e-12), 1)[0]


def sample_entropy(x, m=2, r=0.2, max_n=2000):
    x = np.asarray(x[:max_n], float)
    R = r * np.std(x)

    def count(mm):
        t = np.lib.stride_tricks.sliding_window_view(x, mm)[: len(x) - m]
        c = 0
        for i in range(0, len(t), 500):
            d = np.max(np.abs(t[i:i + 500, None, :] - t[None, :, :]), axis=2)
            c += (d < R).sum()
        return c - len(t)  # remove self-matches

    B, A = count(m), count(m + 1)
    return -np.log((A + 1e-12) / (B + 1e-12))


def lzc(x, fs, band=(30, 45)):
    hi = min(band[1], fs / 2 - 1)
    sos = butter(4, (band[0], hi), btype="band", fs=fs, output="sos")
    xs = sosfiltfilt(sos, x - x.mean())
    s = (xs > np.median(xs)).tolist()  # python list: numpy scalar indexing is ~10x slower
    n = len(s)
    # LZ76, Kaspar & Schuster (1987) algorithm
    c, l, i, k, k_max = 1, 1, 0, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c / (n / np.log2(n))


def dfa(x, n_scales=10):
    y = np.cumsum(x - x.mean())
    # fixed scales (16..600 samples) so the exponent doesn't depend on length
    scales = np.unique(np.geomspace(16, 600, n_scales).astype(int))
    N = len(y)
    F = []
    for s in scales:
        ns = N // s
        seg = y[: ns * s].reshape(ns, s)
        t = np.arange(s) - (s - 1) / 2.0
        slope = (seg * t).sum(1) / (t ** 2).sum()
        trend = seg.mean(1, keepdims=True) + slope[:, None] * t
        F.append(np.sqrt(np.mean((seg - trend) ** 2)))
    return np.polyfit(np.log(scales), np.log(np.array(F) + 1e-12), 1)[0]


def nonlinear_feats(x, fs):
    out = {}
    for c in CHANNELS:
        s = x[CI[c]]
        mob, com = hjorth(s)
        out[f"hjorth__mob_{c}"] = mob
        out[f"hjorth__com_{c}"] = com
        out[f"nonlin__permen_{c}"] = perm_entropy(s)
        out[f"nonlin__higuchi_{c}"] = higuchi_fd(s)
        out[f"nonlin__sampen_{c}"] = sample_entropy(s)
        out[f"nonlin__lzc_{c}"] = lzc(s, fs)
        out[f"nonlin__dfa_{c}"] = dfa(s)
    return out


# ---------------------------------------------------------------- covariance
def band_covs(x, fs):
    covs = []
    for lo, hi in COV_BANDS.values():
        sos = butter(4, (lo, hi), btype="band", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        C = np.cov(xf)
        C = C / np.trace(C)  # scale-normalize: removes global gain/impedance
        covs.append(C + 1e-6 * np.eye(6))
    return np.stack(covs)


def recording_features(x, fs):
    d = {}
    d.update(spectral_feats(x, fs))
    d.update(wpli_feats(x, fs))
    d.update(nonlinear_feats(x, fs))
    return d, band_covs(x, fs)


def load(path, win_s):
    """First win_s seconds of the recording, or None if it is shorter."""
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    fs = raw.info["sfreq"]
    n = int(round(win_s * fs))
    x = raw.get_data(picks=CHANNELS) * 1e6
    if x.shape[1] < n:
        return None, fs
    return x[:, :n], fs


def logeuclid_mean(covs):
    logs = []
    for C in covs:
        w, V = np.linalg.eigh(C)
        logs.append(V @ np.diag(np.log(w)) @ V.T)
    L = np.mean(logs, 0)
    w, V = np.linalg.eigh(L)
    return V @ np.diag(np.exp(w)) @ V.T


def process_person(sdir):
    group = os.path.basename(os.path.dirname(sdir))
    person = os.path.basename(sdir)
    res = {"group": group, "person": person}
    rest_p = os.path.join(sdir, REST_FILE)
    if os.path.exists(rest_p):
        x, fs = load(rest_p, REST_WIN_S)
        res["rest"] = recording_features(x, fs)
        bounds = np.linspace(0, x.shape[1], N_SEGMENTS + 1).astype(int)
        res["rest_seg"] = [recording_features(x[:, bounds[i]:bounds[i + 1]], fs)
                           for i in range(N_SEGMENTS)]
    task = []
    for t in range(1, N_TASKS + 1):
        p = os.path.join(sdir, f"T-{t}_raw.fif")
        if not os.path.exists(p):
            continue
        x, fs = load(p, TASK_WIN_S)
        if x is not None:
            task.append(recording_features(x, fs))
    res["task"] = task
    return res


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dirs = sorted(d for g in GROUPS for d in glob.glob(os.path.join(DATA_ROOT, g, "*")) if os.path.isdir(d))
    print(f"{len(dirs)} subjects, {n_jobs} workers")
    with ProcessPoolExecutor(n_jobs) as ex:
        results = []
        for i, r in enumerate(ex.map(process_person, dirs)):
            results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(dirs)}", flush=True)

    names = next(sorted(r["rest"][0]) for r in results if "rest" in r)
    nf, nb = len(names), len(COV_BANDS)
    n = len(results)

    def vec(d):
        return np.array([d[k] for k in names], dtype=np.float64)

    rest_f = np.full((n, nf), np.nan)
    task_f = np.full((n, nf), np.nan)
    seg_f = np.full((n, N_SEGMENTS, nf), np.nan)
    rest_c = np.full((n, nb, 6, 6), np.nan)
    task_c = np.full((n, nb, 6, 6), np.nan)
    seg_c = np.full((n, N_SEGMENTS, nb, 6, 6), np.nan)
    for i, r in enumerate(results):
        if "rest" in r:
            rest_f[i], rest_c[i] = vec(r["rest"][0]), r["rest"][1]
            for s, (fd, cv) in enumerate(r["rest_seg"]):
                seg_f[i, s], seg_c[i, s] = vec(fd), cv
        if r["task"]:
            task_f[i] = np.nanmean([vec(fd) for fd, _ in r["task"]], 0)
            task_c[i] = np.stack([logeuclid_mean([cv[b] for _, cv in r["task"]]) for b in range(nb)])

    np.savez(OUT_PATH,
             persons=np.array([r["person"] for r in results]),
             groups=np.array([r["group"] for r in results]),
             feat_names=np.array(names), cov_bands=np.array(list(COV_BANDS)),
             rest_feats=rest_f, task_feats=task_f, rest_seg_feats=seg_f,
             rest_cov=rest_c, task_cov=task_c, rest_seg_cov=seg_c)
    print(f"saved {OUT_PATH}: {n} persons, {nf} scalar features/recording, {nb} cov bands")


if __name__ == "__main__":
    main()
