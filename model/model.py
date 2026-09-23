# model.py — должен содержать:
# объект использованной модели;
# загрузку кэшированных (сохранённых) весов в эту модель.

"""
Автономный модуль инференса для ПТСР-диагностики по ЭЭГ.
Не зависит от ноутбука: повторяет весь пайплайн предобработки и признаков
от сырого .edf до вероятности. Требует model.joblib (сохранён рядом).
 
Использование:
    from ptsd_model import load_model, predict, run
    model = load_model("model.joblib")
    proba = predict(model, "path/to/subject_dir")
    run(model, "path/to/input_dir", "predictions.csv")
"""
import re
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy import signal
import mne
import joblib
 
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
 
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
ASYM_PAIRS = {"Fp": ("Fp1", "Fp2"), "T": ("T3", "T4"), "O": ("O1", "O2")}
CONN_PAIRS = [("Fp1", "Fp2"), ("T3", "T4"), ("O1", "O2"),
              ("Fp1", "T3"), ("T3", "O1"), ("Fp2", "T4"), ("T4", "O2")]
CONN_BANDS = ("theta", "alpha", "beta")
MONTAGE = mne.channels.make_standard_montage("standard_1020")
 
 
def classify_file(stem):
    if re.search(r"(?:^|[_\\\\-\\\\s])(\\u043f|p|rest|\\u043f\\u043e\\u043a\\u043e\\u0439)$", stem, re.I):
        return "rest"
    m = re.search(r"(?:^|[_\\\\-\\\\s])(\\\\d+)$", stem)
    return f"task_{m.group(1)}" if m else "unknown"
 
 
def set_montage_safe(raw):
    raw = raw.copy()
    raw.set_montage(MONTAGE, on_missing="warn", match_case=False)
    return raw
 
 
def detect_bad_channels(raw, flat_thresh_uv=0.5, noisy_z_thresh=4.0):
    data = raw.get_data() * 1e6
    std_per_ch = data.std(axis=1)
    flat = std_per_ch < flat_thresh_uv
    log_std = np.log10(std_per_ch + 1e-9)
    z = (log_std - log_std.mean()) / (log_std.std() + 1e-9)
    noisy = np.abs(z) > noisy_z_thresh
    bad_mask = flat | noisy
    return [ch for ch, b in zip(raw.ch_names, bad_mask) if b], dict(zip(raw.ch_names, std_per_ch))
 
 
def preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freq=50.0,
                    n_ica_components=5, eog_ch=("Fp1", "Fp2"),
                    eog_corr_thresh=0.4, max_bad_channels=2, random_state=42):
    log = {"bad_channels": [], "n_bad_channels": 0, "ica_fit_ok": False,
           "n_ica_excluded": 0, "notes": []}
    raw = raw.copy()
    duration_s = raw.n_times / raw.info["sfreq"]
 
    raw = set_montage_safe(raw)
 
    bad_ch, _ = detect_bad_channels(raw)
    log["bad_channels"] = bad_ch
    log["n_bad_channels"] = len(bad_ch)
    if len(bad_ch) > max_bad_channels:
        log["notes"].append("too many bad channels, skipping interpolation")
    elif bad_ch:
        raw.info["bads"] = bad_ch
        raw.interpolate_bads(reset_bads=True, verbose=False)
 
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False)
    if duration_s > 2.0:
        raw.notch_filter(freqs=notch_freq, method="fir", verbose=False)
 
    # re-reference на средний — обязательно, референс аппаратуры на левом ухе
    try:
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    except Exception as e:
        log["notes"].append(f"re-reference error: {e}")
 
    n_comp = min(n_ica_components, len(raw.ch_names) - 1)
    if duration_s < 15 or n_comp < 2:
        log["notes"].append("ICA skipped: recording too short")
        return raw, log
 
    try:
        ica = mne.preprocessing.ICA(n_components=n_comp, random_state=random_state,
                                     max_iter="auto", verbose=False)
        ica.fit(raw, verbose=False)
        eog_idx = []
        for ch in eog_ch:
            if ch in raw.ch_names and ch not in bad_ch:
                idx, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=eog_corr_thresh, verbose=False)
                eog_idx.extend(idx)
        eog_idx = sorted(set(eog_idx))
        if len(eog_idx) > n_comp // 2:
            log["notes"].append("too many suspicious ICA components")
        else:
            ica.exclude = eog_idx
            ica.apply(raw, verbose=False)
            log["ica_fit_ok"] = True
            log["n_ica_excluded"] = len(eog_idx)
    except Exception as e:
        log["notes"].append(f"ICA error: {e}")
 
    return raw, log
 
 
def clean_epochs(data, fs, win_s=2.0, z_thresh=4.0):
    L = int(win_s * fs)
    starts = np.arange(0, data.shape[1] - L + 1, L // 2)
    if len(starts) < 3:
        return None
    epochs = np.stack([data[:, s:s + L] for s in starts])
    rms = epochs.std(axis=2)
    med = np.median(rms, axis=0)
    mad = np.median(np.abs(rms - med), axis=0) * 1.4826 + 1e-12
    z = np.abs(rms - med) / mad
    keep = (z < z_thresh).all(axis=1)
    if keep.sum() < 3:
        keep = np.ones(len(starts), dtype=bool)
    return epochs[keep]
 
 
def relative_band_power(epochs, fs, channels, bands=BANDS):
    f, p = signal.welch(epochs, fs=fs, nperseg=epochs.shape[-1], axis=-1)
    p = np.median(p, axis=0)
    total = p[:, (f >= 1) & (f <= 40)].sum(axis=1, keepdims=True)
    return {name: (p[:, (f >= lo) & (f < hi)].sum(axis=1) / total[:, 0])
            for name, (lo, hi) in bands.items()}
 
 
def asymmetry_features(rp, ci, bands=("theta", "alpha", "beta")):
    out = {}
    for band in bands:
        logp = np.log10(rp[band] + 1e-12)
        for name, (left, right) in ASYM_PAIRS.items():
            out[f"asym_{band}_{name}"] = float(logp[ci[right]] - logp[ci[left]])
    return out
 
 
def aperiodic_and_peak(epochs, fs, channels, fit_range=(2, 30), alpha_range=(7, 13.5), exclude_alpha=(7, 14)):
    f, p = signal.welch(epochs, fs=fs, nperseg=epochs.shape[-1], axis=-1)
    p = np.median(p, axis=0)
    fit_mask = (f >= fit_range[0]) & (f <= fit_range[1]) & \\\\
               ~((f >= exclude_alpha[0]) & (f <= exclude_alpha[1]))
    logf_fit = np.log10(f[fit_mask] + 1e-12)
    out = {}
    for i, ch in enumerate(channels):
        logp_fit = np.log10(p[i, fit_mask] + 1e-12)
        slope, intercept = np.polyfit(logf_fit, logp_fit, 1)
        fit_line = intercept + slope * np.log10(f + 1e-12)
        resid = np.log10(p[i] + 1e-12) - fit_line
        alpha_mask = (f >= alpha_range[0]) & (f <= alpha_range[1])
        if alpha_mask.sum() > 0:
            local_idx = np.argmax(resid[alpha_mask])
            peak_freq = float(f[alpha_mask][local_idx])
            peak_height = float(resid[alpha_mask][local_idx])
        else:
            peak_freq, peak_height = np.nan, np.nan
        out[f"{ch}__exponent"] = float(-slope)
        out[f"{ch}__alpha_peak_freq"] = peak_freq
        out[f"{ch}__alpha_peak_height"] = peak_height
        out[f"{ch}__alpha_prominent"] = bool(peak_height > 0.1) if not np.isnan(peak_height) else False
    return out
 
 
def imaginary_coherence(epochs, fs, ci, pairs=CONN_PAIRS, bands=CONN_BANDS):
    n_win, n_ch, L = epochs.shape
    window = np.hanning(L)
    X = np.fft.rfft(epochs * window, axis=-1)
    freqs = np.fft.rfftfreq(L, 1 / fs)
    out = {}
    for band in bands:
        lo, hi = BANDS[band]
        mask = (freqs >= lo) & (freqs < hi)
        for ch_a, ch_b in pairs:
            ia, ib = ci[ch_a], ci[ch_b]
            Xa, Xb = X[:, ia, :][:, mask], X[:, ib, :][:, mask]
            Sab = (Xa * np.conj(Xb)).mean(axis=0)
            Saa = (Xa * np.conj(Xa)).mean(axis=0).real
            Sbb = (Xb * np.conj(Xb)).mean(axis=0).real
            icoh = np.abs(Sab.imag) / np.sqrt(Saa * Sbb + 1e-12)
            out[f"conn_{band}_{ch_a}-{ch_b}"] = float(np.mean(icoh))
    return out
 
 
def backfit_microstates(data, templates):
    norm_data = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-12)
    corr = templates @ norm_data
    return np.argmax(np.abs(corr), axis=0)
 
 
def microstate_features(labels, fs, n_states):
    feats = {}
    total_time = len(labels) / fs
    if total_time <= 0:
        return {f"ms{k}_{p}": np.nan for k in range(n_states)
                for p in ("occurrence_per_s", "coverage", "mean_dur_s")}
    change_idx = np.where(np.diff(labels) != 0)[0] + 1
    segments = np.split(labels, change_idx)
    for k in range(n_states):
        durations = [len(seg) / fs for seg in segments if len(seg) and seg[0] == k]
        feats[f"ms{k}_occurrence_per_s"] = len(durations) / total_time
        feats[f"ms{k}_coverage"] = float(sum(durations) / total_time) if durations else 0.0
        feats[f"ms{k}_mean_dur_s"] = float(np.mean(durations)) if durations else 0.0
    return feats
 
 
def compute_file_features(path, channels, ci, ms_templates=None):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick(channels)
    raw, prep_log = preprocess_raw(raw)
    data = raw.get_data() * 1e6
    fs = raw.info["sfreq"]
 
    epochs = clean_epochs(data, fs)
    if epochs is None:
        return None
 
    rp = relative_band_power(epochs, fs, channels)
    feats = {}
    for band, vals in rp.items():
        for i, ch in enumerate(channels):
            feats[f"{ch}__rel_{band}"] = float(vals[i])
    feats.update(asymmetry_features(rp, ci))
    feats.update(aperiodic_and_peak(epochs, fs, channels))
    feats.update(imaginary_coherence(epochs, fs, ci))
    if ms_templates is not None:
        labels = backfit_microstates(data, ms_templates)
        feats.update(microstate_features(labels, fs, ms_templates.shape[0]))
    feats["n_clean_epochs"] = int(epochs.shape[0])
    return feats
 
 
def load_model(path="model.joblib"):
    return joblib.load(path)
 
 
def predict(model, subject_dir) -> float:
    """Папка ОДНОГО испытуемого -> вероятность ПТСР в [0, 1]."""
    subject_dir = Path(subject_dir)
    channels = model["channels"]
    ci = {ch: i for i, ch in enumerate(channels)}
 
    files = list(subject_dir.glob("*.edf"))
    rest_f = next((f for f in files if classify_file(f.stem) == "rest"), None)
    if rest_f is None:
        return 0.5
 
    try:
        feats = compute_file_features(rest_f, channels, ci, ms_templates=model.get("ms_templates"))
    except Exception:
        return 0.5
    if not feats:
        return 0.5
 
    feature_names = model["feature_names"]
    x_vals = [feats.get(name.replace("rest__", "", 1), np.nan) for name in feature_names]
    x = np.array([x_vals])
 
    proba = model["pipe"].predict_proba(x)[:, 1]
    return float(np.clip(proba[0], 0.0, 1.0))
 
 
def run(model, input_dir, output_csv) -> None:
    input_dir = Path(input_dir)
    subject_dirs = sorted({p for p in input_dir.rglob("*") if p.is_dir() and list(p.glob("*.edf"))})
    rows = []
    for subj_dir in subject_dirs:
        proba = predict(model, subj_dir)
        rows.append({"subject_id": subj_dir.name, "ptsd_probability": proba})
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Сохранено {len(rows)} предсказаний -> {output_csv}")
