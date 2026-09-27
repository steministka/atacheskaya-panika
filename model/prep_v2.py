"""
Предобработка сырых ЭЭГ-записей (.edf), в финальном виде из
assets_2-preprocessing.ipynb. Заменяет prep.py: добавлены ресемплинг
до 125 Гц (чтобы sfreq не выступала признаком с утечкой по группам)
и проверка gain-ratio по каналу T3 (асимметрия калибровки на гарнитуре
NeuroPlay-6C), выполняемая до детекции плохих каналов, чтобы завышенный
T3 не искажал z-оценки остальных каналов.
"""

import numpy as np
import mne

CHANNELS = ["Fp1", "Fp2", "T3", "T4", "O1", "O2"]
CI = {ch: i for i, ch in enumerate(CHANNELS)}
RANDOM_STATE = 42

MONTAGE = mne.channels.make_standard_montage("standard_1020")


def set_montage_safe(raw):
    raw = raw.copy()
    raw.set_montage(MONTAGE, on_missing="warn", match_case=False)
    return raw


def calculate_channel_gain_ratios(raw, target_channels=("T3",)):
    """Ratio of each target channel's std to the median std of the rest.
    raw must already be filtered."""
    data = raw.get_data() * 1e6
    std_per_ch = data.std(axis=1)
    channel_names = raw.ch_names
    result = {}
    for target in target_channels:
        if target not in channel_names:
            continue
        target_idx = channel_names.index(target)
        others = [std for ch, std in zip(channel_names, std_per_ch) if ch != target]
        if not others:
            result[target] = np.nan
            continue
        result[target] = std_per_ch[target_idx] / np.median(others)
    return result


def detect_bad_channels(
    raw,
    flat_thresh_uv=0.5,
    noisy_z_thresh=4.0,
    channel_gain_ratios=None,
    gain_ratio_min=0.5,
    gain_ratio_max=2.0,
):
    """Finds flat channels, anomalously noisy channels, and channels with an
    unusual std ratio relative to the others (channel_gain_ratios, e.g.
    {"T3": ratio}; skipped if None)."""
    data = raw.get_data() * 1e6
    std_per_ch = data.std(axis=1)
    channel_names = raw.ch_names

    flat = std_per_ch < flat_thresh_uv

    log_std = np.log10(std_per_ch + 1e-9)
    z = (log_std - log_std.mean()) / (log_std.std() + 1e-9)
    noisy = np.abs(z) > noisy_z_thresh

    gain_outlier = np.zeros(len(channel_names), dtype=bool)
    gain_ratio_log = {}
    if channel_gain_ratios is not None:
        for ch, ratio in channel_gain_ratios.items():
            if ch not in channel_names:
                continue
            gain_ratio_log[ch] = ratio
            is_outlier = ratio < gain_ratio_min or ratio > gain_ratio_max
            idx = channel_names.index(ch)
            gain_outlier[idx] = is_outlier

    bad_mask = flat | noisy | gain_outlier
    bad_names = [ch for ch, is_bad in zip(channel_names, bad_mask) if is_bad]

    details = {
        "std_uv": dict(zip(channel_names, std_per_ch)),
        "z_score": dict(zip(channel_names, z)),
        "gain_ratio": gain_ratio_log,
    }
    return bad_names, details


def zscore_normalize_raw(raw, target_std_uv=15.0):
    """Per-channel, per-recording amplitude normalization: subtract each
    channel's own mean and divide by its own std (using only that
    recording's samples -- no cross-subject or cross-split statistics, so
    this cannot leak train/valid or group information), then rescale to a
    fixed target std (in uV) so the signal stays in a physiologically
    plausible range for downstream Welch/REVE processing instead of unit
    variance. Removes a pure per-recording amplifier-gain/scale difference;
    does not touch relative spectral shape.
    """
    raw = raw.copy()
    target_std_v = target_std_uv * 1e-6

    def _zscore(x):
        mean = x.mean()
        std = x.std()
        if std < 1e-12:
            return x
        return (x - mean) / std * target_std_v

    raw.apply_function(_zscore, channel_wise=True, verbose=False)
    return raw


def preprocess_raw(
    raw,
    l_freq=1.0,
    h_freq=40.0,
    notch_freq=50.0,
    n_ica_components=5,
    eog_ch=("Fp1", "Fp2"),
    eog_corr_thresh=0.4,
    max_bad_channels=2,
    min_ica_duration_s=15.0,
    t3_ratio_min=0.5,
    t3_ratio_max=2.0,
    mark_t3_bad=False,
    zscore_normalize=False,
    zscore_target_std_uv=15.0,
    verbose=False,
):
    """Full preprocessing of a single Raw:

    1. Set montage.
    2. Preliminary band-pass.
    3. Check T3 relative amplitude.
    4. Detect bad channels.
    5. Interpolate bad channels.
    6. Notch filter.
    7. Average reference / CAR.
    8. ICA.
    9. (optional) per-channel, per-recording z-score amplitude normalization.

    mark_t3_bad=False: only logs a suspicious T3 gain ratio.
    mark_t3_bad=True: also adds T3 to bad channels for interpolation.
    zscore_normalize=True: rescales each channel of the final cleaned signal
    to a fixed std (zscore_target_std_uv), to remove per-recording
    amplifier-gain/scale differences (e.g. the PTSD-vs-control acquisition
    confound) -- see zscore_normalize_raw().
    """
    log = {
        "bad_channels": [],
        "n_bad_channels": 0,
        "t3_gain_ratio": np.nan,
        "t3_gain_suspicious": False,
        "t3_marked_bad": False,
        "ica_fit_ok": False,
        "ica_applied": False,
        "ica_skipped": False,
        "ica_error": False,
        "ica_skip_reason": None,
        "ica_n_components": 0,
        "ica_n_eog_candidates": 0,
        "n_ica_excluded": 0,
        "ica_excluded_idx": [],
        "referenced": False,
        "duration_s": None,
        "n_channels": None,
        "notes": [],
    }

    raw = raw.copy()
    raw.resample(125.0)  # avoids sfreq acting as a group-leakage feature

    duration_s = raw.n_times / raw.info["sfreq"]
    log["duration_s"] = duration_s
    log["n_channels"] = len(raw.ch_names)

    # 1. Montage
    try:
        raw = set_montage_safe(raw)
    except Exception as e:
        log["notes"].append(f"montage error: {e}")

    # 2. Preliminary band-pass (T3 is checked after this, to avoid slow drift)
    try:
        raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose=verbose)
    except Exception as e:
        log["notes"].append(f"band-pass filter error: {e}")

    # 3. T3 relative amplitude check
    try:
        t3_ratios = calculate_channel_gain_ratios(raw, target_channels=("T3",))
        t3_ratio = t3_ratios.get("T3", np.nan)
        log["t3_gain_ratio"] = t3_ratio
        if np.isfinite(t3_ratio):
            if t3_ratio < t3_ratio_min or t3_ratio > t3_ratio_max:
                log["t3_gain_suspicious"] = True
                log["notes"].append(f"T3 suspicious gain ratio: {t3_ratio:.3f}")
                if mark_t3_bad:
                    raw.info["bads"] = ["T3"]
                    log["t3_marked_bad"] = True
    except Exception as e:
        log["notes"].append(f"T3 gain-ratio error: {e}")

    # 4. Bad channel detection + interpolation
    try:
        bad_ch, _ = detect_bad_channels(raw)
        if mark_t3_bad and log["t3_marked_bad"] and "T3" not in bad_ch:
            bad_ch.append("T3")
        bad_ch = sorted(set(bad_ch))
        log["bad_channels"] = bad_ch
        log["n_bad_channels"] = len(bad_ch)

        if len(bad_ch) > max_bad_channels:
            log["notes"].append(f"too many bad channels ({len(bad_ch)}), skipping interpolation")
        elif bad_ch:
            raw.info["bads"] = bad_ch
            raw.interpolate_bads(reset_bads=True, verbose=verbose)
    except Exception as e:
        log["notes"].append(f"bad-channel processing error: {e}")

    # 5. Notch filter
    if duration_s > 2.0:
        try:
            raw.notch_filter(freqs=notch_freq, method="fir", verbose=verbose)
        except Exception as e:
            log["notes"].append(f"notch filter error: {e}")
    else:
        log["notes"].append(f"notch skipped: duration={duration_s:.2f}s <= 2s")

    # 6. Average reference / CAR
    try:
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose=verbose)
        log["referenced"] = True
    except Exception as e:
        log["notes"].append(f"re-reference error: {e}")

    # 7. ICA
    n_comp = min(n_ica_components, len(raw.ch_names) - 1)
    log["ica_n_components"] = n_comp

    def _finish(raw):
        if zscore_normalize:
            try:
                raw = zscore_normalize_raw(raw, target_std_uv=zscore_target_std_uv)
                log["zscore_normalized"] = True
            except Exception as e:
                log["zscore_normalized"] = False
                log["notes"].append(f"zscore normalization error: {e}")
        else:
            log["zscore_normalized"] = False
        return raw, log

    if n_comp < 2:
        log["ica_skipped"] = True
        log["ica_skip_reason"] = f"too few channels for ICA: {len(raw.ch_names)} channels"
        log["notes"].append(log["ica_skip_reason"])
        return _finish(raw)

    if duration_s < min_ica_duration_s:
        log["ica_skipped"] = True
        log["ica_skip_reason"] = f"recording too short: {duration_s:.2f}s < {min_ica_duration_s:.2f}s"
        log["notes"].append(log["ica_skip_reason"])
        return _finish(raw)

    try:
        ica = mne.preprocessing.ICA(
            n_components=n_comp, random_state=RANDOM_STATE, max_iter="auto", verbose=verbose
        )
        ica.fit(raw, verbose=verbose)
        log["ica_fit_ok"] = True

        eog_idx = []
        available_eog_ch = [ch for ch in eog_ch if ch in raw.ch_names and ch not in raw.info["bads"]]
        if not available_eog_ch:
            log["notes"].append("ICA fitted, but no valid EOG channels available")

        for ch in available_eog_ch:
            idx, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=eog_corr_thresh, verbose=verbose)
            eog_idx.extend(idx)
        eog_idx = sorted(set(eog_idx))
        log["ica_n_eog_candidates"] = len(eog_idx)

        if len(eog_idx) > n_comp // 2:
            log["notes"].append("too many suspicious ICA components, ICA was fitted but not applied")
        else:
            ica.exclude = eog_idx
            ica.apply(raw, verbose=verbose)
            log["ica_applied"] = True
            log["n_ica_excluded"] = len(eog_idx)
            log["ica_excluded_idx"] = eog_idx
    except Exception as e:
        log["ica_error"] = True
        log["notes"].append(f"ICA error: {e}")

    # 9. Per-recording amplitude normalization (optional)
    return _finish(raw)
