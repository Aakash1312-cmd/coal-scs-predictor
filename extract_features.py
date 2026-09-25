"""
Feature Extraction Pipeline for Coal Ultrasonic Signals
======================================================
Predicts Ash Content, Fixed Carbon Content, and Ignition Temperature.
Extracts 20 Ash features, 18 Carbon features, and 20 Thermal/Ignition features
from raw time-domain signals and 50 MHz FFT frequency-domain signals.
Includes:
  - Baseline drift removal (polynomial fitting)
  - Savitzky-Golay smoothing
  - Wavelet denoising (PyWavelets db4)
  - SEC onset picker (Bühling & Maack, JASA 2024)
  - Acoustic velocity from 6 mm thickness and Time-of-Flight
  - Frequency windowing in bands: 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.3 MHz
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, hilbert, find_peaks
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis
# pyrefly: ignore [missing-import]
import pywt


FS = 50e6  # 50 MHz sampling frequency
DT = 1.0 / FS  # 20 ns
PELLET_THICKNESS_M = 0.006  # 6 mm thickness in meters
EPS = 1e-15

# Frequency bands in MHz
BANDS = {
    1: (0.2, 0.4),
    2: (0.4, 0.6),
    3: (0.6, 0.8),
    4: (0.8, 1.0),
    5: (1.0, 1.3),
}

# 12 Target Coal Samples
TARGET_SAMPLES = [
    "C5", "C9", "C10", "C12",
    "C3", "C4", "C13", "C14",
    "C6", "C7", "C15", "C16"
]

# Ground Truth Table (from Proximate Analysis)
GROUND_TRUTH = {
    "C5":  {"ash": 26.48, "carbon": 28.19, "ignition": 317.617, "class": "High"},
    "C9":  {"ash": 39.72, "carbon": 27.89, "ignition": 319.186, "class": "High"},
    "C10": {"ash": 20.10, "carbon": 38.98, "ignition": 334.592, "class": "High"},
    "C12": {"ash": 23.29, "carbon": 34.54, "ignition": 332.468, "class": "High"},
    "C3":  {"ash": np.nan, "carbon": np.nan, "ignition": 412.297, "class": "Low"},  # C3_1 in sheet
    "C4":  {"ash": 27.94, "carbon": 43.78, "ignition": 439.720, "class": "Low"},
    "C13": {"ash": 15.59, "carbon": 40.61, "ignition": 334.714, "class": "Low"},
    "C14": {"ash": 13.09, "carbon": 46.52, "ignition": 339.829, "class": "Low"},
    "C6":  {"ash": 23.81, "carbon": 42.80, "ignition": 346.884, "class": "Medium"},
    "C7":  {"ash": 64.00, "carbon": 8.84,  "ignition": 355.144, "class": "Medium"},
    "C15": {"ash": 10.14, "carbon": 67.43, "ignition": 417.843, "class": "Medium"},
    "C16": {"ash": 12.90, "carbon": 62.09, "ignition": 440.387, "class": "Medium"},
}


def remove_baseline_drift(signal, poly_order=2):
    """
    Remove baseline drifting via polynomial fitting.
    """
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, poly_order)
    baseline = np.polyval(coeffs, x)
    return signal - baseline


def savgol_smoothing(signal, window_length=15, poly_order=2):
    """
    Smooth signal using Savitzky-Golay filter.
    """
    wl = min(window_length, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if wl < 5:
        return signal.copy()
    return savgol_filter(signal, window_length=wl, polyorder=poly_order, mode="interp")


def wavelet_denoise(signal, wavelet="db4", level=4):
    """
    Wavelet denoising using discrete wavelet transform and VisuShrink soft thresholding.
    """
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    lvl = min(level, max_level)
    if lvl < 1:
        return signal.copy()

    coeffs = pywt.wavedec(signal, wavelet, level=lvl)
    # Estimate noise from MAD of highest frequency detail coefficients
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs - np.median(detail_coeffs))) / 0.6745
    if sigma <= 0:
        sigma = np.std(detail_coeffs)

    # Universal threshold (VisuShrink)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply soft thresholding to all detail coefficients
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, value=threshold, mode="soft"))

    denoised = pywt.waverec(new_coeffs, wavelet)
    # Ensure length matches original
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    elif len(denoised) < len(signal):
        denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode="edge")

    return denoised


def preprocess_signal(raw_signal):
    """
    Full preprocessing pipeline for '_proc':
    1. Baseline drift removal (Polynomial fitting)
    2. Savitzky-Golay smoothing
    3. Wavelet denoising
    """
    step1 = remove_baseline_drift(raw_signal, poly_order=2)
    step2 = savgol_smoothing(step1, window_length=15, poly_order=2)
    step3 = wavelet_denoise(step2, wavelet="db4", level=4)
    return step3

def sec_onset_picker(signal, window_samples=50):
    """
    Spectral Entropy Criterion (SEC) picker from Bühling & Maack (JASA 2024).
    Calculates sliding normalized spectral entropy:
      SEC(k) = -Hs(k - W .. k) + Hs(k .. k + W)
    Global minimum marks signal onset.
    """
    N = len(signal)
    W = window_samples
    if N <= 2 * W:
        return 0, 0.0

    # Compute Hs for all sliding windows of length W
    Hs = np.zeros(N - W + 1)
    for i in range(len(Hs)):
        seg = signal[i:i + W]
        fft_mag = np.abs(np.fft.rfft(seg))
        power = fft_mag ** 2
        power[0] = 0  # exclude DC
        tot_power = np.sum(power)
        if tot_power > EPS:
            p = power / tot_power
            p = p[p > 0]
            ent = -np.sum(p * np.log2(p))
            Hs[i] = ent / np.log2(len(power))
        else:
            Hs[i] = 1.0

    # Forward + Backward functions
    k_range = np.arange(W, N - W)
    sec_values = -Hs[k_range - W] + Hs[k_range]
    k_min = k_range[np.argmin(sec_values)]
    onset_time = k_min * DT

    return int(k_min), float(onset_time)


def compute_fft(signal, fs=FS):
    """
    Compute single-sided FFT magnitude and power spectrum in MHz.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freq_hz = np.fft.rfftfreq(n, d=1.0 / fs)
    freq_mhz = freq_hz / 1e6
    mag = np.abs(fft_vals)
    power = mag ** 2
    return freq_mhz, mag, power


def get_band_data(freq_mhz, values, low, high):
    mask = (freq_mhz >= low) & (freq_mhz <= high) & np.isfinite(values)
    return freq_mhz[mask], values[mask]


def calc_spectral_area(freq_mhz, mag, low=None, high=None):
    if low is not None and high is not None:
        f, m = get_band_data(freq_mhz, mag, low, high)
    else:
        # Full spectrum excluding DC
        mask = (freq_mhz > 0) & np.isfinite(mag)
        f, m = freq_mhz[mask], mag[mask]

    if len(f) < 2:
        return 0.0
    return float(trapezoid(m, f))


def calc_dominant_frequency(freq_mhz, mag, low, high):
    f, m = get_band_data(freq_mhz, mag, low, high)
    if len(f) == 0:
        return 0.0
    return float(f[np.argmax(m)])


def calc_spectral_centroid(freq_mhz, mag, low=None, high=None):
    if low is not None and high is not None:
        f, m = get_band_data(freq_mhz, mag, low, high)
    else:
        mask = (freq_mhz > 0) & np.isfinite(mag)
        f, m = freq_mhz[mask], mag[mask]

    tot = np.sum(m)
    if tot <= EPS:
        return 0.0
    return float(np.sum(f * m) / tot)


def calc_spectral_bandwidth(freq_mhz, mag, low, high):
    f, m = get_band_data(freq_mhz, mag, low, high)
    tot = np.sum(m)
    if tot <= EPS or len(f) < 2:
        return 0.0
    centroid = np.sum(f * m) / tot
    variance = np.sum(m * (f - centroid) ** 2) / tot
    return float(np.sqrt(max(variance, 0.0)))


def calc_spectral_entropy(freq_mhz, power, low, high):
    f, p = get_band_data(freq_mhz, power, low, high)
    p = np.maximum(p, 0.0)
    tot = np.sum(p)
    if tot <= EPS or len(p) < 2:
        return 0.0
    probs = p / tot
    probs = probs[probs > EPS]
    if len(probs) <= 1:
        return 0.0
    ent = -np.sum(probs * np.log2(probs))
    norm = np.log2(len(p))
    return float(ent / norm) if norm > EPS else 0.0


def calc_wave_energy(freq_mhz, power, low, high):
    f, p = get_band_data(freq_mhz, power, low, high)
    if len(f) < 2:
        return 0.0
    return float(trapezoid(p, f))


def calc_wave_entropy(power):
    p = np.asarray(power, dtype=float)
    p = p[1:]  # exclude DC
    p = np.maximum(p, 0.0)
    tot = np.sum(p)
    if tot <= EPS or len(p) < 2:
        return 0.0
    probs = p / tot
    probs = probs[probs > EPS]
    if len(probs) <= 1:
        return 0.0
    ent = -np.sum(probs * np.log2(probs))
    return float(ent / np.log2(len(p)))

def calc_rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


def calc_p2p(signal):
    return float(np.max(signal) - np.min(signal))


def calc_absorption_proxy(signal):
    """
    Peak amplitude of the signal (not peak-to-peak).
    """
    return float(np.max(np.abs(signal)))


def calc_snr(signal, noise_segment):
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise_segment ** 2)
    if noise_power <= EPS:
        noise_power = EPS
    return float(10.0 * np.log10(max(sig_power, EPS) / noise_power))


def calc_pulse_envelope_features(time, signal_proc):
    """
    Extracts peak time, rise time, fall time, duration (FWHM), and 1st/2nd peak ratio
    from the Hilbert analytic envelope of the processed signal.
    """
    analytic = hilbert(signal_proc)
    envelope = np.abs(analytic)

    # Find dominant peak (excluding edge margins to avoid Hilbert boundary distortion)
    edge = min(25, len(envelope) // 20)
    search_env = envelope.copy()
    search_env[:edge] = -np.inf
    search_env[-edge:] = -np.inf

    peak_idx = int(np.argmax(search_env))
    peak_val = envelope[peak_idx]
    peak_time = float(time[peak_idx])

    # Local baseline
    baseline = np.min(envelope)
    amp = peak_val - baseline
    if amp <= EPS:
        amp = EPS

    # 10% and 90% levels for rise and fall
    level10 = baseline + 0.10 * amp
    level90 = baseline + 0.90 * amp
    level50 = baseline + 0.50 * amp

    # Rise time (10% to 90% before peak)
    t10_rise = time[0]
    t90_rise = time[peak_idx]
    for i in range(peak_idx - 1, -1, -1):
        if envelope[i] <= level90 <= envelope[i + 1]:
            frac = (level90 - envelope[i]) / max(envelope[i + 1] - envelope[i], EPS)
            t90_rise = time[i] + frac * (time[i + 1] - time[i])
            break
    for i in range(peak_idx - 1, -1, -1):
        if envelope[i] <= level10 <= envelope[i + 1]:
            frac = (level10 - envelope[i]) / max(envelope[i + 1] - envelope[i], EPS)
            t10_rise = time[i] + frac * (time[i + 1] - time[i])
            break
    rise_time = float(max(t90_rise - t10_rise, 0.0))

    # Fall time (90% to 10% after peak)
    t90_fall = time[peak_idx]
    t10_fall = time[-1]
    for i in range(peak_idx, len(envelope) - 1):
        if envelope[i] >= level90 >= envelope[i + 1]:
            frac = (envelope[i] - level90) / max(envelope[i] - envelope[i + 1], EPS)
            t90_fall = time[i] + frac * (time[i + 1] - time[i])
            break
    for i in range(peak_idx, len(envelope) - 1):
        if envelope[i] >= level10 >= envelope[i + 1]:
            frac = (envelope[i] - level10) / max(envelope[i] - envelope[i + 1], EPS)
            t10_fall = time[i] + frac * (time[i + 1] - time[i])
            break
    fall_time = float(max(t10_fall - t90_fall, 0.0))

    # Peak duration (FWHM at 50% height)
    t_left = time[0]
    t_right = time[-1]
    for i in range(peak_idx - 1, -1, -1):
        if envelope[i] <= level50 <= envelope[i + 1]:
            frac = (level50 - envelope[i]) / max(envelope[i + 1] - envelope[i], EPS)
            t_left = time[i] + frac * (time[i + 1] - time[i])
            break
    for i in range(peak_idx, len(envelope) - 1):
        if envelope[i] >= level50 >= envelope[i + 1]:
            frac = (envelope[i] - level50) / max(envelope[i] - envelope[i + 1], EPS)
            t_right = time[i] + frac * (time[i + 1] - time[i])
            break
    peak_duration = float(max(t_right - t_left, 0.0))

    # 1st peak magnitude / 2nd peak magnitude
    peaks, _ = find_peaks(envelope, prominence=0.05 * amp, distance=10)
    if len(peaks) >= 2:
        sorted_peaks = np.sort(envelope[peaks])[::-1]
        peak_ratio = float(sorted_peaks[0] / max(sorted_peaks[1], EPS))
    else:
        peak_ratio = 1.0

    return {
        "peak_time": peak_time,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "peak_duration": peak_duration,
        "peak_ratio": peak_ratio,
    }


def calc_p2p_tdb3(signal_proc):
    """
    P2P-TDB3: Peak to Peak value for the processed signal in the third time window.
    Dividing the time domain into 3 equal windows: [0, T/3], [T/3, 2T/3], [2T/3, T].
    TDB3 is the 3rd window [2T/3, T].
    """
    n = len(signal_proc)
    w_start = int(2 * n / 3)
    tdb3_seg = signal_proc[w_start:]
    return float(np.max(tdb3_seg) - np.min(tdb3_seg))


def parse_signal_column(col_name):
    """
    Parses column names like INTENSITY_C5_P1_01 into:
    sample_code ('C5'), pellet ('P1'), subsample ('01')
    """
    m = re.search(r"C(\d+)(?:_(\d+))?_P(\d+)_(\d+)", col_name, re.I)
    if not m:
        return None
    sample_num = m.group(1)
    pellet_num = m.group(3)
    site_num = m.group(4)
    code = f"C{sample_num}"
    return {
        "coal_sample": code,
        "pellet": f"P{pellet_num}",
        "subsample": site_num,
        "instance_id": f"{code}_P{pellet_num}_{site_num}",
    }



def extract_all_features():
    print("=" * 70)
    print("EXTRACTING FEATURES FOR ASH, CARBON & IGNITION TEMPERATURE")
    print("=" * 70)

    raw_files = {
        "High": "data/high_clean(1).csv",
        "Low": "data/low_clean(1).csv",
        "Medium": "data/medium_clean(1).csv",
    }

    ash_records = []
    carbon_records = []
    ignition_records = []

    total_signals_processed = 0

    for scs_class, file_path in raw_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

        print(f"\nProcessing {scs_class} class data from {file_path}...")
        df = pd.read_csv(file_path)

        # Locate time column
        time_col = [c for c in df.columns if "time" in c.lower()][0]
        time_vals = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

        # Signal columns
        for col in df.columns:
            if col == time_col:
                continue

            meta = parse_signal_column(col)
            if not meta:
                continue

            sample_code = meta["coal_sample"]
            if sample_code not in TARGET_SAMPLES:
                continue

            # Signal series
            raw_sig = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(time_vals) & np.isfinite(raw_sig)
            t = time_vals[valid_mask]
            s_raw = raw_sig[valid_mask]

            if len(s_raw) < 50:
                continue

            s_proc = preprocess_signal(s_raw)

            k_onset, t_onset = sec_onset_picker(s_proc, window_samples=50)
            # Pre-onset noise segment (at least 20 samples)
            k_noise_end = max(20, k_onset if k_onset > 20 else 50)
            noise_segment = s_raw[:k_noise_end] - np.mean(s_raw[:k_noise_end])

            freq_raw, mag_raw, pwr_raw = compute_fft(s_raw, fs=FS)
            freq_proc, mag_proc, pwr_proc = compute_fft(s_proc, fs=FS)
            rms_val = calc_rms(s_raw)
            p2p_val = calc_p2p(s_raw)
            p2p_proc = calc_p2p(s_proc)
            abs_proxy_raw = calc_absorption_proxy(s_raw)
            abs_proxy_proc = calc_absorption_proxy(s_proc)
            skew_val = float(skew(s_raw))
            kurt_val = float(kurtosis(s_raw, fisher=False))  # Pearson kurtosis

            snr_raw = calc_snr(s_raw, noise_segment)
            snr_proc = calc_snr(s_proc, noise_segment)

            pulse_feats = calc_pulse_envelope_features(t, s_proc)
            peak_time_proc = pulse_feats["peak_time"]
            peak_time_raw = float(t[np.argmax(np.abs(s_raw))])
            rise_time_proc = pulse_feats["rise_time"]
            fall_time_proc = pulse_feats["fall_time"]
            peak_dur_proc = pulse_feats["peak_duration"]
            peak_ratio = pulse_feats["peak_ratio"]

            sig_energy_proc = float(np.sum(s_proc ** 2))
            auc_proc = float(trapezoid(np.abs(s_proc), t))

            # Acoustic velocity (thickness 6 mm = 0.006 m / ToF)
            # ToF from SEC onset picker (or peak time as fallback if onset < 1e-7)
            tof = t_onset if t_onset > 1e-7 else peak_time_proc
            acoustic_velocity = float(PELLET_THICKNESS_M / max(tof, EPS))

            p2p_tdb3_val = calc_p2p_tdb3(s_proc)

            # Full spectrum raw features
            area_fft_raw = calc_spectral_area(freq_raw, mag_raw)
            # Dominant frequency magnitude raw (excluding DC)
            mask_no_dc = freq_raw > 0
            mag_dom_raw = float(np.max(mag_raw[mask_no_dc])) if np.any(mask_no_dc) else 0.0

            centroid_full = calc_spectral_centroid(freq_raw, mag_raw)
            wave_entropy_full = calc_wave_entropy(pwr_raw)

            # Band-specific features
            spec_bw_raw = {}
            spec_area = {}
            spec_entropy = {}
            dom_freq_peak = {}
            spec_centroid = {}
            wave_energy = {}

            for b in [1, 2, 3, 4, 5]:
                low, high = BANDS[b]
                spec_bw_raw[b] = calc_spectral_bandwidth(freq_raw, mag_raw, low, high)
                spec_area[b] = calc_spectral_area(freq_raw, mag_raw, low, high)
                spec_entropy[b] = calc_spectral_entropy(freq_raw, pwr_raw, low, high)
                dom_freq_peak[b] = calc_dominant_frequency(freq_raw, mag_raw, low, high)
                spec_centroid[b] = calc_spectral_centroid(freq_raw, mag_raw, low, high)
                wave_energy[b] = calc_wave_energy(freq_raw, pwr_raw, low, high)

            # Ground truth targets
            gt = GROUND_TRUTH[sample_code]


            ash_records.append({
                "instance_id": meta["instance_id"],
                "coal_sample": sample_code,
                "pellet": meta["pellet"],
                "subsample": meta["subsample"],
                "scs_class": scs_class,
                "target_value": gt["ash"],
                "Area under FFT curve_raw": area_fft_raw,
                "Magnitude of dominant frequency_raw": mag_dom_raw,
                "Spectral Bandwidth 1_raw": spec_bw_raw[1],
                "Spectral Bandwidth 2_raw": spec_bw_raw[2],
                "Spectral Bandwidth 3_raw": spec_bw_raw[3],
                "Spectral Area 1": spec_area[1],
                "Spectral Area 2": spec_area[2],
                "Spectral Area 3": spec_area[3],
                "Spectral Area 4": spec_area[4],
                "Spectral Entropy 1": spec_entropy[1],
                "Spectral Entropy 2": spec_entropy[2],
                "Spectral Entropy 3": spec_entropy[3],
                "Spectral Entropy 4": spec_entropy[4],
                "Absorption Proxy_proc": abs_proxy_proc,
                "Peak Time_proc": peak_time_proc,
                "Rise time_proc": rise_time_proc,
                "Fall time_proc": fall_time_proc,
                "Signal Energy_proc": sig_energy_proc,
                "Area under the curve_proc": auc_proc,
                "1st peak magnitude/2nd peak magnitude": peak_ratio,
            })

        
            carbon_records.append({
                "instance_id": meta["instance_id"],
                "coal_sample": sample_code,
                "pellet": meta["pellet"],
                "subsample": meta["subsample"],
                "scs_class": scs_class,
                "target_value": gt["carbon"],
                "RMS value": rms_val,
                "P2P": p2p_val,
                "Dominant Frequency 1 peak": dom_freq_peak[1],
                "Spectral Entropy 2": spec_entropy[2],
                "Dominant Frequency 3 peak": dom_freq_peak[3],
                "Absorption Proxy": abs_proxy_raw,
                "Dominant Frequency 2 peak": dom_freq_peak[2],
                "SNR_raw": snr_raw,
                "Spectral Entropy 1": spec_entropy[1],
                "Spectral Entropy 3": spec_entropy[3],
                "Spectral Bandwidth 4": spec_bw_raw[4],
                "Spectral Centroid": centroid_full,
                "Skew Value": skew_val,
                "SNR_Processed": snr_proc,
                "Spectral Entropy 4": spec_entropy[4],
                "Spectral Centroid 1": spec_centroid[1],
                "Spectral Centroid 2": spec_centroid[2],
                "Spectral Bandwidth 1": spec_bw_raw[1],
            })
            ignition_records.append({
                "instance_id": meta["instance_id"],
                "coal_sample": sample_code,
                "pellet": meta["pellet"],
                "subsample": meta["subsample"],
                "scs_class": scs_class,
                "target_value": gt["ignition"],
                "Spectral Centroid": centroid_full,
                "SNR_raw": snr_raw,
                "Spectral Bandwidth 5": spec_bw_raw[5],
                "Spectral Entropy 4": spec_entropy[4],
                "Spectral Centroid 3": spec_centroid[3],
                "Kurtosis": kurt_val,
                "RMS value": rms_val,
                "Absorption Proxy": abs_proxy_raw,
                "P2P": p2p_val,
                "Wave Energy 3": wave_energy[3],
                "Acoustic Velocity": acoustic_velocity,
                "Peak Time": peak_time_raw,
                "Spectral Centroid 2": spec_centroid[2],
                "Dominant Frequency 1 peak": dom_freq_peak[1],
                "Spectral Bandwidth 3": spec_bw_raw[3],
                "Wave Entropy": wave_entropy_full,
                "P2P-TDB3": p2p_tdb3_val,
                "Peak Duration": peak_dur_proc,
                "Spectral Centroid 4": spec_centroid[4],
                "Dominant Frequency 2 peak": dom_freq_peak[2],
            })

            total_signals_processed += 1

    print(f"\nTotal signals processed: {total_signals_processed}")

    # Convert to DataFrames
    df_ash = pd.DataFrame(ash_records)
    df_carbon = pd.DataFrame(carbon_records)
    df_ignition = pd.DataFrame(ignition_records)

    # Save CSVs in root workspace
    df_ash.to_csv("features_ash_content.csv", index=False)
    df_carbon.to_csv("features_carbon_content.csv", index=False)
    df_ignition.to_csv("features_ignition_temp.csv", index=False)

    print("\nSaved files:")
    print("  - features_ash_content.csv     shape:", df_ash.shape)
    print("  - features_carbon_content.csv  shape:", df_carbon.shape)
    print("  - features_ignition_temp.csv   shape:", df_ignition.shape)

    return df_ash, df_carbon, df_ignition


if __name__ == "__main__":
    extract_all_features()