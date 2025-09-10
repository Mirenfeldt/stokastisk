# --- Python cross-correlation hint ---
# To compute cross-correlation in Python:
#   from scipy.signal import correlate
#   correlate(signal1, signal2, mode='full')
# Or using numpy:
#   np.correlate(signal1, signal2, mode='full')
# See documentation:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
# https://numpy.org/doc/stable/reference/generated/numpy.correlate.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks, correlation_lags
from pathlib import Path

# --- Load ECG from Excel ---
script_dir = Path(__file__).parent
ecg_file = script_dir / "ecg.xlsx"
df = pd.read_excel(ecg_file)
ecg = df["ECG"].to_numpy() / 1000  # Scale from µV to mV

# --- Define template ---
template = ecg[10:80]  # MATLAB 11:80 -> Python 10:80

# --- Sampling info ---
Fs = 100  # Hz
N = len(ecg)

# --- Compute cross-correlation ---
r = correlate(ecg, template, mode="full")
lags = correlation_lags(len(ecg), len(template), mode="full")  # Corresponding lags

# --- Plot ECG and cross-correlation ---
plt.figure(figsize=(10, 5))
plt.plot(ecg, label="ECG")
plt.plot(lags, r, label="Cross-correlation")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.box(False)

# --- Find peaks in the cross-correlation ---
locs, peaks = find_peaks(r, distance=70, height=2)  # distance ~ MinPeakDistance
plt.plot(lags[locs], peaks["peak_heights"], "rx", label="Detected peaks")
plt.legend(loc="upper left")
plt.title("ECG and Cross-Correlation")


# --- Align template to ECG beats ---
plt.figure(figsize=(10, 5))
plt.plot(ecg, linewidth=2, label="ECG")
plt.box(False)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Template alignment on detected peaks")
plt.hold = True  # Python plotting will automatically overlay

median_matrix = np.zeros((len(locs), 70))
n_beats = 0

for i, loc in enumerate(locs):
    tnn = lags[loc]  # Time of "hit"
    if 0 <= tnn <= N - 70:  # Ensure we do not exceed signal length
        plt.plot(np.arange(tnn, tnn + 70), template, "r", linewidth=0.5)
        median_matrix[n_beats, :] = ecg[tnn : tnn + 70]
        n_beats += 1

# Keep only used slots
median_matrix = median_matrix[:n_beats, :]

# --- Plot aligned beats ---
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(median_matrix.T)
plt.box(False)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Aligned ECG beats")

plt.subplot(2, 1, 2)
plt.plot(np.median(median_matrix, axis=0), label="Median beat")
plt.plot(np.mean(median_matrix, axis=0), label="Mean beat")
plt.box(False)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Median and Mean beat")
plt.legend()
plt.tight_layout()
plt.show()
