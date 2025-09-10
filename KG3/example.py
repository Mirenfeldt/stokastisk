import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from pathlib import Path

# --- Determine the path to the script and ECG file ---
script_dir = Path(__file__).parent  # folder where this script is located
ecg_file = script_dir / "ecg.xlsx"  # ECG file in the same folder

# --- Load ECG ---
df = pd.read_excel(ecg_file)
ecg: np.ndarray[float] = df["ECG"].to_numpy(dtype=float)
# --- Plot raw ECG ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(ecg, lw=1)
axs[0, 0].set_title("One realization of a stochastic process X(t)")
axs[0, 0].set_xlabel("Samples")
axs[0, 0].set_ylabel("Amplitude")

# --- Autocorrelation of ECG ---
rx = correlate(ecg, ecg, mode="full")
lags = np.arange(-len(ecg) + 1, len(ecg))
axs[1, 0].plot(lags, rx)
axs[1, 0].set_title("Estimate of the autocorrelation function r_X(τ) of X(t)")
axs[1, 0].set_xlabel("τ [samples]")
axs[1, 0].set_ylabel("Arbitrary value")
axs[1, 0].set_xlim([500, 1500])

# --- Peak detection ---
qrs_peaks, _ = find_peaks(ecg, height=800)
t_peaks, _ = find_peaks(ecg, height=100, width=5)

# Plot detected peaks
axs[0, 0].plot(qrs_peaks, ecg[qrs_peaks], "rx", label="QRS peaks")
axs[0, 0].plot(t_peaks, ecg[t_peaks], "gx", label="T peaks")
axs[0, 0].legend()

# --- Compute HR and QRS-to-T delay ---
mean_rr = np.mean(np.diff(qrs_peaks))
mean_qrs_t_delay = np.mean(t_peaks[:10] - qrs_peaks[:10])

print(f"Mean RR interval:         {mean_rr:.1f} [samples]")
print(f"Mean QRS to T-peak delay: {mean_qrs_t_delay:.1f} [samples]")

# --- Autocorrelation of pure and noisy sine waves ---
t = np.arange(0, 2, 0.01)
x = np.sin(2 * np.pi * 3 * t)
xn = np.sin(2 * np.pi * 3 * t + 2 * np.random.rand(len(t)))

axs[0, 1].plot(t, x, label="Pure sine")
axs[0, 1].plot(t, xn, label="Noisy sine")
axs[0, 1].set_title("Pure and noisy sine")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Amplitude")
axs[0, 1].legend()

rsin = correlate(x, x, mode="full")
rsinn = correlate(xn, xn, mode="full")

axs[1, 1].plot(rsin, label="Pure sine autocorr")
axs[1, 1].plot(rsinn, label="Noisy sine autocorr")
axs[1, 1].set_title("Autocorrelation of pure and noisy sine")
axs[1, 1].set_xlabel("Delay (samples)")
axs[1, 1].set_ylabel("Arbitrary value")
axs[1, 1].legend()

plt.tight_layout()
plt.show()
