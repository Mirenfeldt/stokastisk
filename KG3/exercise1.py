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
from pathlib import Path

# --- Load ECG from Excel as a normal DataFrame ---
script_dir = Path(__file__).parent
ecg_file = script_dir / "ecg.xlsx"
df = pd.read_excel(ecg_file)  # Assumes first row is header
ecg = df["ECG"].to_numpy() / 1000  # Scale from µV to mV

# --- Define template ---
template = ecg[10:80]  # Python uses 0-based indexing (11:80 in MATLAB -> 10:80)

# --- Sampling info ---
Fs = 100  # Hz
N = len(ecg)
t = np.arange(N) / Fs  # time vector

# --- Plot ECG and template ---
fig, axs = plt.subplots(2, 2, figsize=(12, 6))
fig.patch.set_facecolor("white")

# Wide plot for ECG
axs[0, 0].plot(t, ecg)
axs[0, 0].set_title("ECG signal")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Amplitude [mV]")
axs[0, 0].boxplot = False  # replicate 'box off'

# Plot template
axs[1, 0].plot(t[:70], template)
axs[1, 0].set_title("Template")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Amplitude [mV]")

# Add instructions (similar to MATLAB text)
axs[1, 0].text(
    0.85 * t[69],
    0.75 * max(template),
    "Use xcorr to make the cross-correlation\n"
    "between the ECG and the defined template.\n"
    "See numpy.correlate or scipy.signal.correlate\n"
    "documentation for further details.\n"
    "\nWhich point on the template does the\n"
    "maxima of the cross-correlation correspond to?\n"
    "Also!, just remove this textbox when done.",
)

plt.tight_layout()
plt.show()
