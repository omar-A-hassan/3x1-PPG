"""
Visualize PPG signal after preprocessing steps 1, 3, 4
(without respiratory bandpass filter step 2)
"""

import numpy as np
import matplotlib.pyplot as plt
from main import PPGPreprocessor
import pandas as pd

# Load your PPG data
data_path = r"C:\Users\nazeh\BioInfo Trials\3x1-PPG\Dev\3x1-PPG\ppg_data\ppg_data_20251119_202843(16).csv"
df = pd.read_csv(data_path)

# Extract IR values (assuming column name is 'IR Value')
raw_signal = df['IR_Value'].values

print(f"Loaded {len(raw_signal)} samples")

# Initialize preprocessor
pre = PPGPreprocessor(sampling_rate=50)

# Step 1: Handle missing values
cleaned = pre.handle_missing_values(raw_signal)
print(f"Step 1 complete: {len(cleaned)} samples after missing value handling")

# Step 1.5: Bandpass filter
bandpassed = pre.bandpass_filter(cleaned,lowcut=0.1, highcut=4.0)
print(f"Step 2.5 complete: Bandpass filter applied")

# Step 2: Peak enhancement
enhanced = pre.peak_enhancement(bandpassed)
print(f"Step 2 complete: Peak enhancement applied")

# Step 3: Hampel filter
denoised = pre.hampel_filter(enhanced)
print(f"Step 3 complete: Hampel filter applied")

#Step 4: Peak detection 
peaks=pre.detect_peaks_dynamic(denoised)
peaks_values=denoised[peaks]
print(f"Step 4 complete: Detected {len(peaks)} peaks")
print(f"Peak values list length: {len(peaks_values)}")

# Create time axis
time = np.arange(len(raw_signal)) / pre.sampling_rate
peaks_times= time[peaks]
print(f"Time axis created with {len(time)} points")
print(f"Total duration: {time[-1]:.2f} seconds")
print(f"List of peak times: {peaks_times}")

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle('PPG Signal Processing Steps', fontsize=16, fontweight='bold')

# Plot 1: Original raw signal
axes[0].plot(time, raw_signal, color='#1f77b4', linewidth=0.8)
axes[0].set_title('Original Raw Signal', fontsize=12, fontweight='bold')
axes[0].set_ylabel('IR Value', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, time[-1])

# Plot 2: After Step 1 (Handle missing values)
axes[1].plot(time, bandpassed, color='#ff7f0e', linewidth=0.8)
axes[1].set_title('Step 1: Missing Values Handled (Interpolation) + Bandpass Filter', fontsize=12, fontweight='bold')
axes[1].set_ylabel('IR Value', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, time[-1])

# Plot 3: After Step 3 (Peak enhancement - normalization)
axes[2].plot(time, enhanced, color='#2ca02c', linewidth=0.8)
axes[2].set_title('Step 3: Peak Enhancement (Normalized to [0,1])', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Normalized Value', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, time[-1])

# Plot 4: After Step 4 (Hampel filter - outlier removal)
axes[3].plot(time, denoised, color='#d62728', linewidth=0.8)
axes[3].scatter(peaks_times, peaks_values , color='blue', s=40, label="Detected Peaks")
axes[3].set_title('Step 4: Hampel Smoothing + Peak Detection', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Normalized Value', fontsize=10)
axes[3].set_xlabel('Time (seconds)', fontsize=10)
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(0, time[-1])

entropy, esqi = pre.compute_esqi(denoised)
print(f"Entropy for segment {time[0]} - {time[-1]} seconds, Entropy: {entropy:.4f}, ESQI: {esqi:.4f}")

plt.tight_layout()

# Save figure
output_path = r"c:\Users\nazeh\BioInfo Trials\3x1-PPG\Dev\3x1-PPG\services\preprocessing\signal_processing_steps.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.show()

# Also create a zoomed-in view of a clean segment (40-60 seconds)
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12))
fig2.suptitle('PPG Signal Processing Steps - Zoomed' , fontsize=16, fontweight='bold')

# Define zoom window
for  i in range(0,60,20):

    start_sec, end_sec = i, i+20
    start_idx = int(start_sec * pre.sampling_rate)
    end_idx = int(end_sec * pre.sampling_rate)
    time_zoom = time[start_idx:end_idx]
    mask = (peaks_times >= start_sec) & (peaks_times < end_sec)

    entropy, esqi = pre.compute_esqi(denoised[start_idx:end_idx])
    print(f"Entropy for segment {start_sec}-{end_sec} seconds: {entropy:.4f}")

    print(f"Zooming into segment: {start_sec}-{end_sec} seconds ({start_idx}-{end_idx} samples)")

    axes2[0].plot(time_zoom, raw_signal[start_idx:end_idx], color='#1f77b4', linewidth=1.2)
    axes2[0].set_title('Original Raw Signal', fontsize=12, fontweight='bold')
    axes2[0].set_ylabel('IR Value', fontsize=10)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(time_zoom, cleaned[start_idx:end_idx], color='#ff7f0e', linewidth=1.2)
    axes2[1].set_title('Step 1: Missing Values Handled + Bandpass ', fontsize=12, fontweight='bold')
    axes2[1].set_ylabel('IR Value', fontsize=10)
    axes2[1].grid(True, alpha=0.3)

    axes2[2].plot(time_zoom, enhanced[start_idx:end_idx], color='#2ca02c', linewidth=1.2)
    axes2[2].set_title('Step 3: Peak Enhancement', fontsize=12, fontweight='bold')
    axes2[2].set_ylabel('Normalized Value', fontsize=10)
    axes2[2].grid(True, alpha=0.3)

    axes2[3].plot(time_zoom, denoised[start_idx:end_idx], color='#d62728', linewidth=1.2)
    axes2[3].scatter(peaks_times[mask], peaks_values[mask] , color='blue', s=40, label="Detected Peaks")
    axes2[3].set_title('Step 4: Hampel Filter Applied', fontsize=12, fontweight='bold')
    axes2[3].set_ylabel('Normalized Value', fontsize=10)
    axes2[3].set_xlabel('Time (seconds)', fontsize=10)
    axes2[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig, axes2 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)


print("\n✓ Visualization complete!")
