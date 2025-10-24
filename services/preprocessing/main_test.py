"""
PPG CSV Preprocessor with 9-Figure Visualization
Loads CSV file, preprocesses signal, and displays segments in 3x3 grid
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import math

class PPGPreprocessor:
    """
    PPG Signal Preprocessor
    Extracts clean 1-second heartbeat segments from raw PPG data
    """

    def __init__(
        self,
        sampling_rate=100,
        segment_length=1.0,
        lowcut=0.5,
        highcut=8.0,
        filter_order=4,
        peak_height_threshold=20,
        peak_distance_factor=0.8,
        similarity_threshold=0.85
    ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = int(sampling_rate * segment_length)
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.peak_height_threshold = peak_height_threshold
        self.peak_distance = int(peak_distance_factor * sampling_rate)
        self.similarity_threshold = similarity_threshold

    def handle_missing_values(self, ppg_signal):
        """Fill missing values using linear interpolation"""
        ppg = np.asarray(ppg_signal, dtype=np.float64).copy()

        valid_mask = np.isfinite(ppg)
        if not valid_mask.any():
            return None

        missing_ratio = (~valid_mask).sum() / len(ppg)
        if missing_ratio > 0.5:
            return None

        if not valid_mask.all():
            indices = np.arange(len(ppg))
            ppg[~valid_mask] = np.interp(
                indices[~valid_mask],
                indices[valid_mask],
                ppg[valid_mask]
            )

        return ppg

    def bandpass_filter(self, ppg_signal):
        """Apply Butterworth bandpass filter (0.5-8 Hz)"""
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = signal.butter(self.filter_order, [low, high], btype='band')

        try:
            filtered_signal = signal.filtfilt(b, a, ppg_signal)
        except ValueError:
            return None

        return filtered_signal

    def detect_peaks(self, ppg_signal):
        """Detect heartbeat peaks"""
        peaks, _ = signal.find_peaks(
            ppg_signal,
            height=self.peak_height_threshold,
            distance=self.peak_distance
        )
        return peaks

    def extract_peak_centered_windows(self, ppg_signal, peaks):
        """Extract 1-second windows centered on each peak"""
        windows = []
        window_half = self.segment_samples // 2

        for peak in peaks:
            window_start = max(0, peak - window_half)
            window_end = min(len(ppg_signal), peak + window_half)

            if window_end - window_start == self.segment_samples:
                window = ppg_signal[window_start:window_end]
                if np.isfinite(window).all():
                    windows.append(window)

        if len(windows) == 0:
            return np.array([])

        return np.array(windows)

    def compute_template(self, windows):
        """Compute average heartbeat template"""
        return np.mean(windows, axis=0)

    def compute_cosine_similarity(self, window, template):
        """Calculate cosine similarity between window and template"""
        dot_product = np.sum(window * template)
        magnitude_window = np.sqrt(np.sum(window ** 2))
        magnitude_template = np.sqrt(np.sum(template ** 2))

        if magnitude_window == 0 or magnitude_template == 0:
            return 0.0

        similarity = dot_product / (magnitude_window * magnitude_template)
        return similarity

    def filter_windows_by_similarity(self, windows, template):
        """Keep only windows similar to template (>85% match)"""
        filtered_windows = []

        for window in windows:
            similarity = self.compute_cosine_similarity(window, template)
            if similarity >= self.similarity_threshold:
                filtered_windows.append(window)

        if len(filtered_windows) == 0:
            return np.array([])

        return np.array(filtered_windows)

    def preprocess(self, ppg_signal, apply_template_matching=True):
        """Complete preprocessing pipeline"""
        # Step 1: Handle missing values
        cleaned_signal = self.handle_missing_values(ppg_signal)
        if cleaned_signal is None:
            print("ERROR: Too many missing values")
            return None

        # Step 2: Bandpass filtering
        filtered_signal = self.bandpass_filter(cleaned_signal)
        if filtered_signal is None:
            print("ERROR: Filtering failed")
            return None

        # Step 3: Detect peaks
        peaks = self.detect_peaks(filtered_signal)
        if len(peaks) == 0:
            print("ERROR: No peaks detected")
            return None
        print(f"Detected {len(peaks)} peaks")

        # Step 4: Extract windows
        windows = self.extract_peak_centered_windows(filtered_signal, peaks)
        if len(windows) == 0:
            print("ERROR: No valid windows extracted")
            return None
        print(f"Extracted {len(windows)} windows")

        # Step 5: Template matching
        if apply_template_matching and len(windows) > 1:
            template = self.compute_template(windows)
            filtered_windows = self.filter_windows_by_similarity(windows, template)

            if len(filtered_windows) == 0:
                print("WARNING: No windows passed similarity filter, returning all")
                return windows

            print(f"After template matching: {len(filtered_windows)} windows")
            return filtered_windows

        return windows


def load_ppg_from_csv(csv_path):
    """
    Load PPG data from CSV file
    Supports multiple CSV formats
    """
    df = pd.read_csv(csv_path)
    Sample_Index = df['Sample_Index']
    ppg_data = df['IR_Value']
    timestamps = df['Timestamp_ms']
    return ppg_data, timestamps, Sample_Index


def visualize_segments_grid(segments, title="Preprocessed PPG Segments", rows=3, cols=3):
    """
    Visualize preprocessed PPG segments (NumPy array or DataFrame)
    in multiple 3x3 grids (shows all segments).
    """
    # Convert to DataFrame if it's a NumPy array
    if isinstance(segments, np.ndarray):
        segments = pd.DataFrame(segments)

    n_segments = len(segments)
    segments_per_fig = rows * cols
    n_figures = math.ceil(n_segments / segments_per_fig)

    if n_segments == 0:
        print("No segments to display!")
        return

    time_axis = np.linspace(0, 1, segments.shape[1])

    for fig_idx in range(n_figures):
        start_idx = fig_idx * segments_per_fig
        end_idx = min(start_idx + segments_per_fig, n_segments)
        subset = segments.iloc[start_idx:end_idx]

        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        fig.suptitle(f"{title} (Group {fig_idx + 1}/{n_figures})", fontsize=16, fontweight='bold')

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx in range(segments_per_fig):
            ax = axes[idx]
            global_idx = start_idx + idx

            if global_idx < n_segments:
                segment = subset.iloc[idx].values
                ax.plot(time_axis, segment, 'b-', linewidth=1.5, alpha=0.8)

                # Mark peak
                peak_idx = np.argmax(segment)
                ax.plot(time_axis[peak_idx], segment[peak_idx], 'ro', markersize=8, label='Peak')

                # Labels and style
                ax.set_title(f'Segment {global_idx + 1}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Time (seconds)', fontsize=9)
                ax.set_ylabel('Amplitude', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.legend(fontsize=8)

                # Stats box
                stats_text = f'μ={segment.mean():.1f}\nσ={segment.std():.1f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def visualize_all_segments_overlay(segments, title="All Preprocessed Segments Overlay"):
    """
    Visualize all segments overlaid on same plot
    Shows consistency of preprocessing
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_axis = np.linspace(0, 1, segments.shape[1])
    
    # Plot all segments
    for i, segment in enumerate(segments):
        alpha = 0.3 if len(segments) > 20 else 0.5
        ax.plot(time_axis, segment, alpha=alpha, linewidth=1)
    
    # Plot mean template
    template = np.mean(segments, axis=0)
    ax.plot(time_axis, template, 'r-', linewidth=3, label='Mean Template', alpha=0.9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def main():
    """
    Main processing pipeline
    """
    print("=" * 70)
    print("PPG CSV PREPROCESSOR WITH VISUALIZATION")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: CONFIGURE INPUT FILE
    # ========================================================================
    
    # CHANGE THIS to your CSV file path
    csv_file = Path(r"Dev\3x1-PPG\services\ble_receiver\ppg_data\ppg_data_20251024_040102.csv")
    # Or use any of these formats:
    # csv_file = "max30102_data_20251009_151555.csv"
    # csv_file = "max30102_processed_20251013_143045.csv"
    
    if not Path(csv_file).exists():
        print(f"\nERROR: File not found: {csv_file}")
        print("\nPlease update the 'csv_file' variable with your CSV path")
        return
    
    # ========================================================================
    # STEP 2: LOAD DATA
    # ========================================================================
    
    ppg_signal , timestamps , Sample_index = load_ppg_from_csv(csv_file)

    print("\n" + "=" * 70)
    print("LOADING CSV DATA")  
    print(ppg_signal.head())
    print(timestamps.head())

    
    # ========================================================================
    # STEP 3: CONFIGURE PREPROCESSOR
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("PREPROCESSING CONFIGURATION")
    print("=" * 70)
    
    preprocessor = PPGPreprocessor(
        sampling_rate=100,              # 100 Hz (adjust if different)
        segment_length=1.0,             # 1-second windows
        lowcut=0.1,                     # Remove frequencies < 0.5 Hz
        highcut=8.0,                    # Remove frequencies > 8 Hz
        filter_order=4,                 # Butterworth filter order
        peak_height_threshold=30,       # Minimum peak height (ADJUST THIS!)
        peak_distance_factor=0.8,       # Min distance between peaks
        similarity_threshold=0.65       # Template matching threshold 
    )
    
    print(f"Sample rate: {preprocessor.sampling_rate} Hz")
    print(f"Segment length: {preprocessor.segment_length} sec")
    print(f"Segment samples: {preprocessor.segment_samples}")
    print(f"Bandpass filter: {preprocessor.lowcut}-{preprocessor.highcut} Hz")
    print(f"Peak height threshold: {preprocessor.peak_height_threshold}")
    print(f"Similarity threshold: {preprocessor.similarity_threshold}")
    
    # ========================================================================
    # STEP 4: PREPROCESS SIGNAL
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("PREPROCESSING SIGNAL")
    print("=" * 70)
    
    segments = preprocessor.preprocess(ppg_signal, apply_template_matching=True)
    
    if segments is None or len(segments) == 0:
        print("\n✗ PREPROCESSING FAILED!")
        print("\nTroubleshooting:")
        print("1. Adjust peak_height_threshold (try 10, 15, 30, 50)")
        print("2. Check if CSV contains valid PPG data")
        print("3. Verify sampling rate is correct")
        return
    
    print(f"\n✓ SUCCESS!")
    print(f"✓ Extracted {len(segments)} clean segments")
    print(f"✓ Each segment: {segments.shape[1]} samples (1 second)")
    
    # ========================================================================
    # STEP 4.5: SAVE SEGMENTS TO CSV
    # ========================================================================

    print("\n" + "=" * 70)
    print("SAVING SEGMENTS TO CSV")
    print("=" * 70)

    # Convert to DataFrame
    segments_df = pd.DataFrame(segments)

    # Generate timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("Dev/3x1-PPG/services/preprocessing/processed_segments")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"segments_{timestamp}.csv"

    # Save to CSV
    segments_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(segments_df)} segments to: {output_file}")

    # Optional: Print shape for confirmation
    print(f"Segments CSV shape: {segments_df.shape}")

    # ========================================================================
    # STEP 5: DISPLAY STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("SEGMENT STATISTICS")
    print("=" * 70)
    
    print(f"Total segments: {len(segments)}")
    print(f"Segment shape: {segments.shape}")
    print(f"\nSignal statistics:")
    print(f"  Mean:   {segments.mean():.2f}")
    print(f"  Std:    {segments.std():.2f}")
    print(f"  Min:    {segments.min():.2f}")
    print(f"  Max:    {segments.max():.2f}")
    print(f"  Median: {np.median(segments):.2f}")
    
    # ========================================================================
    # STEP 6: VISUALIZE IN 3x3 GRID
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print("Displaying first 9 segments in 3x3 grid...")
    
    visualize_segments_grid(
        segments, 
        title=f"Preprocessed PPG Segments (Total: {len(segments)})",
        rows=3, 
        cols=3
    )
    
    # ========================================================================
    # STEP 7: SHOW OVERLAY PLOT
    # ========================================================================
    
    print("\nDisplaying all segments overlaid...")
    visualize_all_segments_overlay(
        segments,
        title=f"All {len(segments)} Segments Overlay"
    )
    
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✓ {len(segments)} segments ready for machine learning")
    print(f"✓ Shape: {segments.shape} (segments × samples)")
    print("\nNext steps:")
    print("  2. Extract features for analysis")
    print("  3. Compare with other collections")


if __name__ == "__main__":
    main()