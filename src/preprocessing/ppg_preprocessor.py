"""
PPG Signal Preprocessing Module

Implements preprocessing pipeline for PPG signals including:
- 1-second peak-centered signal segmentation
- Butterworth bandpass filtering (0.5-8 Hz)
- Peak detection with height and distance thresholds
- Template matching with cosine similarity
- NaN/missing value handling

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)
Algorithm 1: Precision Interval Segmentation
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine


class PPGPreprocessor:
    """
    Preprocessor for PPG signals following the paper's exact methodology.

    Args:
        sampling_rate (int): Sampling rate of input PPG signal (default: 100 Hz)
        segment_length (float): Length of each segment in seconds (default: 1.0)
        lowcut (float): Low cutoff frequency for bandpass filter (default: 0.5 Hz)
        highcut (float): High cutoff frequency for bandpass filter (default: 8.0 Hz)
        filter_order (int): Order of Butterworth filter (default: 4)
        peak_height_threshold (float): Minimum peak height for detection (default: 20)
        peak_distance_factor (float): Minimum distance between peaks as factor of fs (default: 0.8)
        similarity_threshold (float): Cosine similarity threshold for template matching (default: 0.85)
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
        """
        Handle missing values (NaN/inf) in PPG signal using forward fill.
        If entire signal is NaN or has too many missing values, return None.

        Args:
            ppg_signal (np.ndarray): Raw PPG signal

        Returns:
            np.ndarray or None: Cleaned signal, or None if signal is invalid
        """
        ppg = np.asarray(ppg_signal, dtype=np.float64).copy()

        # Check if all values are NaN/inf
        valid_mask = np.isfinite(ppg)
        if not valid_mask.any():
            return None

        # If more than 50% missing, reject the signal
        missing_ratio = (~valid_mask).sum() / len(ppg)
        if missing_ratio > 0.5:
            return None

        # Forward fill missing values
        if not valid_mask.all():
            indices = np.arange(len(ppg))
            ppg[~valid_mask] = np.interp(
                indices[~valid_mask],
                indices[valid_mask],
                ppg[valid_mask]
            )

        return ppg

    def bandpass_filter(self, ppg_signal):
        """
        Apply Butterworth bandpass filter to remove noise.

        Args:
            ppg_signal (np.ndarray): Raw PPG signal

        Returns:
            np.ndarray: Filtered PPG signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = signal.butter(self.filter_order, [low, high], btype='band')

        # Use filtfilt for zero-phase filtering
        try:
            filtered_signal = signal.filtfilt(b, a, ppg_signal)
        except ValueError:
            # If filtfilt fails, return None
            return None

        return filtered_signal

    def detect_peaks(self, ppg_signal):
        """
        Detect peaks in PPG signal using height and distance thresholds.
        Following Algorithm 1 from the paper.

        Args:
            ppg_signal (np.ndarray): Filtered PPG signal

        Returns:
            np.ndarray: Array of peak indices
        """
        peaks, _ = signal.find_peaks(
            ppg_signal,
            height=self.peak_height_threshold,
            distance=self.peak_distance
        )

        return peaks

    def extract_peak_centered_windows(self, ppg_signal, peaks):
        """
        Extract 1-second windows centered around each detected peak.
        Following Algorithm 1: EXTRACTWINDOWS function.

        Args:
            ppg_signal (np.ndarray): Filtered PPG signal
            peaks (np.ndarray): Array of peak indices

        Returns:
            np.ndarray: Array of windows (n_windows, segment_samples)
        """
        windows = []
        window_half = self.segment_samples // 2

        for peak in peaks:
            # Calculate window boundaries centered on peak
            window_start = max(0, peak - window_half)
            window_end = min(len(ppg_signal), peak + window_half)

            # Only keep windows with exactly segment_samples
            if window_end - window_start == self.segment_samples:
                window = ppg_signal[window_start:window_end]
                # Check for NaN/inf in window
                if np.isfinite(window).all():
                    windows.append(window)

        if len(windows) == 0:
            return np.array([])

        return np.array(windows)

    def compute_template(self, windows):
        """
        Compute template as mean of all windows.
        Following Algorithm 1: COMPUTETEMPLATE function.

        Args:
            windows (np.ndarray): Array of windows

        Returns:
            np.ndarray: Template (mean of windows)
        """
        return np.mean(windows, axis=0)

    def compute_cosine_similarity(self, window, template):
        """
        Compute cosine similarity between window and template.
        Following Algorithm 1: COSINESIMILARITY function.

        Args:
            window (np.ndarray): PPG window
            template (np.ndarray): Reference template

        Returns:
            float: Cosine similarity (0-1)
        """
        # Compute dot product and magnitudes
        dot_product = np.sum(window * template)
        magnitude_window = np.sqrt(np.sum(window ** 2))
        magnitude_template = np.sqrt(np.sum(template ** 2))

        # Avoid division by zero
        if magnitude_window == 0 or magnitude_template == 0:
            return 0.0

        # Cosine similarity
        similarity = dot_product / (magnitude_window * magnitude_template)

        return similarity

    def filter_windows_by_similarity(self, windows, template):
        """
        Filter windows based on cosine similarity with template.
        Following Algorithm 1: FILTERWINDOWSBYSIMILARITY function.

        Args:
            windows (np.ndarray): Array of windows
            template (np.ndarray): Reference template

        Returns:
            np.ndarray: Filtered windows meeting similarity threshold
        """
        filtered_windows = []

        for window in windows:
            similarity = self.compute_cosine_similarity(window, template)
            if similarity >= self.similarity_threshold:
                filtered_windows.append(window)

        if len(filtered_windows) == 0:
            return np.array([])

        return np.array(filtered_windows)

    def preprocess(self, ppg_signal, apply_template_matching=True):
        """
        Full preprocessing pipeline following Algorithm 1: MAIN function.

        Args:
            ppg_signal (np.ndarray): Raw PPG signal
            apply_template_matching (bool): Whether to apply template matching filter

        Returns:
            np.ndarray or None: Preprocessed segments (n_segments, segment_samples),
                               or None if preprocessing fails
        """
        # Step 1: Handle missing values
        cleaned_signal = self.handle_missing_values(ppg_signal)
        if cleaned_signal is None:
            return None

        # Step 2: Bandpass filtering
        filtered_signal = self.bandpass_filter(cleaned_signal)
        if filtered_signal is None:
            return None

        # Step 3: Detect peaks
        peaks = self.detect_peaks(filtered_signal)
        if len(peaks) == 0:
            return None

        # Step 4: Extract windows centered on peaks
        windows = self.extract_peak_centered_windows(filtered_signal, peaks)
        if len(windows) == 0:
            return None

        # Step 5: Template matching (if enabled)
        if apply_template_matching and len(windows) > 1:
            template = self.compute_template(windows)
            filtered_windows = self.filter_windows_by_similarity(windows, template)

            # If no windows pass, return original windows
            if len(filtered_windows) == 0:
                return windows

            return filtered_windows

        return windows


def segment_ppg_signal(ppg_signal, sampling_rate=100, segment_length=1.0):
    """
    Convenience function to quickly segment a PPG signal.

    Args:
        ppg_signal (np.ndarray): Raw PPG signal
        sampling_rate (int): Sampling rate in Hz
        segment_length (float): Segment length in seconds

    Returns:
        np.ndarray or None: Preprocessed segments, or None if preprocessing fails
    """
    preprocessor = PPGPreprocessor(sampling_rate=sampling_rate, segment_length=segment_length)
    segments = preprocessor.preprocess(ppg_signal, apply_template_matching=False)
    return segments


if __name__ == "__main__":
    print("="*60)
    print("PPG Preprocessor Test (Paper Method)")
    print("="*60)

    # Generate synthetic PPG signal (10 seconds at 100 Hz)
    sampling_rate = 100
    duration = 10.0
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # Simulate PPG: combination of cardiac pulse (1.2 Hz = 72 bpm)
    heart_rate = 1.2  # Hz
    ppg_signal = 30 * np.sin(2 * np.pi * heart_rate * t) + 0.5 * np.random.randn(len(t))

    # Initialize preprocessor
    preprocessor = PPGPreprocessor(
        sampling_rate=sampling_rate,
        peak_height_threshold=20,
        similarity_threshold=0.85
    )

    print(f"\nInput signal: {len(ppg_signal)} samples ({duration}s at {sampling_rate}Hz)")

    # Test full preprocessing
    segments = preprocessor.preprocess(ppg_signal, apply_template_matching=True)

    if segments is not None and len(segments) > 0:
        print(f"Successfully extracted {len(segments)} 1-second windows")
        print(f"Window shape: {segments[0].shape}")
        print(f"Segment statistics:")
        print(f"  Mean: {segments.mean():.2f}")
        print(f"  Std: {segments.std():.2f}")
        print(f"  Min: {segments.min():.2f}")
        print(f"  Max: {segments.max():.2f}")
    else:
        print("No valid segments extracted")

    print("\n" + "="*60)
    print("PPG Preprocessor Test Complete")
    print("="*60)
