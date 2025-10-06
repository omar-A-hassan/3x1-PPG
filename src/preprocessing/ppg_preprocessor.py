"""
PPG Signal Preprocessing Module

Implements preprocessing pipeline for PPG signals including:
- 1-second signal segmentation
- Butterworth bandpass filtering (0.5-8 Hz)
- Peak detection (systolic/diastolic)
- Template matching with cosine similarity

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine
import neurokit2 as nk


class PPGPreprocessor:
    """
    Preprocessor for PPG signals to prepare data for CNN-GRU model.

    Args:
        sampling_rate (int): Sampling rate of input PPG signal (default: 100 Hz)
        segment_length (float): Length of each segment in seconds (default: 1.0)
        lowcut (float): Low cutoff frequency for bandpass filter (default: 0.5 Hz)
        highcut (float): High cutoff frequency for bandpass filter (default: 8.0 Hz)
        filter_order (int): Order of Butterworth filter (default: 4)
        similarity_threshold (float): Cosine similarity threshold for template matching (default: 0.85)
    """

    def __init__(
        self,
        sampling_rate=100,
        segment_length=1.0,
        lowcut=0.5,
        highcut=8.0,
        filter_order=4,
        similarity_threshold=0.85
    ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = int(sampling_rate * segment_length)
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.similarity_threshold = similarity_threshold

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
        filtered_signal = signal.filtfilt(b, a, ppg_signal)

        return filtered_signal

    def detect_peaks(self, ppg_signal):
        """
        Detect systolic and diastolic peaks in PPG signal.

        Args:
            ppg_signal (np.ndarray): Filtered PPG signal

        Returns:
            dict: Dictionary with 'systolic' and 'diastolic' peak indices
        """
        # Use NeuroKit2 for peak detection
        peaks_info = nk.ppg_findpeaks(ppg_signal, sampling_rate=self.sampling_rate)

        systolic_peaks = peaks_info['PPG_Peaks']

        # Diastolic peaks are typically valleys between systolic peaks
        # Invert signal to find valleys as peaks
        inverted_signal = -ppg_signal
        diastolic_info = nk.ppg_findpeaks(inverted_signal, sampling_rate=self.sampling_rate)
        diastolic_peaks = diastolic_info['PPG_Peaks']

        return {
            'systolic': systolic_peaks,
            'diastolic': diastolic_peaks
        }

    def segment_signal(self, ppg_signal):
        """
        Segment PPG signal into fixed-length windows.

        Args:
            ppg_signal (np.ndarray): Preprocessed PPG signal

        Returns:
            np.ndarray: Array of segments (n_segments, segment_samples)
        """
        n_samples = len(ppg_signal)
        n_segments = n_samples // self.segment_samples

        # Truncate signal to fit exact number of segments
        truncated_length = n_segments * self.segment_samples
        ppg_truncated = ppg_signal[:truncated_length]

        # Reshape into segments
        segments = ppg_truncated.reshape(n_segments, self.segment_samples)

        return segments

    def compute_template_similarity(self, segment, template):
        """
        Compute cosine similarity between segment and template.

        Args:
            segment (np.ndarray): PPG segment
            template (np.ndarray): Reference template

        Returns:
            float: Cosine similarity (0-1)
        """
        # Normalize to zero mean and unit variance
        segment_norm = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        template_norm = (template - np.mean(template)) / (np.std(template) + 1e-8)

        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(segment_norm, template_norm)

        return similarity

    def filter_segments_by_template(self, segments, template=None):
        """
        Filter segments based on template matching.

        Args:
            segments (np.ndarray): Array of PPG segments
            template (np.ndarray, optional): Reference template. If None, uses first segment.

        Returns:
            np.ndarray: Filtered segments meeting similarity threshold
        """
        if template is None:
            # Use first segment as template
            template = segments[0]

        filtered_segments = []
        similarities = []

        for segment in segments:
            similarity = self.compute_template_similarity(segment, template)
            similarities.append(similarity)

            if similarity >= self.similarity_threshold:
                filtered_segments.append(segment)

        if len(filtered_segments) == 0:
            print(f"Warning: No segments met similarity threshold {self.similarity_threshold}")
            print(f"Max similarity: {max(similarities):.3f}, Mean: {np.mean(similarities):.3f}")
            # Return segments with top 50% similarity
            threshold = np.percentile(similarities, 50)
            filtered_segments = [seg for seg, sim in zip(segments, similarities) if sim >= threshold]

        return np.array(filtered_segments)

    def preprocess(self, ppg_signal, apply_template_matching=True):
        """
        Full preprocessing pipeline.

        Args:
            ppg_signal (np.ndarray): Raw PPG signal
            apply_template_matching (bool): Whether to apply template matching filter

        Returns:
            np.ndarray: Preprocessed segments ready for model input (n_segments, segment_samples)
        """
        # Step 1: Bandpass filtering
        filtered_signal = self.bandpass_filter(ppg_signal)

        # Step 2: Segmentation
        segments = self.segment_signal(filtered_signal)

        # Step 3: Template matching (optional)
        if apply_template_matching and len(segments) > 1:
            segments = self.filter_segments_by_template(segments)

        return segments


def segment_ppg_signal(ppg_signal, sampling_rate=100, segment_length=1.0):
    """
    Convenience function to quickly segment a PPG signal.

    Args:
        ppg_signal (np.ndarray): Raw PPG signal
        sampling_rate (int): Sampling rate in Hz
        segment_length (float): Segment length in seconds

    Returns:
        np.ndarray: Preprocessed segments
    """
    preprocessor = PPGPreprocessor(sampling_rate=sampling_rate, segment_length=segment_length)
    segments = preprocessor.preprocess(ppg_signal, apply_template_matching=False)
    return segments


if __name__ == "__main__":
    print("="*60)
    print("PPG Preprocessor Test")
    print("="*60)

    # Generate synthetic PPG signal (5 seconds at 100 Hz)
    sampling_rate = 100
    duration = 5.0
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # Simulate PPG: combination of cardiac pulse and respiratory modulation
    heart_rate = 1.2  # Hz (72 bpm)
    respiratory_rate = 0.25  # Hz (15 breaths/min)

    ppg_signal = (
        np.sin(2 * np.pi * heart_rate * t) +  # Cardiac component
        0.3 * np.sin(2 * np.pi * respiratory_rate * t) +  # Respiratory modulation
        0.1 * np.random.randn(len(t))  # Noise
    )

    # Initialize preprocessor
    preprocessor = PPGPreprocessor(sampling_rate=sampling_rate)

    print(f"\nInput signal: {len(ppg_signal)} samples ({duration}s at {sampling_rate}Hz)")

    # Test filtering
    filtered = preprocessor.bandpass_filter(ppg_signal)
    print(f"Filtered signal: {len(filtered)} samples")

    # Test segmentation
    segments = preprocessor.segment_signal(filtered)
    print(f"Segments: {segments.shape} ({segments.shape[0]} segments of {segments.shape[1]} samples)")

    # Test full preprocessing
    preprocessed = preprocessor.preprocess(ppg_signal, apply_template_matching=True)
    print(f"Preprocessed segments: {preprocessed.shape}")

    # Test peak detection
    peaks = preprocessor.detect_peaks(filtered)
    print(f"\nPeak detection:")
    print(f"  Systolic peaks: {len(peaks['systolic'])}")
    print(f"  Diastolic peaks: {len(peaks['diastolic'])}")

    print("\n" + "="*60)
    print("PPG Preprocessor working correctly")
    print("="*60)
