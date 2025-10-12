//
//  PreprocessingService.swift
//  PPG Glucose Monitor
//
//  Handles PPG signal preprocessing pipeline
//  Replicates the exact preprocessing from dl-model branch:
//  1. Butterworth bandpass filter (0.5-8.0 Hz)
//  2. Peak detection
//  3. Extract 1-second windows centered on peaks
//  4. Template matching (cosine similarity > 0.85)
//  5. Normalize segments (mean=0, std=1)
//

import Foundation
import Accelerate

// MARK: - Preprocessing Configuration
struct PreprocessingConfig {
    let samplingRate: Double = 100.0        // Hz
    let segmentLength: Double = 1.0         // seconds
    let segmentSamples: Int = 100           // samples per segment
    
    // Butterworth bandpass filter
    let lowcut: Double = 0.5                // Hz
    let highcut: Double = 8.0               // Hz
    let filterOrder: Int = 4
    
    // Peak detection
    let peakHeightThreshold: Double = 20.0
    let peakDistance: Int = 80              // samples (0.8s at 100Hz)
    
    // Template matching
    let similarityThreshold: Double = 0.85
}

// MARK: - Preprocessing Result
struct PreprocessingResult {
    let success: Bool
    let segments: [[Double]]?               // Array of 100-sample segments
    let numSegments: Int
    let qualityScore: Double
    let errorMessage: String?
    
    var segmentsForModel: [[[Double]]]? {
        // Convert to shape (N, 100, 1) for CoreML
        guard let segments = segments else { return nil }
        return segments.map { segment in
            segment.map { [$0] }
        }
    }
}

// MARK: - Preprocessing Service
class PreprocessingService {
    
    private let config = PreprocessingConfig()
    
    // MARK: - Public Methods
    
    /// Preprocess raw PPG signal from ESP32
    func preprocess(rawPPG: [UInt16]) -> PreprocessingResult {
        
        // Step 0: Validate input
        guard rawPPG.count == 6000 else {
            return PreprocessingResult(
                success: false,
                segments: nil,
                numSegments: 0,
                qualityScore: 0.0,
                errorMessage: "Expected 6000 samples, got \(rawPPG.count)"
            )
        }
        
        // Convert UInt16 to Double
        let signal = rawPPG.map { Double($0) }
        
        // Check signal quality
        if let qualityError = checkSignalQuality(signal) {
            return PreprocessingResult(
                success: false,
                segments: nil,
                numSegments: 0,
                qualityScore: 0.0,
                errorMessage: qualityError
            )
        }
        
        // Step 1: Handle missing values (forward fill)
        let cleanedSignal = handleMissingValues(signal)
        
        // Step 2: Butterworth bandpass filter
        guard let filteredSignal = butterworthBandpassFilter(cleanedSignal) else {
            return PreprocessingResult(
                success: false,
                segments: nil,
                numSegments: 0,
                qualityScore: 0.0,
                errorMessage: "Butterworth filter failed"
            )
        }
        
        // Step 3: Detect peaks
        let peaks = detectPeaks(filteredSignal)
        
        guard !peaks.isEmpty else {
            return PreprocessingResult(
                success: false,
                segments: nil,
                numSegments: 0,
                qualityScore: 0.0,
                errorMessage: "No peaks detected - poor signal quality"
            )
        }
        
        // Step 4: Extract peak-centered windows
        let windows = extractPeakCenteredWindows(filteredSignal, peaks: peaks)
        
        guard !windows.isEmpty else {
            return PreprocessingResult(
                success: false,
                segments: nil,
                numSegments: 0,
                qualityScore: 0.0,
                errorMessage: "No valid windows extracted"
            )
        }
        
        // Step 5: Template matching (optional but recommended)
        let filteredWindows = templateMatching(windows)
        
        guard filteredWindows.count >= 10 else {
            return PreprocessingResult(
                success: false,
                segments: nil,
                numSegments: 0,
                qualityScore: 0.0,
                errorMessage: "Too few segments after filtering (\(filteredWindows.count) < 10)"
            )
        }
        
        // Step 6: Normalize each segment
        let normalizedSegments = filteredWindows.map { normalizeSegment($0) }
        
        // Compute quality score
        let qualityScore = computeQualityScore(
            rawSignal: signal,
            segments: normalizedSegments
        )
        
        return PreprocessingResult(
            success: true,
            segments: normalizedSegments,
            numSegments: normalizedSegments.count,
            qualityScore: qualityScore,
            errorMessage: nil
        )
    }
    
    // MARK: - Step 0: Signal Quality Check
    
    private func checkSignalQuality(_ signal: [Double]) -> String? {
        // Check for all zeros
        if signal.allSatisfy({ $0 == 0.0 }) {
            return "Signal is all zeros - sensor not working"
        }
        
        // Check for constant signal
        let std = standardDeviation(signal)
        if std < 10.0 {
            return "Signal has no variation - poor sensor contact"
        }
        
        // Check for saturation
        let maxValue = signal.max() ?? 0.0
        if maxValue > 60000.0 {
            return "Signal saturated - reduce finger pressure"
        }
        
        // Check for too low signal
        let meanValue = mean(signal)
        if meanValue < 1000.0 {
            return "Signal too weak - improve sensor contact"
        }
        
        return nil
    }
    
    // MARK: - Step 1: Handle Missing Values
    
    private func handleMissingValues(_ signal: [Double]) -> [Double] {
        var cleaned = signal
        
        // Forward fill any NaN or inf values
        var lastValid = signal.first ?? 0.0
        for i in 0..<cleaned.count {
            if cleaned[i].isNaN || cleaned[i].isInfinite {
                cleaned[i] = lastValid
            } else {
                lastValid = cleaned[i]
            }
        }
        
        return cleaned
    }
    
    // MARK: - Step 2: Butterworth Bandpass Filter
    
    private func butterworthBandpassFilter(_ signal: [Double]) -> [Double]? {
        // 4th order Butterworth bandpass filter: 0.5-8.0 Hz @ 100 Hz sampling rate
        // Implemented as cascade of 2nd order sections (biquads) using Accelerate
        
        let nyquist = config.samplingRate / 2.0
        let lowNorm = config.lowcut / nyquist
        let highNorm = config.highcut / nyquist
        
        // Design 4th order Butterworth bandpass as cascade of two 2nd order sections
        // Section 1: Highpass (removes DC and low frequencies)
        let hpCoeffs = designButterworthHighpass(cutoff: lowNorm, sampleRate: config.samplingRate)
        
        // Section 2: Lowpass (removes high frequency noise)
        let lpCoeffs = designButterworthLowpass(cutoff: highNorm, sampleRate: config.samplingRate)
        
        // Apply highpass filter
        guard var filtered = applyBiquadFilter(signal, coefficients: hpCoeffs) else {
            return nil
        }
        
        // Apply lowpass filter
        filtered = applyBiquadFilter(filtered, coefficients: lpCoeffs) ?? filtered
        
        return filtered
    }
    
    private func designButterworthHighpass(cutoff: Double, sampleRate: Double) -> [Double] {
        // 2nd order Butterworth highpass biquad coefficients
        // Using bilinear transform
        
        let w0 = 2.0 * .pi * cutoff / sampleRate
        let cosw0 = cos(w0)
        let sinw0 = sin(w0)
        let alpha = sinw0 / (2.0 * sqrt(2.0))  // Q = 1/sqrt(2) for Butterworth
        
        // Highpass biquad coefficients
        let b0 = (1.0 + cosw0) / 2.0
        let b1 = -(1.0 + cosw0)
        let b2 = (1.0 + cosw0) / 2.0
        let a0 = 1.0 + alpha
        let a1 = -2.0 * cosw0
        let a2 = 1.0 - alpha
        
        // Normalize by a0
        return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]
    }
    
    private func designButterworthLowpass(cutoff: Double, sampleRate: Double) -> [Double] {
        // 2nd order Butterworth lowpass biquad coefficients
        // Using bilinear transform
        
        let w0 = 2.0 * .pi * cutoff / sampleRate
        let cosw0 = cos(w0)
        let sinw0 = sin(w0)
        let alpha = sinw0 / (2.0 * sqrt(2.0))  // Q = 1/sqrt(2) for Butterworth
        
        // Lowpass biquad coefficients
        let b0 = (1.0 - cosw0) / 2.0
        let b1 = 1.0 - cosw0
        let b2 = (1.0 - cosw0) / 2.0
        let a0 = 1.0 + alpha
        let a1 = -2.0 * cosw0
        let a2 = 1.0 - alpha
        
        // Normalize by a0
        return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0]
    }
    
    private func applyBiquadFilter(_ signal: [Double], coefficients: [Double]) -> [Double]? {
        // Apply biquad filter using direct form II transposed structure
        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        
        guard coefficients.count == 5 else { return nil }
        
        let b0 = coefficients[0]
        let b1 = coefficients[1]
        let b2 = coefficients[2]
        let a1 = coefficients[3]
        let a2 = coefficients[4]
        
        var filtered = [Double](repeating: 0.0, count: signal.count)
        var x1 = 0.0, x2 = 0.0  // Input delays
        var y1 = 0.0, y2 = 0.0  // Output delays
        
        for i in 0..<signal.count {
            let x0 = signal[i]
            
            // Direct form II transposed
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            
            // Update delays
            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0
            
            filtered[i] = y0
        }
        
        return filtered
    }
    
    // Alternative: Using vDSP for even better performance
    private func applyBiquadFilterAccelerate(_ signal: [Double], coefficients: [Double]) -> [Double]? {
        // This uses Accelerate framework's vDSP_biquad for optimal performance
        // Coefficients format: [b0, b1, b2, 1.0, a1, a2]
        
        guard coefficients.count == 5 else { return nil }
        
        var filtered = [Double](repeating: 0.0, count: signal.count)
        var delays = [Double](repeating: 0.0, count: 4)  // Biquad state
        
        // vDSP biquad expects coefficients in specific format
        var vdspCoeffs = [coefficients[0], coefficients[1], coefficients[2], 
                          1.0, coefficients[3], coefficients[4]]
        
        signal.withUnsafeBufferPointer { signalPtr in
            filtered.withUnsafeMutableBufferPointer { filteredPtr in
                vdspCoeffs.withUnsafeMutableBufferPointer { coeffsPtr in
                    delays.withUnsafeMutableBufferPointer { delaysPtr in
                        vDSP_biquadD(
                            coeffsPtr.baseAddress!,
                            delaysPtr.baseAddress!,
                            signalPtr.baseAddress!,
                            1,  // stride
                            filteredPtr.baseAddress!,
                            1,  // stride
                            vDSP_Length(signal.count)
                        )
                    }
                }
            }
        }
        
        return filtered
    }
    
    // MARK: - Step 3: Peak Detection
    
    private func detectPeaks(_ signal: [Double]) -> [Int] {
        var peaks: [Int] = []
        let distance = config.peakDistance
        let threshold = config.peakHeightThreshold
        
        // Find local maxima
        for i in distance..<(signal.count - distance) {
            let value = signal[i]
            
            // Check if it's a local maximum
            var isMax = true
            for j in (i-distance)...(i+distance) {
                if j != i && signal[j] >= value {
                    isMax = false
                    break
                }
            }
            
            // Check height threshold
            if isMax && value > threshold {
                // Check distance from last peak
                if peaks.isEmpty || (i - peaks.last! >= distance) {
                    peaks.append(i)
                }
            }
        }
        
        return peaks
    }
    
    // MARK: - Step 4: Extract Peak-Centered Windows
    
    private func extractPeakCenteredWindows(_ signal: [Double], peaks: [Int]) -> [[Double]] {
        var windows: [[Double]] = []
        let halfWindow = config.segmentSamples / 2  // 50 samples
        
        for peak in peaks {
            let start = peak - halfWindow
            let end = peak + halfWindow
            
            // Check bounds
            guard start >= 0 && end <= signal.count else {
                continue
            }
            
            let window = Array(signal[start..<end])
            
            // Verify window size
            guard window.count == config.segmentSamples else {
                continue
            }
            
            // Check for NaN/inf in window
            if window.allSatisfy({ $0.isFinite }) {
                windows.append(window)
            }
        }
        
        return windows
    }
    
    // MARK: - Step 5: Template Matching
    
    private func templateMatching(_ windows: [[Double]]) -> [[Double]] {
        guard windows.count > 1 else {
            return windows
        }
        
        // Compute template (mean of all windows)
        let template = computeTemplate(windows)
        
        // Filter windows by cosine similarity
        var filteredWindows: [[Double]] = []
        
        for window in windows {
            let similarity = cosineSimilarity(window, template)
            if similarity >= config.similarityThreshold {
                filteredWindows.append(window)
            }
        }
        
        // If too many filtered out, return original windows
        if filteredWindows.isEmpty {
            return windows
        }
        
        return filteredWindows
    }
    
    private func computeTemplate(_ windows: [[Double]]) -> [Double] {
        let numSamples = config.segmentSamples
        var template = [Double](repeating: 0.0, count: numSamples)
        
        for window in windows {
            for i in 0..<numSamples {
                template[i] += window[i]
            }
        }
        
        for i in 0..<numSamples {
            template[i] /= Double(windows.count)
        }
        
        return template
    }
    
    private func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        var dotProduct = 0.0
        var magnitudeA = 0.0
        var magnitudeB = 0.0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }
        
        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)
        
        guard magnitudeA > 0 && magnitudeB > 0 else {
            return 0.0
        }
        
        return dotProduct / (magnitudeA * magnitudeB)
    }
    
    // MARK: - Step 6: Normalization
    
    private func normalizeSegment(_ segment: [Double]) -> [Double] {
        let meanValue = mean(segment)
        let stdValue = standardDeviation(segment)
        
        guard stdValue > 0 else {
            return segment
        }
        
        return segment.map { ($0 - meanValue) / stdValue }
    }
    
    // MARK: - Quality Score Computation
    
    private func computeQualityScore(rawSignal: [Double], segments: [[Double]]) -> Double {
        var score = 0.0
        
        // Factor 1: Number of segments (more is better)
        let segmentScore = min(Double(segments.count) / 50.0, 1.0)
        score += segmentScore * 0.3
        
        // Factor 2: Signal-to-noise ratio
        let signalStd = standardDeviation(rawSignal)
        let diffs = zip(rawSignal, rawSignal.dropFirst()).map { $1 - $0 }
        let noiseEstimate = standardDeviation(diffs)
        let snr = signalStd / (noiseEstimate + 1e-6)
        let snrScore = min(snr / 100.0, 1.0)
        score += snrScore * 0.3
        
        // Factor 3: Segment consistency
        let segmentStds = segments.map { standardDeviation($0) }
        let meanStd = mean(segmentStds)
        let stdStd = standardDeviation(segmentStds)
        let consistency = 1.0 - (stdStd / (meanStd + 1e-6))
        let consistencyScore = max(0.0, min(consistency, 1.0))
        score += consistencyScore * 0.2
        
        // Factor 4: Template matching success rate
        let templateScore = min(Double(segments.count) / 60.0, 1.0)
        score += templateScore * 0.2
        
        return min(score, 1.0)
    }
    
    // MARK: - Utility Functions
    
    private func mean(_ array: [Double]) -> Double {
        guard !array.isEmpty else { return 0.0 }
        return array.reduce(0.0, +) / Double(array.count)
    }
    
    private func standardDeviation(_ array: [Double]) -> Double {
        guard array.count > 1 else { return 0.0 }
        
        let meanValue = mean(array)
        let variance = array.reduce(0.0) { $0 + pow($1 - meanValue, 2) } / Double(array.count - 1)
        return sqrt(variance)
    }
}
