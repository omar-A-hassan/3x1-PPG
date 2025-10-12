//
//  ModelService.swift
//  PPG Glucose Monitor
//
//  Handles CoreML model inference for glucose prediction
//

import Foundation
import CoreML

// MARK: - Prediction Result
struct PredictionResult {
    let glucoseMgDL: Double
    let confidence: Double
    let numSegments: Int
    let individualPredictions: [Double]
    let timestamp: Date
    
    var formattedGlucose: String {
        return String(format: "%.1f mg/dL", glucoseMgDL)
    }
    
    var glucoseLevel: GlucoseLevel {
        switch glucoseMgDL {
        case ..<70:
            return .low
        case 70..<140:
            return .normal
        case 140..<180:
            return .elevated
        default:
            return .high
        }
    }
}

// MARK: - Glucose Level Categories
enum GlucoseLevel {
    case low        // < 70 mg/dL
    case normal     // 70-140 mg/dL
    case elevated   // 140-180 mg/dL
    case high       // > 180 mg/dL
    
    var description: String {
        switch self {
        case .low: return "Low"
        case .normal: return "Normal"
        case .elevated: return "Elevated"
        case .high: return "High"
        }
    }
    
    var color: String {
        switch self {
        case .low: return "blue"
        case .normal: return "green"
        case .elevated: return "yellow"
        case .high: return "red"
        }
    }
}

// MARK: - Model Service Error
enum ModelServiceError: Error {
    case modelNotLoaded
    case invalidInput
    case predictionFailed(String)
    
    var localizedDescription: String {
        switch self {
        case .modelNotLoaded:
            return "CoreML model not loaded"
        case .invalidInput:
            return "Invalid input data for model"
        case .predictionFailed(let message):
            return "Prediction failed: \(message)"
        }
    }
}

// MARK: - Model Service
class ModelService {
    
    // MARK: - Properties
    private var model: MLModel?
    private let modelName = "TSEncoderPPG"  // Your CoreML model name
    
    // MARK: - Initialization
    
    init() {
        loadModel()
    }
    
    // MARK: - Public Methods
    
    /// Load CoreML model
    private func loadModel() {
        do {
            // TODO: Replace with actual CoreML model loading
            // let config = MLModelConfiguration()
            // model = try TSEncoderPPG(configuration: config).model
            
            print("Model: Loading \(modelName).mlmodel...")
            
            // For now, model is nil (stub)
            // Uncomment above when CoreML model is available
            
            print("Model: ⚠️ Running in STUB mode - no actual model loaded")
            
        } catch {
            print("Model: Failed to load - \(error.localizedDescription)")
        }
    }
    
    /// Predict glucose from preprocessed segments
    func predict(segments: [[[Double]]]) throws -> PredictionResult {
        
        // Validate input
        guard !segments.isEmpty else {
            throw ModelServiceError.invalidInput
        }
        
        // STUB MODE: If no model loaded, return mock prediction
        if model == nil {
            return mockPrediction(numSegments: segments.count)
        }
        
        // REAL MODE: Run CoreML inference
        var predictions: [Double] = []
        
        for segment in segments {
            // Convert segment to MLMultiArray
            guard let input = try? createMLMultiArray(from: segment) else {
                throw ModelServiceError.invalidInput
            }
            
            // Run prediction
            // TODO: Replace with actual CoreML prediction
            // let output = try model.prediction(from: input)
            // let glucose = output.featureValue(for: "glucose")?.doubleValue ?? 0.0
            // predictions.append(glucose)
            
            // Stub: Random prediction
            predictions.append(Double.random(in: 90...140))
        }
        
        // Average predictions across all segments
        let finalGlucose = predictions.reduce(0.0, +) / Double(predictions.count)
        
        // Compute confidence (variance-based)
        let variance = predictions.map { pow($0 - finalGlucose, 2) }.reduce(0.0, +) / Double(predictions.count)
        let confidence = max(0.0, 1.0 - sqrt(variance) / 50.0)  // Lower variance = higher confidence
        
        return PredictionResult(
            glucoseMgDL: finalGlucose,
            confidence: confidence,
            numSegments: segments.count,
            individualPredictions: predictions,
            timestamp: Date()
        )
    }
    
    // MARK: - Helper Methods
    
    private func createMLMultiArray(from segment: [[Double]]) throws -> MLMultiArray {
        // Create MLMultiArray with shape [100, 1] for single segment
        let array = try MLMultiArray(shape: [100, 1], dataType: .double)
        
        for i in 0..<segment.count {
            for j in 0..<segment[i].count {
                let index = [i, j] as [NSNumber]
                array[index] = NSNumber(value: segment[i][j])
            }
        }
        
        return array
    }
    
    // MARK: - Mock Prediction (for testing without model)
    
    private func mockPrediction(numSegments: Int) -> PredictionResult {
        print("Model: Running MOCK prediction (no real model)")
        
        // Generate mock predictions with some variance
        let baseLine = Double.random(in: 90...130)
        let predictions = (0..<numSegments).map { _ in
            baseLine + Double.random(in: -10...10)
        }
        
        let finalGlucose = predictions.reduce(0.0, +) / Double(predictions.count)
        let variance = predictions.map { pow($0 - finalGlucose, 2) }.reduce(0.0, +) / Double(predictions.count)
        let confidence = max(0.0, 1.0 - sqrt(variance) / 50.0)
        
        return PredictionResult(
            glucoseMgDL: finalGlucose,
            confidence: confidence,
            numSegments: numSegments,
            individualPredictions: predictions,
            timestamp: Date()
        )
    }
    
    // MARK: - Model Info
    
    func getModelInfo() -> String {
        if model == nil {
            return """
            ⚠️ STUB MODE - No CoreML Model Loaded
            
            To use real predictions:
            1. Convert PyTorch model to CoreML (.mlmodel)
            2. Add \(modelName).mlmodel to Xcode project
            3. Uncomment model loading code in ModelService.swift
            
            Current: Using mock predictions for testing
            """
        } else {
            return """
            ✓ CoreML Model Loaded
            Model: \(modelName)
            Input: (N, 100, 1) segments
            Output: Glucose (mg/dL)
            """
        }
    }
}

// MARK: - Batch Prediction Support

extension ModelService {
    
    /// Predict with quality filtering
    func predictWithQualityFilter(
        segments: [[[Double]]],
        qualityScore: Double,
        minQualityThreshold: Double = 0.5
    ) throws -> PredictionResult {
        
        guard qualityScore >= minQualityThreshold else {
            throw ModelServiceError.predictionFailed(
                "Signal quality too low (\(String(format: "%.2f", qualityScore)) < \(minQualityThreshold))"
            )
        }
        
        return try predict(segments: segments)
    }
    
    /// Get prediction statistics
    func getPredictionStatistics(_ result: PredictionResult) -> String {
        let predictions = result.individualPredictions
        let mean = predictions.reduce(0.0, +) / Double(predictions.count)
        let variance = predictions.map { pow($0 - mean, 2) }.reduce(0.0, +) / Double(predictions.count)
        let std = sqrt(variance)
        let min = predictions.min() ?? 0.0
        let max = predictions.max() ?? 0.0
        
        return """
        Prediction Statistics:
        ----------------------
        Final Glucose: \(result.formattedGlucose)
        Level: \(result.glucoseLevel.description)
        Confidence: \(String(format: "%.1f%%", result.confidence * 100))
        
        Segment Analysis:
        Total Segments: \(result.numSegments)
        Mean: \(String(format: "%.1f", mean)) mg/dL
        Std Dev: \(String(format: "%.1f", std)) mg/dL
        Range: \(String(format: "%.1f", min)) - \(String(format: "%.1f", max)) mg/dL
        
        Timestamp: \(result.timestamp)
        """
    }
}

// MARK: - CoreML Model Conversion Instructions

/*
 
 CONVERTING PYTORCH MODEL TO COREML
 ==================================
 
 1. Install coremltools:
    pip install coremltools
 
 2. Create conversion script (convert_to_coreml.py):
 
    import torch
    import coremltools as ct
    from src.models import build_tsencoder_ppg
    
    # Load trained model
    model = build_tsencoder_ppg()
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 100, 1)  # Single segment
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="x", shape=(1, 100, 1))],
        outputs=[ct.TensorType(name="glucose")]
    )
    
    # Save
    coreml_model.save("TSEncoderPPG.mlmodel")
 
 3. Add TSEncoderPPG.mlmodel to Xcode project
 
 4. Xcode will auto-generate Swift interface
 
 5. Update ModelService to use generated class:
    let model = try TSEncoderPPG(configuration: MLModelConfiguration())
 
 */
