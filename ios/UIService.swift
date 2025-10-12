//
//  UIService.swift
//  PPG Glucose Monitor
//
//  Main UI coordinator and view models for the app
//

import SwiftUI
import Combine

// MARK: - App State
enum AppState {
    case disconnected
    case connecting
    case ready
    case placeFinger
    case collecting(progress: Float)
    case processing
    case result(PredictionResult)
    case error(String)
}

// MARK: - Main View Model
class GlucoseMonitorViewModel: ObservableObject {
    
    // MARK: - Published Properties
    @Published var appState: AppState = .disconnected
    @Published var isConnected: Bool = false
    @Published var collectionProgress: Float = 0.0
    @Published var statusMessage: String = "Tap to connect"
    @Published var qualityScore: Double = 0.0
    @Published var latestResult: PredictionResult?
    @Published var history: [PredictionResult] = []
    
    // MARK: - Services
    private let bleService: BLEService
    private let preprocessingService: PreprocessingService
    private let modelService: ModelService
    
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    init() {
        bleService = BLEService()
        preprocessingService = PreprocessingService()
        modelService = ModelService()
        
        bleService.delegate = self
        
        loadHistory()
    }
    
    // MARK: - Public Methods
    
    func connect() {
        statusMessage = "Scanning for ESP32..."
        appState = .connecting
        bleService.startScanning()
    }
    
    func disconnect() {
        bleService.disconnect()
        appState = .disconnected
        isConnected = false
        statusMessage = "Disconnected"
    }
    
    func startMeasurement() {
        guard appState == .ready else { return }
        
        statusMessage = "Place finger on sensor..."
        appState = .placeFinger
        collectionProgress = 0.0
        bleService.startCollection()
    }
    
    func cancelMeasurement() {
        bleService.cancelCollection()
        appState = .ready
        statusMessage = "Ready to measure"
        collectionProgress = 0.0
    }
    
    // MARK: - Private Methods
    
    private func processData(_ rawPPG: [UInt16]) {
        appState = .processing
        statusMessage = "Processing signal..."
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Step 1: Preprocess
            let preprocessingResult = self.preprocessingService.preprocess(rawPPG: rawPPG)
            
            DispatchQueue.main.async {
                self.qualityScore = preprocessingResult.qualityScore
                
                guard preprocessingResult.success,
                      let segments = preprocessingResult.segmentsForModel else {
                    self.appState = .error(preprocessingResult.errorMessage ?? "Preprocessing failed")
                    self.statusMessage = "Error: \(preprocessingResult.errorMessage ?? "Unknown")"
                    return
                }
                
                // Step 2: Run model inference
                do {
                    let prediction = try self.modelService.predict(segments: segments)
                    
                    self.latestResult = prediction
                    self.history.append(prediction)
                    self.saveHistory()
                    
                    self.appState = .result(prediction)
                    self.statusMessage = "Glucose: \(prediction.formattedGlucose)"
                    
                } catch {
                    self.appState = .error(error.localizedDescription)
                    self.statusMessage = "Prediction error"
                }
            }
        }
    }
    
    private func loadHistory() {
        // TODO: Load from UserDefaults or CoreData
        history = []
    }
    
    private func saveHistory() {
        // TODO: Save to UserDefaults or CoreData
    }
}

// MARK: - BLE Service Delegate
extension GlucoseMonitorViewModel: BLEServiceDelegate {
    
    func bleService(_ service: BLEService, didUpdateState state: BLEState) {
        DispatchQueue.main.async {
            switch state {
            case .disconnected:
                self.isConnected = false
                self.appState = .disconnected
                self.statusMessage = "Disconnected"
                
            case .scanning:
                self.statusMessage = "Scanning..."
                
            case .connecting:
                self.statusMessage = "Connecting..."
                
            case .connected:
                self.statusMessage = "Connected"
                
            case .ready:
                self.isConnected = true
                self.appState = .ready
                self.statusMessage = "Ready to measure"
            }
        }
    }
    
    func bleService(_ service: BLEService, didReceiveStatus status: CollectionStatus, progress: Float?) {
        DispatchQueue.main.async {
            if let progress = progress {
                self.collectionProgress = progress / 100.0
            }
            
            switch status {
            case .idle:
                self.appState = .ready
                self.statusMessage = "Ready"
                
            case .placeFinger:
                self.appState = .placeFinger
                self.statusMessage = "Place finger on sensor"
                
            case .adjustPressure:
                self.statusMessage = "Adjust finger pressure"
                
            case .collecting:
                self.appState = .collecting(progress: self.collectionProgress)
                self.statusMessage = String(format: "Collecting: %.0f%%", self.collectionProgress * 100)
                
            case .transmitting:
                self.statusMessage = "Transmitting data..."
                
            case .complete:
                self.statusMessage = "Processing..."
                
            case .errorQuality:
                self.appState = .error("Poor signal quality")
                self.statusMessage = "Error: Poor quality"
                
            case .ready:
                self.appState = .ready
                self.statusMessage = "Ready"
            }
        }
    }
    
    func bleService(_ service: BLEService, didReceiveData data: [UInt16]) {
        DispatchQueue.main.async {
            self.processData(data)
        }
    }
    
    func bleService(_ service: BLEService, didEncounterError error: Error) {
        DispatchQueue.main.async {
            self.appState = .error(error.localizedDescription)
            self.statusMessage = "Error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Main Content View
struct ContentView: View {
    @StateObject private var viewModel = GlucoseMonitorViewModel()
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background gradient
                LinearGradient(
                    gradient: Gradient(colors: [Color.blue.opacity(0.3), Color.purple.opacity(0.3)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 20) {
                    // Header
                    headerView
                    
                    Spacer()
                    
                    // Main content based on state
                    mainContentView
                    
                    Spacer()
                    
                    // Control buttons
                    controlButtonsView
                    
                    // Status bar
                    statusBarView
                }
                .padding()
            }
            .navigationTitle("PPG Glucose Monitor")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    // MARK: - Subviews
    
    private var headerView: some View {
        HStack {
            Circle()
                .fill(viewModel.isConnected ? Color.green : Color.gray)
                .frame(width: 12, height: 12)
            
            Text(viewModel.isConnected ? "Connected" : "Disconnected")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            if viewModel.qualityScore > 0 {
                Text(String(format: "Quality: %.0f%%", viewModel.qualityScore * 100))
                    .font(.caption)
                    .foregroundColor(qualityColor(viewModel.qualityScore))
            }
        }
        .padding()
        .background(Color.white.opacity(0.8))
        .cornerRadius(10)
    }
    
    @ViewBuilder
    private var mainContentView: some View {
        switch viewModel.appState {
        case .disconnected, .connecting:
            connectionView
            
        case .ready:
            readyView
            
        case .placeFinger:
            placeFingerView
            
        case .collecting(let progress):
            collectingView(progress: progress)
            
        case .processing:
            processingView
            
        case .result(let result):
            resultView(result: result)
            
        case .error(let message):
            errorView(message: message)
        }
    }
    
    private var connectionView: some View {
        VStack(spacing: 20) {
            Image(systemName: "antenna.radiowaves.left.and.right")
                .font(.system(size: 80))
                .foregroundColor(.blue)
            
            Text("Connect to ESP32")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Make sure your device is powered on")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private var readyView: some View {
        VStack(spacing: 20) {
            Image(systemName: "hand.point.up.left.fill")
                .font(.system(size: 80))
                .foregroundColor(.green)
            
            Text("Ready to Measure")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Tap 'Start' to begin glucose measurement")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private var placeFingerView: some View {
        VStack(spacing: 20) {
            Image(systemName: "hand.raised.fill")
                .font(.system(size: 80))
                .foregroundColor(.orange)
                .symbolEffect(.pulse)
            
            Text("Place Finger on Sensor")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Ensure good contact and stay still")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private func collectingView(progress: Float) -> some View {
        VStack(spacing: 20) {
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 20)
                    .frame(width: 200, height: 200)
                
                Circle()
                    .trim(from: 0, to: CGFloat(progress))
                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 20, lineCap: .round))
                    .frame(width: 200, height: 200)
                    .rotationEffect(.degrees(-90))
                    .animation(.linear, value: progress)
                
                VStack {
                    Text("\(Int(progress * 100))%")
                        .font(.system(size: 40, weight: .bold))
                    Text("\(Int(progress * 60))s / 60s")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Text("Collecting PPG Data")
                .font(.title3)
                .fontWeight(.semibold)
            
            Text("Keep finger steady on sensor")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private var processingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(2)
            
            Text("Processing Signal")
                .font(.title2)
                .fontWeight(.bold)
                .padding(.top)
            
            Text("Analyzing PPG data...")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private func resultView(result: PredictionResult) -> some View {
        VStack(spacing: 30) {
            // Glucose value
            VStack(spacing: 10) {
                Text(String(format: "%.0f", result.glucoseMgDL))
                    .font(.system(size: 80, weight: .bold))
                    .foregroundColor(glucoseLevelColor(result.glucoseLevel))
                
                Text("mg/dL")
                    .font(.title3)
                    .foregroundColor(.secondary)
                
                Text(result.glucoseLevel.description)
                    .font(.headline)
                    .foregroundColor(glucoseLevelColor(result.glucoseLevel))
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)
                    .background(glucoseLevelColor(result.glucoseLevel).opacity(0.2))
                    .cornerRadius(20)
            }
            
            // Metadata
            VStack(spacing: 8) {
                HStack {
                    Text("Confidence:")
                    Spacer()
                    Text(String(format: "%.0f%%", result.confidence * 100))
                        .fontWeight(.semibold)
                }
                
                HStack {
                    Text("Segments:")
                    Spacer()
                    Text("\(result.numSegments)")
                        .fontWeight(.semibold)
                }
                
                HStack {
                    Text("Time:")
                    Spacer()
                    Text(result.timestamp, style: .time)
                        .fontWeight(.semibold)
                }
            }
            .font(.caption)
            .padding()
            .background(Color.white.opacity(0.8))
            .cornerRadius(10)
        }
        .padding()
    }
    
    private func errorView(message: String) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 80))
                .foregroundColor(.red)
            
            Text("Error")
                .font(.title2)
                .fontWeight(.bold)
            
            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding()
        }
    }
    
    private var controlButtonsView: some View {
        HStack(spacing: 20) {
            if !viewModel.isConnected {
                Button(action: { viewModel.connect() }) {
                    Label("Connect", systemImage: "antenna.radiowaves.left.and.right")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            } else {
                if case .collecting = viewModel.appState {
                    Button(action: { viewModel.cancelMeasurement() }) {
                        Label("Cancel", systemImage: "xmark.circle")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                } else if case .ready = viewModel.appState {
                    Button(action: { viewModel.startMeasurement() }) {
                        Label("Start Measurement", systemImage: "play.circle.fill")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                } else if case .result = viewModel.appState {
                    Button(action: { viewModel.appState = .ready }) {
                        Label("New Measurement", systemImage: "arrow.clockwise")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
                
                Button(action: { viewModel.disconnect() }) {
                    Image(systemName: "power")
                        .padding()
                        .background(Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
        }
    }
    
    private var statusBarView: some View {
        Text(viewModel.statusMessage)
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.vertical, 8)
    }
    
    // MARK: - Helper Functions
    
    private func qualityColor(_ score: Double) -> Color {
        if score >= 0.7 {
            return .green
        } else if score >= 0.5 {
            return .orange
        } else {
            return .red
        }
    }
    
    private func glucoseLevelColor(_ level: GlucoseLevel) -> Color {
        switch level {
        case .low: return .blue
        case .normal: return .green
        case .elevated: return .orange
        case .high: return .red
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
