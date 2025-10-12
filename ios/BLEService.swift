//
//  BLEService.swift
//  PPG Glucose Monitor
//
//  Handles Bluetooth Low Energy communication with ESP32
//

import Foundation
import CoreBluetooth

// MARK: - BLE Service UUIDs
struct BLEServiceUUIDs {
    static let service = CBUUID(string: "4fafc201-1fb5-459e-8fcc-c5c9c331914b")
    static let dataCharacteristic = CBUUID(string: "beb5483e-36e1-4688-b7f5-ea07361b26a8")
    static let statusCharacteristic = CBUUID(string: "beb5483e-36e1-4688-b7f5-ea07361b26a9")
    static let controlCharacteristic = CBUUID(string: "beb5483e-36e1-4688-b7f5-ea07361b26aa")
}

// MARK: - BLE State
enum BLEState {
    case disconnected
    case scanning
    case connecting
    case connected
    case ready
}

// MARK: - Collection Status
enum CollectionStatus: String {
    case idle = "IDLE"
    case placeFinger = "PLACE_FINGER"
    case adjustPressure = "ADJUST_PRESSURE"
    case collecting = "COLLECTING"
    case transmitting = "TRANSMITTING"
    case complete = "COMPLETE"
    case errorQuality = "ERROR_QUALITY"
    case ready = "READY"
    
    init(from string: String) {
        if string.hasPrefix("PROGRESS:") {
            self = .collecting
        } else if string.hasPrefix("TRANSMIT:") {
            self = .transmitting
        } else {
            self = CollectionStatus(rawValue: string) ?? .idle
        }
    }
}

// MARK: - BLE Service Delegate
protocol BLEServiceDelegate: AnyObject {
    func bleService(_ service: BLEService, didUpdateState state: BLEState)
    func bleService(_ service: BLEService, didReceiveStatus status: CollectionStatus, progress: Float?)
    func bleService(_ service: BLEService, didReceiveData data: [UInt16])
    func bleService(_ service: BLEService, didEncounterError error: Error)
}

// MARK: - BLE Service
class BLEService: NSObject {
    
    // MARK: - Properties
    weak var delegate: BLEServiceDelegate?
    
    private var centralManager: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var dataCharacteristic: CBCharacteristic?
    private var statusCharacteristic: CBCharacteristic?
    private var controlCharacteristic: CBCharacteristic?
    
    private(set) var state: BLEState = .disconnected {
        didSet {
            delegate?.bleService(self, didUpdateState: state)
        }
    }
    
    // Data collection
    private var ppgDataBuffer: [UInt16] = []
    private let expectedSamples = 6000  // 60 seconds @ 100 Hz
    
    // MARK: - Initialization
    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    // MARK: - Public Methods
    
    /// Start scanning for ESP32 device
    func startScanning() {
        guard centralManager.state == .poweredOn else {
            print("BLE: Bluetooth not powered on")
            return
        }
        
        state = .scanning
        ppgDataBuffer.removeAll()
        
        print("BLE: Starting scan for ESP32-PPG-Glucose...")
        centralManager.scanForPeripherals(
            withServices: [BLEServiceUUIDs.service],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
    }
    
    /// Stop scanning
    func stopScanning() {
        centralManager.stopScan()
        if state == .scanning {
            state = .disconnected
        }
    }
    
    /// Disconnect from peripheral
    func disconnect() {
        guard let peripheral = peripheral else { return }
        centralManager.cancelPeripheralConnection(peripheral)
    }
    
    /// Send start collection command to ESP32
    func startCollection() {
        guard state == .ready,
              let characteristic = controlCharacteristic else {
            print("BLE: Not ready to start collection")
            return
        }
        
        ppgDataBuffer.removeAll()
        let command = Data([0x53])  // 'S' = Start
        peripheral?.writeValue(command, for: characteristic, type: .withResponse)
        print("BLE: Sent START command")
    }
    
    /// Send cancel command to ESP32
    func cancelCollection() {
        guard let characteristic = controlCharacteristic else { return }
        
        let command = Data([0x43])  // 'C' = Cancel
        peripheral?.writeValue(command, for: characteristic, type: .withResponse)
        print("BLE: Sent CANCEL command")
    }
    
    /// Send reset command to ESP32
    func resetDevice() {
        guard let characteristic = controlCharacteristic else { return }
        
        let command = Data([0x52])  // 'R' = Reset
        peripheral?.writeValue(command, for: characteristic, type: .withResponse)
        print("BLE: Sent RESET command")
    }
}

// MARK: - CBCentralManagerDelegate
extension BLEService: CBCentralManagerDelegate {
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("BLE: Powered ON")
            state = .disconnected
        case .poweredOff:
            print("BLE: Powered OFF")
            state = .disconnected
        case .resetting:
            print("BLE: Resetting")
        case .unauthorized:
            print("BLE: Unauthorized")
        case .unsupported:
            print("BLE: Unsupported")
        case .unknown:
            print("BLE: Unknown")
        @unknown default:
            print("BLE: Unknown state")
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                       advertisementData: [String: Any], rssi RSSI: NSNumber) {
        
        let name = peripheral.name ?? "Unknown"
        print("BLE: Discovered device: \(name), RSSI: \(RSSI)")
        
        // Look for our ESP32 device
        if name.contains("ESP32-PPG") {
            print("BLE: Found ESP32-PPG-Glucose! Connecting...")
            self.peripheral = peripheral
            central.stopScan()
            state = .connecting
            central.connect(peripheral, options: nil)
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("BLE: Connected to \(peripheral.name ?? "device")")
        state = .connected
        
        peripheral.delegate = self
        peripheral.discoverServices([BLEServiceUUIDs.service])
    }
    
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("BLE: Disconnected from \(peripheral.name ?? "device")")
        state = .disconnected
        
        self.peripheral = nil
        self.dataCharacteristic = nil
        self.statusCharacteristic = nil
        self.controlCharacteristic = nil
        
        if let error = error {
            delegate?.bleService(self, didEncounterError: error)
        }
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("BLE: Failed to connect")
        state = .disconnected
        
        if let error = error {
            delegate?.bleService(self, didEncounterError: error)
        }
    }
}

// MARK: - CBPeripheralDelegate
extension BLEService: CBPeripheralDelegate {
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        
        for service in services {
            if service.uuid == BLEServiceUUIDs.service {
                print("BLE: Found PPG service")
                peripheral.discoverCharacteristics([
                    BLEServiceUUIDs.dataCharacteristic,
                    BLEServiceUUIDs.statusCharacteristic,
                    BLEServiceUUIDs.controlCharacteristic
                ], for: service)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        
        for characteristic in characteristics {
            switch characteristic.uuid {
            case BLEServiceUUIDs.dataCharacteristic:
                print("BLE: Found data characteristic")
                dataCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
                
            case BLEServiceUUIDs.statusCharacteristic:
                print("BLE: Found status characteristic")
                statusCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
                
            case BLEServiceUUIDs.controlCharacteristic:
                print("BLE: Found control characteristic")
                controlCharacteristic = characteristic
                
            default:
                break
            }
        }
        
        // Check if all characteristics are ready
        if dataCharacteristic != nil && statusCharacteristic != nil && controlCharacteristic != nil {
            state = .ready
            print("BLE: All characteristics ready!")
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error = error {
            print("BLE: Error updating value: \(error)")
            return
        }
        
        guard let data = characteristic.value else { return }
        
        switch characteristic.uuid {
        case BLEServiceUUIDs.dataCharacteristic:
            handleDataReceived(data)
            
        case BLEServiceUUIDs.statusCharacteristic:
            handleStatusReceived(data)
            
        default:
            break
        }
    }
    
    // MARK: - Data Handling
    
    private func handleDataReceived(_ data: Data) {
        // Convert Data to array of UInt16 (little-endian)
        let sampleCount = data.count / 2
        var samples: [UInt16] = []
        
        for i in 0..<sampleCount {
            let offset = i * 2
            let sample = data.withUnsafeBytes { bytes -> UInt16 in
                let pointer = bytes.baseAddress!.advanced(by: offset)
                return pointer.assumingMemoryBound(to: UInt16.self).pointee
            }
            samples.append(sample)
        }
        
        ppgDataBuffer.append(contentsOf: samples)
        
        print("BLE: Received \(samples.count) samples, total: \(ppgDataBuffer.count)/\(expectedSamples)")
        
        // Check if collection complete
        if ppgDataBuffer.count >= expectedSamples {
            print("BLE: Data collection complete! \(ppgDataBuffer.count) samples")
            delegate?.bleService(self, didReceiveData: ppgDataBuffer)
        }
    }
    
    private func handleStatusReceived(_ data: Data) {
        guard let statusString = String(data: data, encoding: .utf8) else { return }
        
        print("BLE: Status: \(statusString)")
        
        // Parse progress if present
        var progress: Float?
        if statusString.hasPrefix("PROGRESS:") {
            let progressStr = statusString.replacingOccurrences(of: "PROGRESS:", with: "")
            progress = Float(progressStr)
        } else if statusString.hasPrefix("TRANSMIT:") {
            let progressStr = statusString.replacingOccurrences(of: "TRANSMIT:", with: "")
            progress = Float(progressStr)
        }
        
        let status = CollectionStatus(from: statusString)
        delegate?.bleService(self, didReceiveStatus: status, progress: progress)
    }
}
