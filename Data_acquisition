import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import pandas as pd
import numpy as np
import time
from datetime import datetime
import threading
import queue

# Configuration
PORT = 'COM6'  # Change to your port
BAUD_RATE = 921600  # Must match Arduino (increased from 115200)
MAX_POINTS = 3000  # Display last 10 seconds at 300Hz
SAVE_INTERVAL = 1000  # Save to disk every 1000 samples

# Initialize serial with large buffer
ser = serial.Serial(PORT, BAUD_RATE, timeout=0.01)
ser.reset_input_buffer()

# Thread-safe queue for data
data_queue = queue.Queue()

# Data storage for plotting (limited size)
timestamps = deque(maxlen=MAX_POINTS)
red_values = deque(maxlen=MAX_POINTS)
ir_values = deque(maxlen=MAX_POINTS)
sample_numbers = deque(maxlen=MAX_POINTS)

# Complete data storage (unlimited - all samples saved here)
all_data = []
save_buffer = []

# Statistics
total_samples = 0
missed_samples = 0
last_sample_num = None
start_time = time.time()
last_save_time = time.time()

# Skip header and any initial garbage
for _ in range(5):
    ser.readline()

print("Starting data acquisition at ~300Hz...")
print("Close the plot window to stop and save all data.\n")

# Separate thread for reading serial data (prevents blocking)
def read_serial():
    global ser
    while ser.is_open:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and not line.startswith('#'):  # Skip comment lines
                    data_queue.put(line)
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(0.001)

# Start serial reading thread
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
line1, = ax1.plot([], [], 'r-', linewidth=0.5, alpha=0.8)
line2, = ax2.plot([], [], 'b-', linewidth=0.5, alpha=0.8)

ax1.set_ylabel('Red Light (ADC)', fontsize=10)
ax1.set_title('MAX30102 Real-time Data @ ~300Hz', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (seconds)', fontsize=10)
ax2.set_ylabel('IR Light (ADC)', fontsize=10)
ax2.grid(True, alpha=0.3)

# Statistics text boxes
stats_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                      verticalalignment='top', fontfamily='monospace', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

quality_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2, stats_text, quality_text

def update(frame):
    global total_samples, missed_samples, last_sample_num, save_buffer, last_save_time
    
    # Process all available data in queue (up to 100 samples per frame)
    samples_processed = 0
    while not data_queue.empty() and samples_processed < 100:
        try:
            line = data_queue.get_nowait()
            parts = line.split(',')
            
            if len(parts) == 4:
                sample_num = int(parts[0])
                timestamp = float(parts[1])
                red = int(parts[2])
                ir = int(parts[3])
                
                # Check for missing samples
                if last_sample_num is not None:
                    expected = last_sample_num + 1
                    if sample_num != expected:
                        gap = sample_num - expected
                        missed_samples += gap
                        print(f"⚠ Warning: {gap} sample(s) missed! (jumped from {last_sample_num} to {sample_num})")
                
                last_sample_num = sample_num
                
                # Add to display buffers
                timestamps.append(timestamp)
                red_values.append(red)
                ir_values.append(ir)
                sample_numbers.append(sample_num)
                
                # Add to save buffer
                save_buffer.append([sample_num, timestamp, red, ir])
                
                total_samples += 1
                samples_processed += 1
                
        except Exception as e:
            print(f"Parse error: {e}, line: {line}")
    
    # Periodic save to prevent memory overflow
    current_time = time.time()
    if len(save_buffer) >= SAVE_INTERVAL or (current_time - last_save_time > 10):
        all_data.extend(save_buffer)
        save_buffer.clear()
        last_save_time = current_time
    
    # Update plots
    if len(timestamps) > 1:
        # Auto-scale Y-axis with some padding
        if len(red_values) > 10:
            red_min, red_max = min(red_values), max(red_values)
            ir_min, ir_max = min(ir_values), max(ir_values)
            
            red_padding = (red_max - red_min) * 0.1 or 1000
            ir_padding = (ir_max - ir_min) * 0.1 or 1000
            
            ax1.set_ylim(red_min - red_padding, red_max + red_padding)
            ax2.set_ylim(ir_min - ir_padding, ir_max + ir_padding)
        
        # Show sliding window (last 10 seconds)
        current_time = timestamps[-1]
        window_start = max(0, current_time - 10)
        
        ax1.set_xlim(window_start, current_time + 0.5)
        ax2.set_xlim(window_start, current_time + 0.5)
        
        # Update line data
        line1.set_data(list(timestamps), list(red_values))
        line2.set_data(list(timestamps), list(ir_values))
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        actual_rate = total_samples / elapsed_time if elapsed_time > 0 else 0
        loss_rate = (missed_samples / max(total_samples, 1)) * 100
        buffer_size = ser.in_waiting
        queue_size = data_queue.qsize()
        
        # Update statistics display
        stats_str = f'Samples Received: {total_samples:,}\n'
        stats_str += f'Actual Rate: {actual_rate:.1f} Hz\n'
        stats_str += f'Missed Samples: {missed_samples}\n'
        stats_str += f'Loss Rate: {loss_rate:.3f}%\n'
        stats_str += f'Elapsed: {elapsed_time:.1f}s'
        stats_text.set_text(stats_str)
        
        # Buffer health indicator
        buffer_health = "🟢 GOOD" if buffer_size < 512 else "🟡 WARN" if buffer_size < 2048 else "🔴 OVERFLOW RISK"
        quality_str = f'Serial Buffer: {buffer_size} bytes\n'
        quality_str += f'Queue Size: {queue_size}\n'
        quality_str += f'Status: {buffer_health}\n'
        quality_str += f'Stored: {len(all_data) + len(save_buffer):,} samples'
        quality_text.set_text(quality_str)
    
    return line1, line2, stats_text, quality_text

def save_all_data():
    """Save all collected data to CSV"""
    print("\nSaving data...")
    
    # Combine any remaining buffer data
    if save_buffer:
        all_data.extend(save_buffer)
    
    if len(all_data) > 0:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'max30102_data_{timestamp}.csv'
        
        df = pd.DataFrame(all_data, columns=['SampleNum', 'Timestamp', 'Red', 'IR'])
        df.to_csv(filename, index=False)
        
        # Generate analysis report
        print(f"\n{'='*60}")
        print(f"DATA ACQUISITION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Samples Collected: {len(df):,}")
        print(f"Missed Samples: {missed_samples}")
        print(f"Loss Rate: {(missed_samples / max(len(df), 1)) * 100:.4f}%")
        print(f"Duration: {df['Timestamp'].max():.2f} seconds")
        print(f"Average Sample Rate: {len(df) / df['Timestamp'].max():.2f} Hz")
        print(f"Data saved to: {filename}")
        print(f"{'='*60}\n")
        
        # Save summary statistics
        stats_filename = f'acquisition_stats_{timestamp}.txt'
        with open(stats_filename, 'w') as f:
            f.write(f"MAX30102 Data Acquisition Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Samples: {len(df):,}\n")
            f.write(f"Missed Samples: {missed_samples}\n")
            f.write(f"Loss Rate: {(missed_samples / max(len(df), 1)) * 100:.4f}%\n")
            f.write(f"Duration: {df['Timestamp'].max():.2f} seconds\n")
            f.write(f"Average Rate: {len(df) / df['Timestamp'].max():.2f} Hz\n")
            f.write(f"Red - Min: {df['Red'].min()}, Max: {df['Red'].max()}, Mean: {df['Red'].mean():.2f}\n")
            f.write(f"IR  - Min: {df['IR'].min()}, Max: {df['IR'].max()}, Mean: {df['IR'].mean():.2f}\n")
            
            # Check for sample gaps
            if len(df) > 1:
                df_sorted = df.sort_values('SampleNum')
                gaps = df_sorted['SampleNum'].diff()
                gap_locations = gaps[gaps > 1]
                if len(gap_locations) > 0:
                    f.write(f"\nGaps detected at sample numbers:\n")
                    for idx, gap in gap_locations.items():
                        f.write(f"  Sample {df_sorted.loc[idx, 'SampleNum']}: gap of {int(gap)-1} samples\n")
        
        print(f"Statistics saved to: {stats_filename}")
        return True
    else:
        print("No data to save!")
        return False

def on_close(event):
    """Handle window close event"""
    print("\nClosing acquisition...")
    ser.close()
    save_all_data()
    plt.close('all')

# Register cleanup
import atexit
atexit.register(lambda: save_all_data() if ser.is_open else None)

# Connect close event
fig.canvas.mpl_connect('close_event', on_close)

# Animate with fast update (50ms = 20fps for smooth visualization)
ani = animation.FuncAnimation(fig, update, init_func=init,
                             interval=50, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

# Cleanup
if ser.is_open:
    ser.close()
save_all_data()