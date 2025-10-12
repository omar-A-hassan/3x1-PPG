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
BAUD_RATE = 115200  # Must match Arduino
MAX_POINTS = 1000  # Display last 10 seconds at 100Hz
SAVE_INTERVAL = 500  # Save to disk every 500 samples

# Initialize serial with large buffer
ser = serial.Serial(PORT, BAUD_RATE, timeout=0.01)
ser.reset_input_buffer()

# Thread-safe queue for data
data_queue = queue.Queue()

# Data storage for plotting (limited size for display)
timestamps = deque(maxlen=MAX_POINTS)
raw_values = deque(maxlen=MAX_POINTS)
no_baseline_values = deque(maxlen=MAX_POINTS)
no_50hz_values = deque(maxlen=MAX_POINTS)
processed_values = deque(maxlen=MAX_POINTS)

# Complete data storage (unlimited - all samples saved here)
all_data = []
save_buffer = []

# Statistics
total_samples = 0
start_time = time.time()
last_save_time = time.time()

# Skip header and any initial garbage
print("Waiting for sensor to stabilize...")
time.sleep(2)
ser.reset_input_buffer()

print("Starting data acquisition at ~100Hz...")
print("Close the plot window to stop and save all data.\n")

# Separate thread for reading serial data (prevents blocking)
def read_serial():
    global ser
    while ser.is_open:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and not line.startswith('Starting') and not line.startswith('MAX30102'):
                    data_queue.put(line)
        except Exception as e:
            print(f"Serial read error: {e}")
            time.sleep(0.001)

# Start serial reading thread
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# Create figure with 4 subplots (one for each processing stage)
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

line1, = axes[0].plot([], [], 'b-', linewidth=0.8, alpha=0.8, label='Raw IR')
line2, = axes[1].plot([], [], 'g-', linewidth=0.8, alpha=0.8, label='No Baseline')
line3, = axes[2].plot([], [], 'm-', linewidth=0.8, alpha=0.8, label='After 50Hz Filter')
line4, = axes[3].plot([], [], 'r-', linewidth=0.8, alpha=0.8, label='Final Processed')

titles = ['Raw IR Signal', 'After Baseline Removal', 'After 50Hz Notch Filter', 'Final Processed Signal']
colors = ['lightblue', 'lightgreen', 'plum', 'lightcoral']

for i, (ax, title, color) in enumerate(zip(axes, titles, colors)):
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

axes[3].set_xlabel('Time (seconds)', fontsize=10)

# Statistics text box
stats_text = axes[0].text(0.02, 0.02, '', transform=axes[0].transAxes, 
                          verticalalignment='bottom', fontfamily='monospace', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line2, line3, line4, stats_text

def update(frame):
    global total_samples, save_buffer, last_save_time
    
    # Process all available data in queue (up to 50 samples per frame)
    samples_processed = 0
    while not data_queue.empty() and samples_processed < 50:
        try:
            line = data_queue.get_nowait()
            
            # Parse space-separated values: raw no_baseline no_50hz processed
            parts = line.split()
            
            if len(parts) == 4:
                raw = float(parts[0])
                no_baseline = float(parts[1])
                no_50hz = float(parts[2])
                processed = float(parts[3])
                
                # Calculate timestamp based on sample count and sample rate
                timestamp = total_samples / 100.0  # 100 Hz sampling
                
                # Add to display buffers
                timestamps.append(timestamp)
                raw_values.append(raw)
                no_baseline_values.append(no_baseline)
                no_50hz_values.append(no_50hz)
                processed_values.append(processed)
                
                # Add to save buffer
                save_buffer.append([timestamp, raw, no_baseline, no_50hz, processed])
                
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
    if len(timestamps) > 10:
        time_array = list(timestamps)
        
        # Show sliding window (last 10 seconds)
        current_time_val = timestamps[-1]
        window_start = max(0, current_time_val - 10)
        
        # Update each subplot
        line1.set_data(time_array, list(raw_values))
        line2.set_data(time_array, list(no_baseline_values))
        line3.set_data(time_array, list(no_50hz_values))
        line4.set_data(time_array, list(processed_values))
        
        # Auto-scale each axis
        for ax, values in zip(axes, [raw_values, no_baseline_values, no_50hz_values, processed_values]):
            if len(values) > 10:
                val_min, val_max = min(values), max(values)
                padding = (val_max - val_min) * 0.1 or 100
                ax.set_ylim(val_min - padding, val_max + padding)
            ax.set_xlim(window_start, current_time_val + 0.5)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        actual_rate = total_samples / elapsed_time if elapsed_time > 0 else 0
        buffer_size = ser.in_waiting
        queue_size = data_queue.qsize()
        
        # Update statistics display
        stats_str = f'Samples: {total_samples:,} | Rate: {actual_rate:.1f} Hz | '
        stats_str += f'Time: {elapsed_time:.1f}s | Buffer: {buffer_size}B | Queue: {queue_size}'
        stats_text.set_text(stats_str)
    
    return line1, line2, line3, line4, stats_text

def save_all_data():
    """Save all collected data to CSV"""
    print("\nSaving data...")
    
    # Combine any remaining buffer data
    if save_buffer:
        all_data.extend(save_buffer)
    
    if len(all_data) > 0:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'max30102_processed_{timestamp}.csv'
        
        df = pd.DataFrame(all_data, columns=['Timestamp', 'Raw', 'No_Baseline', 'After_50Hz', 'Processed'])
        df.to_csv(filename, index=False)
        
        # Generate analysis report
        print(f"\n{'='*60}")
        print(f"DATA ACQUISITION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Samples Collected: {len(df):,}")
        print(f"Duration: {df['Timestamp'].max():.2f} seconds")
        print(f"Average Sample Rate: {len(df) / df['Timestamp'].max():.2f} Hz")
        print(f"\nSignal Statistics:")
        print(f"  Raw IR - Mean: {df['Raw'].mean():.1f}, Std: {df['Raw'].std():.1f}")
        print(f"  Processed - Mean: {df['Processed'].mean():.1f}, Std: {df['Processed'].std():.1f}")
        print(f"  Noise Reduction: {(1 - df['Processed'].std() / df['Raw'].std()) * 100:.1f}%")
        print(f"\nData saved to: {filename}")
        print(f"{'='*60}\n")
        
        return filename
    return None

def on_close(event):
    """Handle window close event"""
    print("\nClosing acquisition...")
    ser.close()
    filename = save_all_data()
    
    # Generate summary plot
    if filename and len(all_data) > 0:
        print("Generating summary plot...")
        df = pd.DataFrame(all_data, columns=['Timestamp', 'Raw', 'No_Baseline', 'After_50Hz', 'Processed'])
        
        fig_summary, axes_summary = plt.subplots(4, 1, figsize=(12, 10))
        
        axes_summary[0].plot(df['Timestamp'], df['Raw'], 'b-', linewidth=0.5, alpha=0.7)
        axes_summary[0].set_title('Raw Signal')
        axes_summary[0].set_ylabel('Amplitude')
        axes_summary[0].grid(True, alpha=0.3)
        
        axes_summary[1].plot(df['Timestamp'], df['No_Baseline'], 'g-', linewidth=0.5, alpha=0.7)
        axes_summary[1].set_title('After Baseline Removal')
        axes_summary[1].set_ylabel('Amplitude')
        axes_summary[1].grid(True, alpha=0.3)
        
        axes_summary[2].plot(df['Timestamp'], df['After_50Hz'], 'm-', linewidth=0.5, alpha=0.7)
        axes_summary[2].set_title('After 50Hz Filter')
        axes_summary[2].set_ylabel('Amplitude')
        axes_summary[2].grid(True, alpha=0.3)
        
        axes_summary[3].plot(df['Timestamp'], df['Processed'], 'r-', linewidth=0.5, alpha=0.7)
        axes_summary[3].set_title('Final Processed Signal')
        axes_summary[3].set_xlabel('Time (seconds)')
        axes_summary[3].set_ylabel('Amplitude')
        axes_summary[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_filename = filename.replace('.csv', '_summary.png')
        plt.savefig(summary_filename, dpi=300)
        print(f"Summary plot saved to: {summary_filename}")
        plt.show()
    
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