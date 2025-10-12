import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============ CHANGE THIS TO YOUR FILE ============
csv_file = 'max30102_processed_20251013_020240.csv'
# ==================================================

def plot_csv(filename):
    """Simple plotter for MAX30102 data"""
    
    # Check if file exists
    path = Path(filename)
    if not path.exists():
        print(f"Error: File '{filename}' not found!")
        print("Please update the 'csv_file' variable at the top.")
        return
    
    # Load data
    print(f"Loading: {filename}")
    df = pd.read_csv(filename)
    print(f"Columns found: {list(df.columns)}")
    print(f"Rows: {len(df):,}\n")
    
    # Find timestamp column
    time_col = None
    for col in df.columns:
        if col.lower() in ['timestamp', 'time']:
            time_col = col
            break
    
    if time_col is None:
        print("No timestamp column found, using row index")
        time_data = df.index.values
    else:
        time_data = df[time_col].values
    
    # Find all signal columns (exclude timestamp)
    signal_cols = [col for col in df.columns if col != time_col]
    
    if len(signal_cols) == 0:
        print("No signal columns found!")
        return
    
    # Create subplots
    n_signals = len(signal_cols)
    fig, axes = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))
    
    # Handle single plot case
    if n_signals == 1:
        axes = [axes]
    
    # Color scheme
    colors = ['blue', 'green', 'magenta', 'red', 'orange', 'purple', 'brown']
    
    # Plot each signal
    for i, (ax, col) in enumerate(zip(axes, signal_cols)):
        color = colors[i % len(colors)]
        
        data = df[col].values
        
        ax.plot(time_data, data, color=color, linewidth=0.8, alpha=0.8)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        stats_text = f'μ={mean_val:.1f}, σ={std_val:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Label bottom plot
    axes[-1].set_xlabel('Time (seconds)' if time_col else 'Sample Number', fontsize=10)
    
    # Overall title
    duration = time_data[-1] - time_data[0] if time_col else len(df)
    sample_rate = len(df) / duration if time_col and duration > 0 else 0
    
    if time_col:
        title = f'{filename} | {len(df):,} samples | {duration:.2f}s | {sample_rate:.1f} Hz'
    else:
        title = f'{filename} | {len(df):,} samples'
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_file = filename.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SIGNAL STATISTICS")
    print("="*60)
    for col in signal_cols:
        data = df[col].values
        print(f"{col:20s} | Mean: {np.mean(data):10.2f} | Std: {np.std(data):10.2f}")
    
    # Calculate noise reduction if both raw and processed exist
    raw_col = next((c for c in signal_cols if 'raw' in c.lower()), None)
    proc_col = next((c for c in signal_cols if 'processed' in c.lower() or 'final' in c.lower()), None)
    
    if raw_col and proc_col:
        raw_std = np.std(df[raw_col])
        proc_std = np.std(df[proc_col])
        reduction = (1 - proc_std / raw_std) * 100
        print("="*60)
        print(f"Noise Reduction: {reduction:.2f}%")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    plot_csv(csv_file)