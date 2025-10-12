import argparse
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
from datetime import timedelta
import sys

path = Path('max30102_data_20251009_151555.csv')

def format_duration(seconds: float) -> str:
    # Simple H:MM:SS.sss formatting
    total = float(seconds)
    if total < 0:
        total = 0.0
    hrs = int(total // 3600)
    mins = int((total % 3600) // 60)
    secs = total % 60
    return f"{hrs:d}:{mins:02d}:{secs:06.3f}"


def load_data(path: Path):
    df = pd.read_csv(path)

    # find timestamp-like column
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "time")), None)

    if ts_col is not None:
        numeric = pd.to_numeric(df[ts_col], errors="coerce")
        if numeric.notna().any():
            # Treat numeric timestamps as milliseconds -> convert to seconds
            seconds = numeric.astype(float) / 1000.0
            datetimes = pd.to_datetime(seconds, unit="s", origin="unix")
        else:
            datetimes = pd.to_datetime(df[ts_col], errors="coerce")
            seconds = datetimes.view("int64") / 1e9  # ns -> s
    else:
        seconds = pd.Series(df.index.astype(float))
        datetimes = None

    # pick Red and IR columns (case-insensitive)
    def pick(col_lower):
        for c in df.columns:
            if c.lower() == col_lower:
                return df[c].reset_index(drop=True)
        return None

    red = pick("red")
    ir = pick("ir")

    if red is None and ir is None:
        raise ValueError("CSV must contain at least one of 'Red' or 'IR' columns (case-insensitive).")

    seconds = pd.Series(seconds).reset_index(drop=True)
    if isinstance(datetimes, (pd.Series, pd.DatetimeIndex)):
        datetimes = pd.Series(datetimes).reset_index(drop=True)
    else:
        datetimes = None

    return seconds, datetimes, red, ir


def plot_segments(seconds, red, ir, which='both', segment_length=5.0):
    start = float(seconds.min())
    end = float(seconds.max())
    duration = max(0.0, end - start)

    print(f"Samples: {len(seconds)}")
    print(f"Total duration: {duration:.3f} s ({format_duration(duration)})\n")

    # create window edges; ensure at least one window
    edges = np.arange(start, end + 1e-9, segment_length)
    if len(edges) <= 1:
        edges = np.array([start, start + segment_length])

    plotted = 0
    for i in range(len(edges) - 1):
        seg_start = edges[i]
        seg_end = edges[i + 1]
        mask = (seconds >= seg_start) & (seconds < seg_end)
        if mask.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 3))
        x = seconds[mask] if seconds is not None else seconds[mask]

        plotted_series = 0
        if which in ('both', 'red') and red is not None:
            y = red[mask].reset_index(drop=True)
            ax.plot(x, y, color='red', label='Red', linewidth=0.8)
            plotted_series += 1
        if which in ('both', 'ir') and ir is not None:
            y = ir[mask].reset_index(drop=True)
            ax.plot(x, y, color='tab:gray', label='IR', linewidth=0.8)
            plotted_series += 1

        if plotted_series == 0:
            plt.close(fig)
            continue

        if seconds is not None:
            fig.autofmt_xdate()
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("Seconds since start")

        ax.set_ylabel("Sensor reading")
        ax.set_title(f"Segment {i}: {seg_start:.3f}s → {seg_end:.3f}s")
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        plotted += 1

    if plotted == 0:
        print("No data found inside any segment windows. Check timestamps and segment length.")


def main():
    
    global path
    
    which = 'both'  # options: 'red', 'ir', 'both'

    seconds, datetimes, red, ir = load_data(path)
    plot_segments(seconds,red=red, ir=ir, which=which)


if __name__ == "__main__":
    main()