"""
Evaluation Metrics for Glucose Prediction

Implements standard metrics for glucose monitoring:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Clarke Error Grid Analysis (CEGA)

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Square Error.

    Args:
        y_true (np.ndarray): True glucose values (mg/dL)
        y_pred (np.ndarray): Predicted glucose values (mg/dL)

    Returns:
        float: RMSE in mg/dL
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.

    Args:
        y_true (np.ndarray): True glucose values (mg/dL)
        y_pred (np.ndarray): Predicted glucose values (mg/dL)

    Returns:
        float: MAE in mg/dL
    """
    return mean_absolute_error(y_true, y_pred)


def compute_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): True glucose values (mg/dL)
        y_pred (np.ndarray): Predicted glucose values (mg/dL)

    Returns:
        float: MAPE as percentage
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_r2(y_true, y_pred):
    """
    Compute R² (coefficient of determination).

    Args:
        y_true (np.ndarray): True glucose values (mg/dL)
        y_pred (np.ndarray): Predicted glucose values (mg/dL)

    Returns:
        float: R² score (0-1, higher is better)
    """
    return r2_score(y_true, y_pred)


def clarke_error_grid_zone(true_bg, pred_bg):
    """
    Determine Clarke Error Grid zone for a single prediction.

    Zones:
    - A: Clinically accurate (within 20% or both <70)
    - B: Benign errors (outside A but not leading to inappropriate treatment)
    - C: Overcorrection (risk of treating acceptable BG as extreme)
    - D: Failure to detect extreme BG
    - E: Confusing hypo/hyperglycemia

    Args:
        true_bg (float): True blood glucose (mg/dL)
        pred_bg (float): Predicted blood glucose (mg/dL)

    Returns:
        str: Zone letter ('A', 'B', 'C', 'D', or 'E')
    """
    # Zone A: Clinically accurate
    if (true_bg < 70 and pred_bg < 70) or \
       (abs(true_bg - pred_bg) <= 0.2 * true_bg):
        return 'A'

    # Zone E: Confusing hypo/hyperglycemia
    if (true_bg <= 70 and pred_bg >= 180) or \
       (true_bg >= 180 and pred_bg <= 70):
        return 'E'

    # Zone C: Overcorrection
    if (true_bg >= 70 and true_bg <= 290 and pred_bg >= true_bg + 110) or \
       (true_bg >= 130 and true_bg <= 180 and pred_bg <= (7/5) * true_bg - 182):
        return 'C'

    # Zone D: Failure to detect
    if (true_bg >= 240 and pred_bg >= 70 and pred_bg <= 180) or \
       (true_bg <= 70 and pred_bg >= 180) or \
       (true_bg <= 70 and pred_bg >= 70 and pred_bg <= 180):
        return 'D'

    # Zone B: Benign errors (everything else)
    return 'B'


def clarke_error_grid_analysis(y_true, y_pred, show_plot=True, save_path=None):
    """
    Perform Clarke Error Grid Analysis.

    Args:
        y_true (np.ndarray): True glucose values (mg/dL)
        y_pred (np.ndarray): Predicted glucose values (mg/dL)
        show_plot (bool): Whether to display the plot
        save_path (str, optional): Path to save the plot

    Returns:
        dict: Zone percentages {'A': %, 'B': %, 'C': %, 'D': %, 'E': %}
    """
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for true_bg, pred_bg in zip(y_true, y_pred):
        zone = clarke_error_grid_zone(true_bg, pred_bg)
        zones[zone] += 1

    # Convert to percentages
    total = len(y_true)
    zone_percentages = {k: (v / total) * 100 for k, v in zones.items()}

    # Create plot
    if show_plot or save_path:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot reference lines for zones
        ax.plot([0, 400], [0, 400], 'k--', alpha=0.5, linewidth=1)  # Perfect prediction

        # Zone boundaries (simplified representation)
        ax.axhline(70, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(70, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(180, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(180, color='gray', linestyle='--', alpha=0.3)

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)

        # Labels and formatting
        ax.set_xlabel('True Blood Glucose (mg/dL)', fontsize=12)
        ax.set_ylabel('Predicted Blood Glucose (mg/dL)', fontsize=12)
        ax.set_title('Clarke Error Grid Analysis', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(400, np.max(y_true) * 1.1))
        ax.set_ylim(0, max(400, np.max(y_pred) * 1.1))
        ax.grid(True, alpha=0.3)

        # Add zone percentages as text
        zone_text = '\n'.join([f"Zone {k}: {v:.1f}%" for k, v in zone_percentages.items()])
        ax.text(0.02, 0.98, zone_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Clarke Error Grid saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    return zone_percentages


def compute_all_metrics(y_true, y_pred):
    """
    Compute all evaluation metrics.

    Args:
        y_true (np.ndarray): True glucose values (mg/dL)
        y_pred (np.ndarray): Predicted glucose values (mg/dL)

    Returns:
        dict: Dictionary with all metrics
    """
    metrics = {
        'RMSE': compute_rmse(y_true, y_pred),
        'MAE': compute_mae(y_true, y_pred),
        'MAPE': compute_mape(y_true, y_pred),
        'R2': compute_r2(y_true, y_pred)
    }

    # Clarke Error Grid zones (without plot)
    zones = clarke_error_grid_analysis(y_true, y_pred, show_plot=False)
    metrics['Clarke_Zones'] = zones

    return metrics


def print_metrics(metrics):
    """
    Pretty print metrics.

    Args:
        metrics (dict): Dictionary from compute_all_metrics
    """
    print("="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"RMSE:  {metrics['RMSE']:.2f} mg/dL")
    print(f"MAE:   {metrics['MAE']:.2f} mg/dL")
    print(f"MAPE:  {metrics['MAPE']:.2f}%")
    print(f"R²:    {metrics['R2']:.4f}")
    print("\nClarke Error Grid Analysis:")
    for zone, percentage in metrics['Clarke_Zones'].items():
        print(f"  Zone {zone}: {percentage:.1f}%")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("Testing Evaluation Metrics")
    print("="*60)

    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 500

    # True glucose values
    y_true = np.random.uniform(70, 180, n_samples)

    # Predicted with some error (MAE ~10 mg/dL)
    y_pred = y_true + np.random.normal(0, 10, n_samples)

    # Clip to realistic range
    y_pred = np.clip(y_pred, 40, 400)

    print(f"\nTest data: {n_samples} samples")
    print(f"True range: {y_true.min():.1f} - {y_true.max():.1f} mg/dL")
    print(f"Pred range: {y_pred.min():.1f} - {y_pred.max():.1f} mg/dL")

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred)

    # Print results
    print_metrics(metrics)

    # Test Clarke Error Grid with plot
    print("\nGenerating Clarke Error Grid plot...")
    clarke_error_grid_analysis(
        y_true,
        y_pred,
        show_plot=False,
        save_path='test_clarke_grid.png'
    )

    print("\n" + "="*60)
    print("Evaluation metrics working correctly")
    print("="*60)
