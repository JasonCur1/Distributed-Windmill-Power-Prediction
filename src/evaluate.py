import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from torch.utils.data import DataLoader

from model import WindPowerModel
from data_loader import WindTurbineDataset, load_metadata


def load_test_data(test_csv_path):
    """Load test dataset"""
    print(f"Loading test data from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)

    metadata = load_metadata()
    state_to_id = metadata['state_to_id']
    county_to_id = metadata['county_to_id']

    dataset = WindTurbineDataset(df, state_to_id, county_to_id)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    print(f"Test set size: {len(df):,} samples")
    return dataloader, df


def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate model and return predictions + ground truth"""
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            for key in inputs:
                inputs[key] = inputs[key].to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(dataloader)} batches")

    return np.array(all_predictions), np.array(all_targets)


def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    mse_val = mean_squared_error(targets, predictions)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(targets, predictions)
    r2_val = r2_score(targets, predictions)

    # MAPE calculation: Avoid division by zero issues if possible, though +1e-8 helps
    mape_val = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    metrics = {
        'mse': float(mse_val),
        'rmse': float(rmse_val),
        'mae': float(mae_val),
        'r2': float(r2_val),
        'mape': float(mape_val),
    }

    # Additional statistics
    errors = predictions - targets
    metrics['mean_error'] = float(np.mean(errors))
    metrics['std_error'] = float(np.std(errors))
    metrics['max_error'] = float(np.max(np.abs(errors)))

    return metrics


def plot_predictions(predictions, targets, output_path='outputs/evaluation/predictions_vs_actual.png'):
    """Plot predicted vs actual capacity factors"""
    plt.figure(figsize=(10, 10))

    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.3, s=1)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Capacity Factor')
    plt.ylabel('Predicted Capacity Factor')
    plt.title('Predicted vs Actual Capacity Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_error_distribution(predictions, targets, output_path='outputs/evaluation/error_distribution.png'):
    """Plot distribution of prediction errors"""
    errors = predictions - targets

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error histogram
    axes[0, 0].hist(errors, bins=50, edgecolor='black')
    axes[0, 0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Prediction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Absolute error histogram
    axes[0, 1].hist(np.abs(errors), bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Absolute Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Absolute Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Error vs actual value
    axes[1, 0].scatter(targets, errors, alpha=0.3, s=1)
    axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Actual Capacity Factor')
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Error vs Actual Value')
    axes[1, 0].grid(True, alpha=0.3)

    # Error vs predicted value
    axes[1, 1].scatter(predictions, errors, alpha=0.3, s=1)
    axes[1, 1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Capacity Factor')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Error vs Predicted Value')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def analyze_by_state(predictions, targets, df, output_path='outputs/evaluation/performance_by_state.png'):
    """Analyze model performance by state"""
    df['prediction'] = predictions
    df['error'] = predictions - targets
    df['abs_error'] = np.abs(predictions - targets)

    # Compute metrics per state
    state_metrics = df.groupby('t_state').agg({
        'abs_error': 'mean',
        'error': 'std',
        'capacity_factor': 'count'
    }).reset_index()
    state_metrics.columns = ['state', 'mae', 'std_error', 'count']
    state_metrics = state_metrics.sort_values('mae', ascending=False)

    # Plot top 20 states by error
    plt.figure(figsize=(12, 8))
    top_states = state_metrics.head(20)
    plt.barh(top_states['state'], top_states['mae'])
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('State')
    plt.title('Top 20 States by Prediction Error')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

    return state_metrics


def analyze_by_wind_speed(predictions, targets, df, output_path='outputs/evaluation/performance_by_wind_speed.png'):
    """Analyze model performance across different wind speeds"""
    df['prediction'] = predictions
    df['error'] = predictions - targets
    df['abs_error'] = np.abs(predictions - targets)

    # Bin wind speeds
    df['wind_speed_bin'] = pd.cut(df['wind_speed'], bins=10)

    wind_metrics = df.groupby('wind_speed_bin').agg({
        'abs_error': 'mean',
        'capacity_factor': 'count'
    }).reset_index()
    wind_metrics.columns = ['wind_speed_bin', 'mae', 'count']

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # MAE by wind speed
    x_labels = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in wind_metrics['wind_speed_bin']]
    axes[0].bar(range(len(wind_metrics)), wind_metrics['mae'])
    axes[0].set_xticks(range(len(wind_metrics)))
    axes[0].set_xticklabels(x_labels, rotation=45)
    axes[0].set_xlabel('Wind Speed Bin (m/s)')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('Prediction Error by Wind Speed')
    axes[0].grid(True, alpha=0.3)

    # Sample count by wind speed
    axes[1].bar(range(len(wind_metrics)), wind_metrics['count'])
    axes[1].set_xticks(range(len(wind_metrics)))
    axes[1].set_xticklabels(x_labels, rotation=45)
    axes[1].set_xlabel('Wind Speed Bin (m/s)')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Sample Distribution by Wind Speed')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

    return wind_metrics


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained wind power model')
    parser.add_argument('--model_path', type=str, default='outputs/models/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/wind_turbine_test.csv',
                        help='Path to test dataset')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run evaluation on')
    args = parser.parse_args()

    # Create output directory
    os.makedirs('outputs/evaluation', exist_ok=True)

    print("\n" + "="*60)
    print("WIND POWER MODEL EVALUATION")
    print("="*60 + "\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = WindPowerModel()

    checkpoint = torch.load(args.model_path, map_location=args.device)

    # Check if the checkpoint is a wrapper dictionary or just weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("  Detected checkpoint dictionary. Loading 'model_state_dict'...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  Detected direct state dict. Loading directly...")
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully\n")

    # Load test data
    test_loader, test_df = load_test_data(args.test_data)

    # Run evaluation
    predictions, targets = evaluate_model(model, test_loader, device=args.device)

    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    metrics = compute_metrics(predictions, targets)

    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():20s}: {value:.6f}")

    # Save metrics
    with open('outputs/evaluation/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to outputs/evaluation/metrics.json")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")

    plot_predictions(predictions, targets)
    plot_error_distribution(predictions, targets)

    # Analyze by state
    print("\nAnalyzing performance by state...")
    state_metrics = analyze_by_state(predictions, targets, test_df.copy())
    state_metrics.to_csv('outputs/evaluation/state_metrics.csv', index=False)
    print("State metrics saved to outputs/evaluation/state_metrics.csv")

    # Analyze by wind speed
    print("\nAnalyzing performance by wind speed...")
    wind_metrics = analyze_by_wind_speed(predictions, targets, test_df.copy())
    wind_metrics.to_csv('outputs/evaluation/wind_speed_metrics.csv', index=False)
    print("Wind speed metrics saved to outputs/evaluation/wind_speed_metrics.csv")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print("\nAll outputs saved to outputs/evaluation/")


if __name__ == "__main__":
    main()