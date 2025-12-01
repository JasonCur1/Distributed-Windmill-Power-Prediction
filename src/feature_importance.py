import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json

from model import WindPowerModel
from data_loader import WindTurbineDataset, load_metadata


def compute_baseline_metrics(model, dataloader, device='cpu'):
    """Compute baseline MAE on original data"""
    model.eval()
    model.to(device)

    all_errors = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            errors = torch.abs(outputs - targets)
            all_errors.extend(errors.cpu().numpy())

    return np.mean(all_errors)


def permutation_importance(model, dataloader, feature_idx, feature_name, device='cpu'):
    """
    Compute importance by permuting a feature and measuring performance drop

    Args:
        model: Trained model
        dataloader: Test data loader
        feature_idx: Index of feature in continuous features tensor (0-3)
        feature_name: Name of the feature
        device: Device to run on

    Returns:
        MAE after permutation
    """
    model.eval()
    model.to(device)

    all_errors = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Permute the specified feature
            if feature_name in ['state', 'county']:
                # For categorical features
                key = 'state_id' if feature_name == 'state' else 'county_id'
                original = inputs[key].clone()
                inputs[key] = inputs[key][torch.randperm(len(inputs[key]))]
            else:
                # For continuous features
                original = inputs['continuous'][:, feature_idx].clone()
                inputs['continuous'][:, feature_idx] = inputs['continuous'][:, feature_idx][torch.randperm(len(inputs['continuous']))]

            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            targets = targets.to(device)

            # Compute error with permuted feature
            outputs = model(inputs)
            errors = torch.abs(outputs - targets)
            all_errors.extend(errors.cpu().numpy())

    return np.mean(all_errors)


def analyze_feature_importance(model_path, test_data_path, device='cpu'):
    """Analyze importance of all features"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60 + "\n")

    # Load model
    print(f"Loading model from {model_path}...")
    model = WindPowerModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    metadata = load_metadata()
    dataset = WindTurbineDataset(df, metadata['state_to_id'], metadata['county_to_id'])
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    # Compute baseline
    print("\nComputing baseline performance...")
    baseline_mae = compute_baseline_metrics(model, dataloader, device)
    print(f"Baseline MAE: {baseline_mae:.6f}")

    # Features to test
    features = [
        ('state', None, 'State (Categorical)'),
        ('county', None, 'County (Categorical)'),
        ('xlong', 0, 'Longitude'),
        ('ylat', 1, 'Latitude'),
        ('wind_speed', 2, 'Wind Speed'),
        ('timestamp', 3, 'Timestamp'),
    ]

    importance_scores = {}

    print("\nComputing permutation importance...")
    for feat_name, feat_idx, display_name in features:
        print(f"  Testing {display_name}...")

        # Compute MAE with this feature permuted
        permuted_mae = permutation_importance(model, dataloader, feat_idx, feat_name, device)

        # Importance = increase in error when feature is permuted
        importance = permuted_mae - baseline_mae
        importance_pct = (importance / baseline_mae) * 100

        importance_scores[display_name] = {
            'importance': importance,
            'importance_pct': importance_pct,
            'permuted_mae': permuted_mae
        }

        print(f"    Permuted MAE: {permuted_mae:.6f}")
        print(f"    Importance: {importance:.6f} (+{importance_pct:.2f}%)")

    # Sort by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1]['importance'], reverse=True)

    # Print summary
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RANKING")
    print("="*60)
    for i, (feat_name, scores) in enumerate(sorted_features, 1):
        print(f"{i}. {feat_name:25s}: +{scores['importance']:.6f} (+{scores['importance_pct']:.2f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute importance
    features_names = [f[0] for f in sorted_features]
    importances = [f[1]['importance'] for f in sorted_features]

    axes[0].barh(features_names, importances)
    axes[0].set_xlabel('Increase in MAE')
    axes[0].set_title('Feature Importance (Absolute)')
    axes[0].grid(True, alpha=0.3)

    # Relative importance
    importances_pct = [f[1]['importance_pct'] for f in sorted_features]

    axes[1].barh(features_names, importances_pct)
    axes[1].set_xlabel('% Increase in MAE')
    axes[1].set_title('Feature Importance (Relative)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/evaluation/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to outputs/evaluation/feature_importance.png")

    # Save results
    results = {
        'baseline_mae': baseline_mae,
        'features': {name: scores for name, scores in sorted_features}
    }

    with open('outputs/evaluation/feature_importance.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to outputs/evaluation/feature_importance.json")

    return importance_scores


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--model_path', type=str, default='outputs/models/final_model.pt')
    parser.add_argument('--test_data', type=str, default='data/wind_turbine_test.csv')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    analyze_feature_importance(args.model_path, args.test_data, args.device)


if __name__ == "__main__":
    main()