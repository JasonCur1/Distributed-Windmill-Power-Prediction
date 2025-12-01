"""
Script to create train/test split from the full dataset
Run this BEFORE training to create proper train/test splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def create_split(input_csv, train_csv, test_csv, test_size=0.2, random_state=42):
    """
    Create train/test split ensuring spatial distribution is preserved

    Args:
        input_csv: Path to full dataset
        train_csv: Output path for training set
        test_csv: Output path for test set
        test_size: Fraction of data for test set (default 0.2 = 20%)
        random_state: Random seed for reproducibility
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df):,}")

    # Stratified split by state to ensure each state is represented in both sets
    print(f"\nCreating {int((1-test_size)*100)}%/{int(test_size*100)}% train/test split...")
    print("Stratifying by state to preserve spatial distribution...")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['t_state']  # Ensure each state is proportionally represented
    )

    print(f"\nTrain set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Verify state distribution
    print("\nVerifying state distribution...")
    train_states = set(train_df['t_state'].unique())
    test_states = set(test_df['t_state'].unique())
    all_states = set(df['t_state'].unique())

    print(f"  States in train: {len(train_states)}")
    print(f"  States in test: {len(test_states)}")
    print(f"  States in both: {len(train_states & test_states)}")
    print(f"  Total unique states: {len(all_states)}")

    if train_states & test_states == all_states:
        print("  All states represented in both train and test sets")
    else:
        missing_in_train = all_states - train_states
        missing_in_test = all_states - test_states
        if missing_in_train:
            print(f"  States missing in train: {missing_in_train}")
        if missing_in_test:
            print(f"  States missing in test: {missing_in_test}")

    # Save splits
    print(f"\nSaving train set to {train_csv}...")
    train_df.to_csv(train_csv, index=False)

    print(f"Saving test set to {test_csv}...")
    test_df.to_csv(test_csv, index=False)

    print("\nTrain/test split complete!")

    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    for name, data in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n{name} SET:")
        print(f"  Samples: {len(data):,}")
        print(f"  States: {data['t_state'].nunique()}")
        print(f"  Counties: {data['t_county'].nunique()}")
        print(f"  Capacity factor range: [{data['capacity_factor'].min():.4f}, {data['capacity_factor'].max():.4f}]")
        print(f"  Wind speed range: [{data['wind_speed'].min():.2f}, {data['wind_speed'].max():.2f}]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create train/test split for wind turbine data')
    parser.add_argument('--input', type=str, default='data/wind_turbine_data.csv',
                        help='Input CSV file with full dataset')
    parser.add_argument('--train_output', type=str, default='data/wind_turbine_train.csv',
                        help='Output CSV file for training set')
    parser.add_argument('--test_output', type=str, default='data/wind_turbine_test.csv',
                        help='Output CSV file for test set')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    create_split(
        args.input,
        args.train_output,
        args.test_output,
        args.test_size,
        args.random_state
    )


if __name__ == "__main__":
    main()