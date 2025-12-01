# Spatial data partitioning logic
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from datetime import datetime

class WindTurbineDataset(Dataset):
    def __init__(self, df, state_to_id, county_to_id):
        self.df = df.reset_index(drop=True)
        self.state_to_id = state_to_id
        self.county_to_id = county_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        state_id = self.state_to_id[row['t_state']]
        county_id = self.county_to_id[row['t_county']]

        # Convert timestamp to Unix timestamp (seconds since epoch)
        # Format: 1970-01-01 00:00:02.021010100
        if isinstance(row['timestamp'], str):
            dt = pd.to_datetime(row['timestamp'])
            timestamp_unix = dt.timestamp()
        elif isinstance(row['timestamp'], pd.Timestamp):
            timestamp_unix = row['timestamp'].timestamp()
        else:
            timestamp_unix = float(row['timestamp']) # use as is if already numeric

        continuous_features = torch.tensor([
            row['xlong'],
            row['ylat'],
            row['wind_speed'],
            timestamp_unix
        ], dtype=torch.float32)

        target = torch.tensor(row['capacity_factor'], dtype=torch.float32)

        return {
            'state_id': torch.tensor(state_id, dtype=torch.long),
            'county_id': torch.tensor(county_id, dtype=torch.long),
            'continuous': continuous_features
        }, target

def load_metadata():
    with open('config/metadata.json', 'r') as f:
        metadata = json.load(f)
    return metadata

def load_worker_assignment(num_workers):
    filename = f"config/worker_assignment_{num_workers}.json"
    with open(filename, 'r') as f:
        assignment = json.load(f)
    return assignment

def get_distributed_dataloader(rank, world_size, data_path, batch_size=32, num_workers_dataloader=4):
    print(f"[Rank {rank}] Loading data partition...")

    # Load metadata
    metadata = load_metadata()
    state_to_id = metadata['state_to_id']
    county_to_id = metadata['county_to_id']

    # Load worker assignment
    assignment = load_worker_assignment(world_size)
    my_assignment = assignment['workers'][rank]
    my_states = my_assignment['states']
    expected_samples = my_assignment['num_samples']

    print(f"[Rank {rank}] Assigned {len(my_states)} states: {my_states}")
    print(f"[Rank {rank}] Expected samples: {expected_samples:,}")

    # Load full dataset
    print(f"[Rank {rank}] Reading CSV file...")
    df = pd.read_csv(data_path)

    # Filter to only this rank's states
    my_data = df[df['t_state'].isin(my_states)].copy()
    actual_samples = len(my_data)

    print(f"[Rank {rank}] Actual samples loaded: {actual_samples:,}")

    if abs(actual_samples - expected_samples) > 100:
        print(f"[Rank {rank}] WARNING: Sample count mismatch!")
        print(f"  Expected: {expected_samples:,}, Got: {actual_samples:,}")

    dataset = WindTurbineDataset(my_data, state_to_id, county_to_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_dataloader,
        pin_memory=True,
        persistent_workers=True if num_workers_dataloader > 0 else False
    )

    print(f"[Rank {rank}] DataLoader created with {len(dataloader)} batches")

    return dataloader