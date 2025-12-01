# Spatial data partitioning logic
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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

        continuous_features = torch.tensor([
            row['xlong'],
            row['ylat'],
            row['wind_speed'],
            row['timestamp'] # needs proprocessing
        ], dtype=torch.float32)

        target = torch.tensor(row['capacity_factor'], dtype=torch.float32)

        return {
            'state_id': torch.tensor(state_id, dtype=torch.long),
            'county_id': torch.tensor(county_id, dtype=torch.long),
            'conintuous_features': continuous_features
        }, target

    def get_distributed_dataloader(rank, world_size, batch_size=32):
        # Each rank loads a different spatial partition of the data
        df = pd.read_csv('data/wind_turbine_data.csv')

        state_to_id = {state: i for i, state in enumerate(df['t_state'].unique())} # Iterate through csv and get all unique states
        county_to_id = {county: i for i, county in enumerate(df['t_county'].unique())}

        # IMPORTANT: here we divide states across workers
        all_states = sorted(df['t_state'].unique())
        states_per_worker = len(all_states) // world_size

        start_index = rank * states_per_worker
        end_index = (rank + 1) * states_per_worker if rank < world_size - 1 else len(all_states)

        my_states = all_states[start_index:end_index]
        my_data = df[df['t_state'].isin(my_states)]

        print(f"[Rank {rank}] Assigned states: {my_states}")
        print(f"[Rank {rank}] Data samples: {len(my_data)}")

        dataset = WindTurbineDataset(my_data, state_to_id, county_to_id)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        return dataloader