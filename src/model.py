"""
Neural Network for predicting wind turbine capacity factor.
"""

import torch
import torch.nn as nn

class WindPowerModel(nn.Module):
    def __init__(self, num_states=50, num_counties=3000):
        super().__init__()

        # Categorical features
        self.state_embedding = nn.Embedding(num_states, 16)
        self.county_embedding = nn.Embedding(num_counties, 32)

        self.feature_net = nn.Sequential( # These are the continuous features
            nn.Linear(4, 64), # xlong, ylat, wind_speed, timestamp features
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Combine all features
        self.output_net = nn.Sequential(
            nn.Linear(16 + 32 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Predict capacity_factor
            nn.Sigmoid()  # Capacity factor is between 0 and 1
        )

    def forward(self, x):
        """
        x: dict with keys 'state_id', 'county_id', 'continuous_features'
        """
        state_emb = self.state_embedding(x['state_id'])
        county_emb = self.county_embedding(x['county_id'])
        features = self.feature_net(x['continuous_features'])

        combined = torch.cat([state_emb, county_emb, features], dim=1)
        output = self.output_net(combined)

        return output.squeeze()
