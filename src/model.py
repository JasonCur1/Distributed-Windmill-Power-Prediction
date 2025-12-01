"""
Neural Network for predicting wind turbine capacity factor.
"""

import torch
import torch.nn as nn

class WindPowerModel(nn.Module):
    def __init__(self, num_states=37, num_counties=395):
        super().__init__()

        # Categorical features
        self.state_embedding = nn.Embedding(num_states, 16)
        self.county_embedding = nn.Embedding(num_counties, 32)

        self.feature_net = nn.Sequential( # These are the continuous features
            nn.Linear(4, 64), # xlong, ylat, wind_speed, timestamp features
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        # Combine all features: state(16) + county(32) + features(128) = 176
        self.output_net = nn.Sequential(
            nn.Linear(176, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Capacity factor is between 0 and 1
        )

    def forward(self, x):
        """
        x: dict with keys 'state_id', 'county_id', 'continuous_features'
        """
        state_emb = self.state_embedding(x['state_id'])
        county_emb = self.county_embedding(x['county_id'])

        features = self.feature_net(x['continuous'])

        combined = torch.cat([state_emb, county_emb, features], dim=1)

        output = self.output_net(combined)

        return output.squeeze(-1)
