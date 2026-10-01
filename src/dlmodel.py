import torch
import torch.nn as nn
from config import config

torch.manual_seed(13)

class DLModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        layer1size, layer2size, layer3size = config.params.nn.layers_size
        dropout_rate = config.params.nn.dropout

        self.model = nn.Sequential(
            nn.Linear(input_dim, layer1size),
            nn.BatchNorm1d(layer1size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(layer1size, layer2size),
            nn.BatchNorm1d(layer2size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(layer2size, layer3size),
            nn.BatchNorm1d(layer3size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(layer3size, 1)
        )

    def forward(self, x):
        return self.model(x)