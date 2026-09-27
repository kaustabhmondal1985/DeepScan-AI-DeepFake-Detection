from typing import Tuple

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):

    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # x shape:
        # [batch, sequence_length, 2048]

        output, (hidden, cell) = self.lstm(x)

        # output:
        # [batch, sequence_length, hidden_size]

        # hidden:
        # [num_layers, batch, hidden_size]

        # Use final layer's hidden state
        temporal_embedding = hidden[-1]

        return output, temporal_embedding