import torch
import torch.nn as nn


class TemporalBlock(nn.Module):

    """
    temporal feature extractor using 1D convolution, then ReLU, the pooling
    Inputs: number rof input features per time step and number of output features from the convolutional layer
    """
    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) #compresses sequence dimension to 1
        )


    def forward(self, x):
        """
            forward pass through the temporal block

            Inputs: tensor
            Outputs: compressed tensor (batch size, hidden dimensions)
            """
        x = x.permute(0, 2, 1) #switch to (batch_size, in_channels, seq_len) for Conv1d
        out = self.temporal(x)
        tensor = out.squeeze(-1) #remove the last dimension
        return tensor

class MFFSTGCN(nn.Module):
    """
        Multi-Frequency Fusion Graph Convolutional Network.

        This model separately processes short-term (hour), mid-term (day), and long-term (week)
        sequences using TemporalBlocks, then fuses their representations for final prediction.

        Inputs:
        in_channels: Number of features per time step for each input sequence.
        hidden_dim: Number of hidden dimensions in each TemporalBlock.
        """
    def __init__(self, in_channels=17, hidden_dim=32):
        super().__init__()
        # Temporal encoders
        self.hour_block = TemporalBlock(in_channels, hidden_dim)
        self.day_block = TemporalBlock(in_channels, hidden_dim)
        self.week_block = TemporalBlock(in_channels, hidden_dim)

        # Final classifer combining time scales
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() #outputs probability
        )

    def forward(self, Xh, Xd, Xw):
        """
        forward oass of the model

        Inputs:
        Xh: hour level
        Xd: day level
        Xw: week level
        Outputs: tensor
        """
        h = self.hour_block(Xh)
        d = self.day_block(Xd[:, 0]) # only first day
        w = self.week_block(Xw[:, 0]) # only first week
        features = torch.cat([h, d, w], dim=1)
        tensor = self.classifier(features).squeeze(-1)
        return tensor
