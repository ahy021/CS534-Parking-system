import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os


class ParkingDataset(Dataset):
    """
    custom PyTorch Dataset for loading parking meter availability data
    - the model is trained to predict whether a meter will be available x minutes into the future (lookahead)
    """
    def __init__(self, data_dir, lookahead_minutes, Th=60, Td=1, Tw=0):
        # Load features and labels from disk
        self.features = np.load(os.path.join(data_dir, "features.npy"))
        self.labels = np.load(os.path.join(data_dir, f"labels_{lookahead_minutes}.npy"))

        # Load metadata (meter IDs and timestamps)
        with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.meters = meta["meters"]
        self.times = meta["times"]

        self.Th = Th #hour level
        self.Td = Td #number of past days
        self.Tw = Tw #number of past weeks
        self.lookahead = lookahead_minutes
        self.samples = [] #list of valid (meter, timestamp) pairs
        self.build_index() #populate self.samples

    def build_index(self):
        """
        Identifies all valid (meter, time) pairs to use as training samples, skipping any samples that are incomplete or have missing labels
        """
        num_meters, total_minutes, _ = self.features.shape
        for meter in range(num_meters):
            for t in range(self.Th, total_minutes - self.lookahead):
                if self.labels[meter, t + self.lookahead] == -1: #skip if missing
                    continue

                # ensures enough data exists for different timeframes
                if t - self.Th < 0:
                    continue
                if t - 1440 * self.Td < 0 and self.Td > 0:
                    continue
                if t - 10080 * self.Tw < 0 and self.Tw > 0:
                    continue
                self.samples.append((meter, t))

    def __len__(self):
        length = len(self.samples)
        return length

    def __getitem__(self, idx):
        """
        returns a dictionary containing the input tensors:

        dict: {
                Xh: Hour-level input tensor of shape [Th, F],
                Xd: Daily context tensor of shape [Td, Th, F]
                Xw: Weekly context tensor of shape [Tw, Th, F]
                y: Ground truth label (0 or 1)
                meter: Meter index
                time_idx: Time index in the original timeline
            }
        """
        meter,t = self.samples[idx]

        # Hourly input: [Th, F]
        Xh = self.features[meter, t - self.Th:t, :]  # shape: [Th, F]

        # Daily input: [Td, Th, F]
        if self.Td > 0:
            Xd = np.stack([
                self.features[meter, t - 1440 * i: t - 1440 * i + self.Th, :]
                for i in range(1, self.Td + 1)
            ])
        else:
            Xd = np.zeros((1, self.Th, self.features.shape[2]), dtype=np.float32)

        # Weekly input: [Tw, Th, F]
        if self.Tw > 0:
            Xw = np.stack([
                self.features[meter, t - 10080 * i: t - 10080 * i + self.Th, :]
                for i in range(1, self.Tw + 1)
            ])
        else:
            Xw = np.zeros((1, self.Th, self.features.shape[2]), dtype=np.float32)

        # Label for future availability
        y = self.labels[meter, t + self.lookahead]

        return {
            'Xh': torch.tensor(Xh, dtype=torch.float32),
            'Xd': torch.tensor(Xd, dtype=torch.float32),
            'Xw': torch.tensor(Xw, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'meter': meter,
            'time_idx': t
        }
