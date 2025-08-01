import torch
import pickle
import random
from ParkingDataset import ParkingDataset
from MFFSTGCNModel import MFFSTGCN
from datetime import timedelta
import pandas as pd

LOOKAHEAD = 30
NUM_METERS = 10
DATA_DIR = "prepared_data"
MODEL_PATH = "best_model_30.pt"
Th = 60

with open(f"{DATA_DIR}/metadata.pkl", "rb") as f:
    meta = pickle.load(f)
all_meters = meta["meters"]
all_times = meta["times"]

dataset = ParkingDataset(DATA_DIR, lookahead_minutes=LOOKAHEAD, Th=Th, Td=1, Tw=0)
samples = dataset.samples
cutoff = all_times[-1] - timedelta(minutes=LOOKAHEAD + Th)
valid_samples = [(m, t) for m, t in samples if meta["times"][t] <= cutoff]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFFSTGCN(in_channels=17).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

max_attempts = 1000
for _ in range(max_attempts):
    chosen = random.sample(valid_samples, NUM_METERS)
    results = []

    for meter_idx, time_idx in chosen:
        sample_idx = dataset.samples.index((meter_idx, time_idx))
        batch = dataset[sample_idx]
        Xh = batch["Xh"].unsqueeze(0).to(device)
        Xd = batch["Xd"].unsqueeze(0).to(device)
        Xw = batch["Xw"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(Xh, Xd, Xw).item()

        results.append({
            "MeterID": meta["meters"][meter_idx],
            "Timestamp": meta["times"][time_idx],
            "Predicted_Probability": round(prob, 4),
            "Actual_Label": int(batch["y"].item())
        })

    mid_preds = sum(0.5 < r["Predicted_Probability"] < 0.9 for r in results)
    low_preds = sum(r["Predicted_Probability"] < 0.5 for r in results)

    if mid_preds >= 3 and low_preds >= 2:
        break
else:
    raise RuntimeError("Couldn't find suitable sample set after 1000 attempts.")

df = pd.DataFrame(results)
print(df.to_string(index=False))
