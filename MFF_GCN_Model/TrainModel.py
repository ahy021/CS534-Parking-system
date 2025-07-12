import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from ParkingDataset import ParkingDataset
from MFFSTGCNModel import MFFSTGCN
import os
import time

def evaluate(model, loader, device):
    """
    evaluates model on validation or test set

    returns dictionary of log loss, brier score, and ROC-AUC score
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch["Xh"].to(device), batch["Xd"].to(device), batch["Xw"].to(device))
            y_true += batch["y"].tolist()
            y_pred += out.tolist()

    y_pred_clipped = []
    for p in y_pred:
        if p < 1e-7:
            y_pred_clipped.append(1e-7)
        elif p > 1 - 1e-7:
            y_pred_clipped.append(1 - 1e-7)
        else:
            y_pred_clipped.append(p)

    # dictionary of performance metrics
    dict_metrics ={
        "log_loss": log_loss(y_true, y_pred_clipped),
        "brier": brier_score_loss(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }
    return dict_metrics

def train_one_epoch(model, loader, optimizer, device):
    """
    trains the model for one epoch and returns the average training BCE loss over the epoch
    """
    model.train()
    total_loss = 0
    criterion = torch.nn.BCELoss()
    start_time = time.time()

    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        out = model(batch["Xh"].to(device),
                    batch["Xd"].to(device),
                    batch["Xw"].to(device))
        loss = criterion(out, batch["y"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #print batch, loss, and time taken to run (for tracking purposes)
        if i % 10000 == 0:
            print(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | Time: {time.time() - start_time:.2f}s")
            start_time = time.time()

    avg_training_loss = total_loss / len(loader)
    return avg_training_loss

if __name__ == "__main__":
    lookahead = 30 #looking 10 mins ahead
    dataset = ParkingDataset("prepared_data", lookahead, Td=1, Tw=0) # use 1 day prior for Xd and (for now) skip week data
    print(f"Total samples: {len(dataset)}")

    # Temporal Train/Val/Test Split
    sorted_samples = sorted(dataset.samples, key=lambda x: x[1])  # by time_idx
    sample_to_idx = {s: i for i, s in enumerate(dataset.samples)}

    # Extract test set from last full day
    last_time = sorted_samples[-1][1]
    last_day_start = last_time - (last_time % 1440)  # minute-level alignment
    test_samples = [s for s in sorted_samples if s[1] >= last_day_start]
    train_val_samples = [s for s in sorted_samples if s[1] < last_day_start]

    # Split train/val (80/20) from remaining samples
    split_index = int(0.8 * len(train_val_samples))
    train_samples = train_val_samples[:split_index]
    val_samples = train_val_samples[split_index:]

    # Use fast index mapping
    train_ds = Subset(dataset, [sample_to_idx[s] for s in train_samples])
    val_ds   = Subset(dataset, [sample_to_idx[s] for s in val_samples])
    test_ds  = Subset(dataset, [sample_to_idx[s] for s in test_samples])

    print(f"Train: {len(train_ds)} - Val: {len(val_ds)} | - Test: {len(test_ds)}")

    # --- Device selection ---
    if torch.cuda.is_available():
        print(f"CUDA is available. {torch.cuda.device_count()} GPU(s) detected:")
        for i in range(torch.cuda.device_count()):
            print(f"  [GPU {i}] {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Final device in use: {device}")

    torch.set_num_threads(os.cpu_count()) #set number of threads for PyTorch

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = MFFSTGCN(in_channels=17).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = 0.0

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"Val LogLoss: {val_metrics['log_loss']:.4f}, Brier: {val_metrics['brier']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")

        # Save best model
        if val_metrics['roc_auc'] > best_auc:
            best_auc = val_metrics['roc_auc']
            torch.save(model.state_dict(), "best_model_30.pt")
            print("Saved new best model to best_model_30.pt")

    # Final Test Evaluation
    print("\nEvaluating best model on test set:")
    model.load_state_dict(torch.load("best_model_30.pt"))
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Log Loss: {test_metrics['log_loss']:.4f}, Brier: {test_metrics['brier']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")
