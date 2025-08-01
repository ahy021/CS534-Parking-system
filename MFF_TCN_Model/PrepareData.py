import pandas as pd
import numpy as np
import os
import pickle

TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p" #format of the OccupancyDateTime column
LOOKAHEADS = [5, 10, 30] #minutes ahead to look at


def load_raw_data(csv_path):
    """
    load_raw_data loads and preprocesses the Seattle parking CSV
    - parses timestamps
    - sorts by time and SourceElementKey
    - computes availability at parking meters
    - adds one-hot encoded features for the hours from 8:00 to 18:00

    Inputs: CSV path
    Outputs: preprocessed dataframe with added features
    """
    df = pd.read_csv(csv_path)
    df["OccupancyDateTime"] = pd.to_datetime(df["OccupancyDateTime"], format=TIME_FORMAT)
    df.sort_values(by=["OccupancyDateTime", "SourceElementKey"], inplace=True) #sorts values in the dataframe

    df["available_count"] = df["ParkingSpaceCount"] - df["PaidOccupancy"] #feature: availability_count = [total spots available at parking meter] - [number of spots paid for at parking meter]

    # One-hot hour: 8:00 to 22:00 (15 possible values)
    df["hour"] = df["OccupancyDateTime"].dt.hour
    for h in range(8, 23):
        df[f"hour_{h}"] = (df["hour"] == h).astype(int)

    return df


def generate_labels(df, lookaheads):
    """
    adds binary labels for future parking availability
    - each label corresponds to whether a parking space will be available x minutes in the future

    Inputs: preprocessed dataframe and minutes to lookahead
    Outputs: dataframe with additional label columns
    """
    df = df.copy()
    df.set_index("OccupancyDateTime", inplace=True)
    for x in lookaheads:
        shifted = df.groupby("SourceElementKey")["ParkingSpaceCount"].shift(-x)
        occupied = df.groupby("SourceElementKey")["PaidOccupancy"].shift(-x)
        available = shifted - occupied
        df[f"label_{x}min"] = (available > 0).astype(int)
    df.reset_index(inplace=True)
    return df

def build_time_grid(df):
    """
    Builds a time range (every minute) from the dataset's satrt to end

    Inputs: the dataframe
    Outputs: all minute-level timestamps from min to max
    """
    time_grid = pd.date_range(
                    start=df["OccupancyDateTime"].min().floor("T"),
                    end=df["OccupancyDateTime"].max().ceil("T"),
                    freq="1min"
    )
    return time_grid


def reshape_tensor(df, lookaheads):
    """
    Converts dataframe into structured numpy tensors
    - outputs a feature tensor [num_meters, num_minutes, num_features]
    - outputs a label tensor [num_meters. num_minutes]

    Inputs: dataframe and lookaheads
    Outputs:
    tuple:
    - feature_tensor (np.ndarray): features [N, T, F=13]
    - label_tensors (dict): {lookahead: label tensor [N, T]}
    - all_meters (list): ordered list of meter IDs
    - all_times (DatetimeIndex): full minute-by-minute timeline
    """
    all_meters = sorted(df["SourceElementKey"].unique())
    all_times = build_time_grid(df)
    meter_index = {m: i for i, m in enumerate(all_meters)}
    time_index = {t: i for i, t in enumerate(all_times)}

    num_meters = len(all_meters)
    num_times = len(all_times)
    F = 17  # 15 one-hot hours + availability flag + availability count

    feature_tensor = np.zeros((num_meters, num_times, F), dtype=np.float32)
    label_tensors = {x: np.full((num_meters, num_times), fill_value=-1, dtype=np.int8) for x in lookaheads}

    for _, row in df.iterrows():
        m_idx = meter_index[row["SourceElementKey"]]
        t_idx = time_index.get(row["OccupancyDateTime"], None)
        if t_idx is None:
            continue

        # Feature vector: one-hot hour, availability flag, available count
        feat = []
        for h in range(8, 23):
            feat.append(row.get(f"hour_{h}", 0))
        feat.append(1.0 if row["available_count"] > 0 else 0.0)
        feat.append(float(row["available_count"]))

        feature_tensor[m_idx, t_idx, :] = feat

        for x in lookaheads:
            label = row.get(f"label_{x}min", -1)
            label_tensors[x][m_idx, t_idx] = label

    return feature_tensor, label_tensors, all_meters, all_times

def save_prepared_data(output_dir, features, labels_dict, meters, times):
    """
    saves processed features, labels, and metadata to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features.npy"), features)
    for x, labels in labels_dict.items():
        np.save(os.path.join(output_dir, f"labels_{x}.npy"), labels)
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({"meters": meters, "times": times}, f)

# Data prep workflow
if __name__ == "__main__":
    df = load_raw_data("Paid_Parking__Last_48_Hours_.csv")
    df = generate_labels(df, LOOKAHEADS)
    features, labels_dict, meters, times = reshape_tensor(df, LOOKAHEADS)
    save_prepared_data("prepared_data", features, labels_dict, meters, times)
