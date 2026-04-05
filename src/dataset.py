import numpy as np
import pandas as pd
from src.config import (
    LOCATIONS, AP_POSITIONS, SAMPLES_PER_LOCATION,
    NOISE_STD, TX_POWER, PATH_LOSS_EXP
)

def rssi_from_distance(d, tx_power=TX_POWER, n=PATH_LOSS_EXP):
    """Free-space path loss model: RSSI = TxPower - 10*n*log10(d)"""
    d = max(d, 0.1)
    return tx_power - 10 * n * np.log10(d)

def generate_dataset(n_samples=SAMPLES_PER_LOCATION, seed=42):
    """Generate synthetic RSSI fingerprint dataset."""
    np.random.seed(seed)
    X, y = [], []
    for loc_id, (lx, ly) in LOCATIONS.items():
        for _ in range(n_samples):
            row = []
            for (ax, ay) in AP_POSITIONS:
                d = np.sqrt((lx - ax)**2 + (ly - ay)**2)
                rssi = rssi_from_distance(d)
                rssi += np.random.normal(0, NOISE_STD)
                row.append(round(rssi, 2))
            X.append(row)
            y.append(loc_id)
    cols = [f"AP{i+1}" for i in range(len(AP_POSITIONS))]
    df = pd.DataFrame(X, columns=cols)
    df["location"] = y
    return df
