"""Run this standalone to save the dataset to CSV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import generate_dataset

df = generate_dataset()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rssi_dataset.csv")
df.to_csv(out, index=False)
print(f"Dataset saved: {out}")
print(df.describe().round(2))
