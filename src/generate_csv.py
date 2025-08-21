import pandas as pd
from sklearn.datasets import load_digits
import os


# Load digits dataset (8x8 grayscale digits, values roughly 0..16)
digits = load_digits()
X = digits.data            # shape: (1797, 64)
y = digits.target          # shape: (1797,)


# Build DataFrame with named pixel columns + label
pix_cols = [f"pix{i}" for i in range(64)]
df = pd.DataFrame(X, columns=pix_cols)
df["label"] = y


# Create data folder and save
os.makedirs("../data", exist_ok=True)
csv_path = "../data/digit_data.csv"
df.to_csv(csv_path, index=False)


print(f"CSV generated at: {csv_path} with {len(df)} samples.")
