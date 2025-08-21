from sklearn.datasets import load_digits
import pandas as pd
import os

# Load sklearn digits dataset
digits = load_digits()
X = digits.data       # shape: (1797, 64)
y = digits.target     # shape: (1797,)

# Create DataFrame
df = pd.DataFrame(X, columns=[f"pixel_{i}" for i in range(64)])
df["label"] = y

# Save to CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/digit_data.csv", index=False)
print("CSV file created: data/digit_data.csv")
