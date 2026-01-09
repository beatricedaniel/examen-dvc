"""
Data splitting script for silica concentration prediction.
Splits the dataset into training and testing sets (80/20).
Target variable: silica_concentrate (last column)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define paths
RAW_DATA_PATH = "data/raw_data/raw.csv"
OUTPUT_DIR = "data/processed_data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the raw data
print("Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH)

# Separate features and target
# Exclude 'date' (first column) and 'silica_concentrate' (last column) from features
# Target is the last column: silica_concentrate
X = df.iloc[:, 1:-1]  # All columns except first (date) and last (target)
y = df.iloc[:, -1]    # Last column (silica_concentrate)

print(f"Dataset shape: {df.shape}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")

# Save the split datasets
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

print(f"\nSplit datasets saved to {OUTPUT_DIR}/")
print("Files created:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")
