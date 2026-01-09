"""
Data normalization script.
Normalizes the training and test features using StandardScaler.
Fits the scaler on training data and applies it to both train and test sets.
"""

import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Define paths
INPUT_DIR = "data/processed_data"
OUTPUT_DIR = "data/processed_data"
MODELS_DIR = "models/data"

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the split datasets
print("Loading split datasets...")
X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on training data
print("\nFitting scaler on training data...")
scaler.fit(X_train)

# Transform both training and test data
print("Transforming training and test data...")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save the scaled datasets
X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_test_scaled.csv"), index=False)

# Save the scaler for later use (if needed)
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"\nScaled datasets saved to {OUTPUT_DIR}/")
print("Files created:")
print("  - X_train_scaled.csv")
print("  - X_test_scaled.csv")
print(f"  - scaler.pkl (saved to {MODELS_DIR}/)")
