"""
Model evaluation script.
Evaluates the trained model on the test set and calculates metrics.
Saves predictions and evaluation metrics.
"""

import pandas as pd
import os
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define paths
INPUT_DIR = "data/processed_data"
MODEL_DIR = "models/models"
OUTPUT_DATA_DIR = "data/processed_data"
OUTPUT_METRICS_DIR = "metrics"

# Create output directories if they don't exist
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

# Load the scaled test data
print("Loading scaled test data...")
X_test_scaled = pd.read_csv(os.path.join(INPUT_DIR, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv")).squeeze()

print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_test shape: {y_test.shape}")

# Load the trained model
print("\nLoading trained model...")
model = joblib.load(os.path.join(MODEL_DIR, "gbr_model.pkl"))

# Make predictions
print("Making predictions on test set...")
y_pred = model.predict(X_test_scaled)

print(f"Predictions shape: {y_pred.shape}")

# Calculate evaluation metrics
print("\nCalculating evaluation metrics...")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Save predictions
predictions_df = pd.DataFrame({
    'y_test': y_test.values,
    'y_pred': y_pred
})
predictions_path = os.path.join(OUTPUT_DATA_DIR, "predictions.csv")
predictions_df.to_csv(predictions_path, index=False)

# Save metrics as JSON
metrics = {
    "mse": float(mse),
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2)
}

scores_path = os.path.join(OUTPUT_METRICS_DIR, "scores.json")
with open(scores_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nPredictions saved to {predictions_path}")
print(f"Metrics saved to {scores_path}")
