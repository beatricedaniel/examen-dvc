"""
Model training script.
Trains a Gradient Boosting Regressor using the best parameters from GridSearch.
Saves the trained model as a pickle file.
"""

import pandas as pd
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# Define paths
INPUT_DIR = "data/processed_data"
PARAMS_DIR = "models/data"
OUTPUT_DIR = "models/models"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the scaled training data
print("Loading scaled training data...")
X_train_scaled = pd.read_csv(os.path.join(INPUT_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv")).squeeze()

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train shape: {y_train.shape}")

# Load the best parameters from GridSearch
print("\nLoading best parameters...")
best_params = joblib.load(os.path.join(PARAMS_DIR, "best_params.pkl"))
print(f"Best parameters: {best_params}")

# Initialize the Gradient Boosting Regressor with best parameters
gbr_model = GradientBoostingRegressor(
    **best_params,
    random_state=42
)

# Train the model
print("\nTraining the model...")
gbr_model.fit(X_train_scaled, y_train)

print("Model training completed!")

# Save the trained model
model_path = os.path.join(OUTPUT_DIR, "gbr_model.pkl")
joblib.dump(gbr_model, model_path)

print(f"\nTrained model saved to {model_path}")
