"""
GridSearch script for hyperparameter tuning.
Uses Gradient Boosting Regressor to find the best hyperparameters.
Saves the best parameters as a pickle file.
"""

import pandas as pd
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define paths
INPUT_DIR = "data/processed_data"
OUTPUT_DIR = "models/data"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the scaled training data
print("Loading scaled training data...")
X_train_scaled = pd.read_csv(os.path.join(INPUT_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv")).squeeze()

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"y_train shape: {y_train.shape}")

# Define the parameter grid for GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Perform GridSearch with cross-validation
print("\nPerforming GridSearch with cross-validation...")
print(f"Parameter grid: {param_grid}")
print("This may take a few minutes...")

grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nBest parameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"Best cross-validation score (neg MSE): {best_score:.4f}")

# Save the best parameters
best_params_path = os.path.join(OUTPUT_DIR, "best_params.pkl")
joblib.dump(best_params, best_params_path)

print(f"\nBest parameters saved to {best_params_path}")
