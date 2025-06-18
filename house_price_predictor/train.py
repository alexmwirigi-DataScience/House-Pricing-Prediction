# 1. IMPORT OUR TOOLS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import logging
from pathlib import Path

# 2. SETUP LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 3. MAIN TRAIN FUNCTION
def train(data_path: str, model_output_path: str):
    """
    Trains a Random Forest Regressor on housing data and saves the model.

    Args:
        data_path (str): Path to the training CSV file.
        model_output_path (str): Path to save the trained model file (.pkl or .joblib).
    """
    logging.info("🚀 Starting model training...")

    # --- Step 1: Load Data ---
    df_train = pd.read_csv(data_path)
    logging.info(f"📊 Loaded training data with {len(df_train)} rows and {len(df_train.columns)} columns.")

    # --- Step 2: One-Hot Encode Categorical Column ---
    if 'ocean_proximity' in df_train.columns:
        df_train = pd.get_dummies(df_train, columns=['ocean_proximity'], drop_first=True)
        logging.info("🔣 Applied one-hot encoding to 'ocean_proximity'.")
    else:
        logging.warning("⚠️ 'ocean_proximity' column not found — skipping encoding.")

    # --- Step 3: Split into X and y ---
    if 'median_house_value' not in df_train.columns:
        raise ValueError("Target column 'median_house_value' not found in training data.")

    X_train = df_train.drop("median_house_value", axis=1)
    y_train = df_train["median_house_value"]
    model_columns = X_train.columns

    logging.info(f"✅ Prepared features (X) and target (y). Feature count: {len(model_columns)}")

    # --- Step 4: Define the Model ---
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # use all CPUs
    )
    logging.info("🧠 Training RandomForestRegressor model...")
    model.fit(X_train, y_train)
    logging.info("🎯 Model training complete.")

    # --- Step 5: Save the Model and Metadata ---
    model_payload = {
        'model': model,
        'columns': model_columns
    }

    # Ensure output folder exists
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save with joblib
    joblib.dump(model_payload, model_output_path)
    logging.info(f"💾 Model saved to: {model_output_path}")
