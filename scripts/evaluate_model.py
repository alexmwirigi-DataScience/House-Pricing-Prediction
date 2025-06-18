# 1. IMPORT OUR TOOLS
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import sys
from pathlib import Path
import logging
import json # To save our results in a nice format

# 2. SETUP LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 3. HELP PYTHON FIND OUR BLUEPRINTS
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Now we can import our prediction blueprint function
from house_price_predictor.predict import make_prediction

# 4. DEFINE THE MAIN WORKFLOW
def run_evaluation():
    """Loads test data, makes predictions, and evaluates the model."""
    
    # Define file paths
    test_data_path = project_root / 'data' / 'processed' / 'test.csv'
    model_path = project_root / 'models' / 'random_forest_v1.joblib'
    metrics_path = project_root / 'models' / 'evaluation_metrics.json'

    logging.info(f"Loading test data from {test_data_path}")
    df_test = pd.read_csv(test_data_path)

    # Separate features (X_test) from the true answers (y_test)
    X_test = df_test.drop("median_house_value", axis=1)
    y_test = df_test["median_house_value"]

    # Use our blueprint to get the model's predictions
    predictions = make_prediction(input_data=X_test, model_path=model_path)

    # --- Calculate the scores ---
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    logging.info("--- Model Evaluation Results ---")
    logging.info(f"  R-squared: {r2:.4f}") # .4f formats the number to 4 decimal places
    logging.info(f"  Mean Squared Error: {mse:.4f}")
    logging.info("---------------------------------")

    # --- Save the metrics to a file ---
    # This is a best practice. It lets us track performance over time.
    metrics = {
        'r2_score': r2,
        'mean_squared_error': mse
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"Evaluation metrics saved to {metrics_path}")


# 5. MAKE THE SCRIPT RUNNABLE
if __name__ == '__main__':
    run_evaluation()