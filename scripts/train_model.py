# 1. IMPORT OUR TOOLS
import sys
from pathlib import Path
import logging

# 2. SETUP LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 3. HELP PYTHON FIND OUR BLUEPRINTS
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Now we can import our training blueprint function
from house_price_predictor.train import train


# 4. DEFINE THE MAIN WORKFLOW
def run_training():
    """Defines the paths and runs the model training."""
    
    # Define where our training data is
    data_path = project_root / 'data' / 'processed' / 'train.csv'
    
    # Define where we want to save our finished model
    # It's good practice to name it something descriptive
    model_output_path = project_root / 'models' / 'random_forest_v1.joblib'

    # Call the training function from our blueprint
    try:
        train(data_path=data_path, model_output_path=model_output_path)
    except Exception as e:
        logging.error("An error occurred during model training.")
        logging.error(e, exc_info=True) # This logs the full error traceback
        raise # Re-raise the exception to stop the script

# 5. MAKE THE SCRIPT RUNNABLE
if __name__ == '__main__':
    run_training()