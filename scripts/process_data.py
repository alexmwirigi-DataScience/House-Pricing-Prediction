# 1. IMPORT TOOLS
import pandas as pd
from pathlib import Path
import sys
import logging

# 2. SETUP LOGGING
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 3. FIND PROJECT ROOT AND IMPORT OUR OWN TOOL
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))  # Allow Python to find our custom modules

# Import your custom split function (must be in src or proper folder)
from processing.preprocessor import split_data


# 4. MAIN WORKFLOW FUNCTION
def run_processing():
    """
    Loads the raw housing data, fills missing values, splits into train/test,
    and saves the processed data to disk.
    """

    # ---- Paths ----
    raw_data_path = project_root / 'data' / 'raw' / 'california-housing-prices.csv'
    processed_data_folder = project_root / 'data' / 'processed'
    train_path = processed_data_folder / 'train.csv'
    test_path = processed_data_folder / 'test.csv'

    # ---- Create Output Folder if Needed ----
    processed_data_folder.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load Raw Data ----
    logging.info(f"📥 Loading raw data from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)

    # ---- Step 2: Clean Missing Values ----
    if 'total_bedrooms' in raw_df.columns:
        median_val = raw_df['total_bedrooms'].median()
        raw_df['total_bedrooms'].fillna(median_val, inplace=True)
        logging.info(f"🧽 Filled missing 'total_bedrooms' with median: {median_val}")
    else:
        logging.warning("'total_bedrooms' column not found — skipping missing value treatment.")

    # ---- Step 3: Split Data ----
    logging.info("✂️ Splitting dataset into train/test sets...")
    train_df, test_df = split_data(raw_df)

    # ---- Step 4: Save Processed Data ----
    logging.info(f"💾 Saving training data to: {train_path}")
    train_df.to_csv(train_path, index=False)

    logging.info(f"💾 Saving testing data to: {test_path}")
    test_df.to_csv(test_path, index=False)

    logging.info("✅ Data processing pipeline completed successfully!")


# 5. RUN THE PIPELINE
if __name__ == "__main__":
    run_processing()
