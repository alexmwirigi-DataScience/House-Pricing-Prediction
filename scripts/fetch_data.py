import os
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# define a function 
def load_data(data_url: str, data_csv: str):
   
    try:
        os.makedirs(os.path.dirname(data_csv), exist_ok=True)

        if not os.path.exists(data_csv):
            logging.info(f"Downloading data to {data_csv}")
            response = requests.get(data_url)
            response.raise_for_status()

            with open(data_csv, "wb") as f:
                f.write(response.content)

            logging.info("Download completed successfully.")
        else:
            logging.info(f"The file already exists at {data_csv}, skipping download.")

    except requests.RequestException as e:
        logging.error(f"Download failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    DATA_PATH = os.path.join("data", "raw", "california-housing-prices.csv")
    
    load_data(DATA_URL, DATA_PATH)
