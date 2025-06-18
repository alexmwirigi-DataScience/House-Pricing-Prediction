import pandas as pd
import joblib
import logging

def make_prediction(input_data, model_path: str):
    """
    Loads a trained model and makes predictions on input data.

    Args:
        input_data (pd.DataFrame): The data to make predictions on.
        model_path (str): The path to the saved model file.

    Returns:
        An array of predictions.
    """
    logging.info("Loading model for prediction...")
    # Load the model payload (which includes the model and the column order)
    model_payload = joblib.load(model_path)
    model = model_payload['model']
    model_columns = model_payload['columns']
    
    # --- Prepare the input data ---
    # We must perform the exact same transformations as we did on the training data.
    # 1. One-hot encode the 'ocean_proximity' column
    input_data = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)
    
    # 2. Realign columns to match the model's training order
    # This is a CRITICAL step. It ensures that if the input data has columns
    # in a different order or is missing some, we fix it.
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    logging.info("Making predictions...")
    predictions = model.predict(input_data)
    
    logging.info("Predictions made successfully.")
    return predictions