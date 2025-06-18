import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#create a function that splits data
#input1 = df:pd.dataframe
#input2 = test_size = 0.2
#input3 = random_state = 42
#input3 = stratify col= = none

def split_data(df:pd.DataFrame, test_size = 0.2 , random_state = 42, stratify_col = None):
    if stratify_col:
        stratify_vals = df[stratify_col]
        logging.info(f"Stratifying by column: '{stratify_col}'")
    else:
        stratify_vals = None
        logging.info("No stratification used.")

    train_df,test_df = train_test_split(df,test_size = test_size, random_state=random_state, stratify = stratify_vals)
    logging.info(f"Split complete → Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df

 
     

     




