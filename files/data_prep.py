import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import logging
import utils
import os


# setting logger
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# setting warning
warnings.filterwarnings("ignore")

# setting the seed 
np.random.seed(40)  


def data_prep(TRAIN_DF_PATH, TEST_DF_PATH) :
    """
    Function to preprocess data
    
    Args: 
        pd.DataFrame of train raw and test raw
    
    Returns:
        pd.DataFrame of train, val and test pre processed
    """  

    ######################### GET DATA FROM ORIGINAL DATA FILES #########################

    try :
        # get data
        train_df = utils.get_data(TRAIN_DF_PATH)
        test_df = utils.get_data(TEST_DF_PATH)
    except Exception as e :
        logger.exception("Unable to download training & test CSV. Error: %s", e)


    # Split the data into train_dfing and test_df sets. (0.80, 0.20) split.
    train_df, val_df = train_test_split(train_df, test_size=0.20, stratify=train_df['TARGET'], random_state=42)


    ###################### DROP COLUMNS CONTAINING TOO MUCH NAN VALUES ######################

    train_df = utils.drop_column_with_nan(train_df)
    val_df = utils.drop_column_with_nan(val_df)
    test_df = utils.drop_column_with_nan(test_df)


    ############################ ENCODING OF CATEGORICAL VARIABLES #########################

    train_df = utils.encoding(train_df)
    val_df = utils.encoding(val_df)
    test_df = utils.encoding(test_df)


    ############################ SOME DATAFRAME OPERATIONS ##########################

    # remove target 
    train_df_labels = train_df['TARGET']
    val_df_labels = val_df['TARGET']

    # keep only columns which are present in both dfs
    train_df, test_df = train_df.align(test_df, join='inner', axis=1)
    train_df, val_df = train_df.align(val_df, join='inner', axis=1)

    # add back target
    train_df['TARGET'] = train_df_labels
    val_df['TARGET'] = val_df_labels

    print('train_df shape after processing : ', train_df.shape)
    print('val_df shape after processing : ', val_df.shape)
    print('test_df shape after processing : ', test_df.shape)


    ############################ SOME ANOMALIES OPERATIONS ##########################
    
    train_df = utils.fix_anomalies(train_df)
    val_df = utils.fix_anomalies(val_df)
    test_df = utils.fix_anomalies(test_df)

    ############################ ADDING BUSINESS FEATURES ###########################

    train_enrich = utils.feature_eng(train_df)
    val_enrich = utils.feature_eng(val_df)
    test_enrich = utils.feature_eng(test_df)


    ################## RECORDING OF THE NEW DATAFRAMES IN A HARD COPY ##################

    # if not os.path.exists(r"..\\files\\data") :
    #     os.makedirs(r"..\\files\\data")

    TRAIN_DF_PATH = r'..\\data\\train_enrich.csv'
    utils.write_data(TRAIN_DF_PATH, train_enrich)

    VAL_DF_PATH = r'..\\data\\val_enrich.csv'
    utils.write_data(VAL_DF_PATH, val_enrich)

    TEST_DF_PATH = r'..\\data\\test_enrich.csv'
    utils.write_data(TEST_DF_PATH, test_enrich)


    return train_enrich, val_enrich, test_enrich



# if __name__ == "__main__" :
TRAIN_DF_PATH = r'..\\data\\application_train.csv'
TEST_DF_PATH = r'..\\data\\application_test.csv' 
train_enrich, val_enrich, test_enrich = data_prep(TRAIN_DF_PATH, TEST_DF_PATH)
