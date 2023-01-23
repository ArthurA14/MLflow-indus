import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import utils
import warnings
warnings.filterwarnings('ignore')
import logging


# setting the seed
np.random.seed(40)  

# setting logger 
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


TRAIN_URL = r'../data/application_train.csv'
TEST_URL = r'../data/application_test.csv' 

try :
    # get data
    train = utils.get_data(TRAIN_URL)
    test = utils.get_data(TEST_URL)
except Exception as e :
    logger.exception("Unable to download training & test CSV. Error: %s", e)


def data_prep(train, test):
    """
    Function to preprocess data
    
    Args: 
        pd.DataFrame of train raw and test raw
    
    Returns:
        pd.DataFrame of train, val and test pre processed
    """

    # Split the data into training and test sets. (0.80, 0.20) split.
    train, val = train_test_split(train, test_size=0.20, stratify=train['TARGET'], random_state=42)


    ############################ Encodage des variables cat #########################

    train = utils.encoding(train, is_trainable=True)
    val = utils.encoding(val, is_trainable=False)
    test = utils.encoding(test, is_trainable=False) 

    ############################ SOME DATAFRAME OPERATIONS ##########################

    # remove target 
    train_labels = train['TARGET']
    val_labels = val['TARGET']

    # keep only columns in both dfs
    train, test = train.align(test, join='inner', axis=1)
    train, val = train.align(val, join='inner', axis=1)

    # add back target
    train['TARGET'] = train_labels
    val['TARGET'] = val_labels

    print('Train features shape after processing: ', train.shape)
    print('Val features shape after processing: ', val.shape)
    print('Test features shape after processing: ', test.shape)


    ############################ SOME ANOMALIES OPERATIONS ##########################
    
    train = utils.fix_anomalies(train)
    val = utils.fix_anomalies(val)
    test = utils.fix_anomalies(test)


    ############################ ADDING BUSINESS FEATURES ###########################

    train_enrich = utils.feature_eng(train)
    val_enrich = utils.feature_eng(val)
    test_enrich = utils.feature_eng(test)
    
    URL = r'../data/train_enrich.csv'
    utils.write_data(URL, train_enrich)
    URL = r'../data/val_enrich.csv'
    utils.write_data(URL, val_enrich)
    URL = r'../data/test_enrich.csv'
    utils.write_data(URL, test_enrich) 

    return train, val, test


if __name__ == "__main__" :
    train, val, test = data_prep(train, test)

        
