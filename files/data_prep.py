import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import utils
import warnings
warnings.filterwarnings('ignore')
import logging


# setting the seed
np.random.seed(40)  

# setting logger 
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


TRAIN_URL = r'../../data/application_train.csv'
TEST_URL = r'../../data/application_test.csv' 


try:
    EXPERIMENT_NAME = "mlflow-demo"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
except Exception as e:
    logger.exception("Experiment ID already set. Error: %s", e)


try :
    # get data
    train = utils.get_data(TRAIN_URL)
    test = utils.get_data(TEST_URL)
except Exception as e :
    logger.exception("Unable to download training & test CSV. Error: %s", e)


# Split the data into training and test sets. (0.80, 0.20) split.
train, val = train_test_split(train, test_size=0.20, stratify=train['TARGET'], random_state=42)


############################ Encodage des variables catégorielles ##########################
def encoding(train_df, val_df, test_df):
    """
    An encoding helper function to create new dataframes with business features
    
    Args: 
        pd.DataFrame: train dataset
        pd.DataFrame: validation dataset
        pd.DataFrame: test dataset
    
    Returns:
        pd.DataFrame: three train, validation and test datasets with encoded categorical features
    """

    print('Training Features shape: ', train_df.shape)
    print('Training Features shape: ', val_df.shape)
    print('Testing Features shape: ', test_df.shape)

    # label encoding categorial variables if cat count <= 2
    le = LabelEncoder()
    le_cols = 0

    for col in train_df:
        if train_df[col].dtype == 'object' :
            if len(list(train_df[col].unique())) <= 2:
                # Train on the training data and avoid leakage
                le.fit(train_df[col])
                # Transform both training and testing data
                train_df[col] = le.transform(train_df[col])
                val_df[col] = le.transform(val_df[col])
                test_df[col] = le.transform(test_df[col])
                le_cols += 1

    # one-hot encoding of remaining categorical variables
    train = pd.get_dummies(train_df)
    val = pd.get_dummies(val_df)
    test = pd.get_dummies(test_df)

    print('Training Features shape after encoding: ', train.shape)
    print('Training Features shape after encoding: ', val.shape)
    print('Testing Features shape after encoding: ', test.shape)

    return train, val, test

train, val, test = encoding(train, val, test)


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

print('Train df shape: ', train.shape)
print('Val df shape: ', val.shape)
print('Test df shape: ', test.shape)


############################ SOME ANOMALIES OPERATIONS ##########################

# on indique les valeurs abbérantes detectées lors de l'EDA avec un booléen 
train['DAYS_EMPLOYED_ANOMALIES'] = train["DAYS_EMPLOYED"] == 365243

# on les remplace avec np.nan pour imputation ultérieure
train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# on réplique les manipulations dans les dfs de val et de test
val['DAYS_EMPLOYED_ANOMALIES'] = val["DAYS_EMPLOYED"] == 365243
val["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
test['DAYS_EMPLOYED_ANOMALIES'] = test["DAYS_EMPLOYED"] == 365243
test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
