import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
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


#create experiement
EXPERIMENT_NAME = "mlflow-demo"
try:
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


############################ Encodage des variables cat #########################

train, val, test = utils.encoding(train, val, test)

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
