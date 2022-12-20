import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import os
import warnings
warnings.filterwarnings('ignore')

import logging

EXPERIMENT_NAME = "mlflow-demo"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def get_data(URL) :
    return pd.read_csv(URL, sep=",", header='infer')

def write_data(URL, df) : 
    return df.to_csv(URL, sep=";", index=False, header=True)

# if __name__ == "__main__" :
warnings.filterwarnings("ignore")
np.random.seed(40)  

train_url = r'..\\data\\application_train.csv'
test_url = r'..\\data\\application_test.csv' 

try :
    # get data
    train_dataset = get_data(train_url)
    test_dataset = get_data(test_url)
except Exception as e :
    logger.exception("Unable to download training & test CSV. Error: %s", e)

# Split the data into training and test sets. (0.80, 0.20) split.
train, val = train_test_split(train_dataset, test_size=0.20, stratify=train_dataset['TARGET'], random_state=42)

# missing values
mis_val = train.isnull().sum()
mis_val_percent = 100.00 * train.isnull().sum() / len(train)
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
mis_val_table_rename_columns = mis_val_table.rename(columns={0:'Missing Values', 1:'Percentage'})

mask = mis_val_table_rename_columns.iloc[:,1] != 0
mis_val_table_rename_columns = mis_val_table_rename_columns[mask].sort_values('Percentage', ascending=False).round(1)


# Encoding categorial variables
le = LabelEncoder()
le_cols = 0

for col in train:
    if train[col].dtype == 'object' :
        if len(list(train[col].unique())) <= 2:
            # Train on the training data and avoid leakage
            le.fit(train[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])
            val[col] = le.transform(val[col])
            test_dataset[col] = le.transform(test_dataset[col])
            
            le_cols += 1

# one-hot encoding of categorical variables
train = pd.get_dummies(train)
val = pd.get_dummies(val)
test = pd.get_dummies(test_dataset)

print('Training Features shape: ', train.shape)
print('Training Features shape: ', val.shape)
print('Testing Features shape: ', test.shape)


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

# on indique les valeurs abbérantes avec un booléen 
train['DAYS_EMPLOYED_ANOMALIES'] = train["DAYS_EMPLOYED"] == 365243
# on les remplace avec np.nan pour imputation ultérieure
train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
# on réplique les manipulations dans les dfs de val et de test
val['DAYS_EMPLOYED_ANOMALIES'] = val["DAYS_EMPLOYED"] == 365243
val["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
test['DAYS_EMPLOYED_ANOMALIES'] = test["DAYS_EMPLOYED"] == 365243
test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
