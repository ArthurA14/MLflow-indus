import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib


def get_data(URL) :
 
    """This method will be used to extract data from a csv file

        :param str URL: Path to the csv file 

        :returns: The extracted pandas dataframe

        :rtype: pd.DataFrame
    """
    return pd.read_csv(URL, sep=",")
    

def write_data(URL, df) : 

    """This method will be used to write a dataframe to csv

        :param str URL: Path to the csv file 
        :param pd.DataFrame df: Dataframe to save as a csv file

        :returns: The CSV file

        :rtype: csv
    """

    return df.to_csv(URL, sep=",", index=False, header=True)


def fix_anomalies(df):
    """
    Helper to spot anomalies in the data
    
    Args: 
        pd.DataFrame: dataframe to inspect
    
    Returns:
        pd.DataFrame 
    """

    # on indique les valeurs abbérantes detectées lors de l'EDA avec un booléen 
    df['DAYS_EMPLOYED_ANOMALIES'] = df["DAYS_EMPLOYED"] == 365243
    # on les remplace avec np.nan pour imputation ultérieure
    df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    
    return df


def encoding(df, is_trainable=True):
    """
    An encoding helper function to encode categorical vars
    
    Args: 
        pd.DataFrame: dataset to encode
        bool: True if train set, False otherwise. Default: True.
    
    Returns:
        pd.DataFrame: dataset with encoded categorical features
    """ 
    
    print('Features shape before encoding: ', df.shape) 

    # label encoding categorial variables if cat count <= 2
    le = LabelEncoder()
    encoder_filename = "encoder.save"   
    encoded_cols = list()
    if is_trainable:
        for col in df:
            if df[col].dtype == 'object' :
                if len(list(df[col].unique())) <= 2:
                        # Train on the training data and avoid leakage
                        le.fit(df[col])
                        # Transform
                        df[col] = le.transform(df[col])
                        encoded_cols.append(col)
        # saving fitted encoder
        joblib.dump(le, encoder_filename) 
    else:
        le = joblib.load(encoder_filename) 
        for col in encoded_cols:
            if df[col].dtype == 'object' :
                if len(list(df[col].unique())) <= 2:
                    df[col] = le.transform(df[col])
    

    # one-hot encoding of remaining categorical variables
    df = pd.get_dummies(df) 

    print('Features shape after encoding: ', df.shape) 

    return df 


def feature_eng(df):
    """
    A feature engineering function to create new dataframes with business features
    
    Args: 
        pd.DataFrame: dataset to enrich 
    
    Returns:
        pd.DataFrame: dataset with added business features
    """

    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df
