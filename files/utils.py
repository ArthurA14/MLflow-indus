import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import joblib


def get_data(path) :
    """
    Helper to extract data from a csv file
    
    Args: 
        path to the csv file
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(path, sep=",", header='infer')


def write_data(path, df) : 
    """
    Helper which writes a dataframe to csv
    
    Args: 
        Path: path to the csv file
        pd.DataFrame: dataframe to save as a csv file
    
    Returns:
        CSV file
    """
    return df.to_csv(path, sep=";", index=False, header=True)


def drop_column_with_nan(df) :
    """
    An helper function which drop columns with a too large amount of nan values
    
    Args: 
        pd.DataFrame: any dataset
    
    Returns:
        pd.DataFrame: the same dataset stripped of these columns
    """
    mask = df.isnull().any(axis=0) # a columns list with missing data
    columns_with_nan  = df.columns[mask]
    for column in columns_with_nan:
        if df[column].isnull().sum() / df.shape[0] > 0.60:
            df.drop(column, 1, inplace=True)
    return df


def fix_anomalies(df) :
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


def encoding(df, is_train=True) :
    """
    An encoding helper function to create new dataframes with business features
    
    Args: 
        pd.DataFrame: dataframe to encode
        boolean: True if train set, false otherwise. Default: True
    
    Returns:
        pd.DataFrame: dataframe with encoded categorical features
    """

    print(f'{df} features shape : ', df.shape)

    # Label encoding categorial variables if categorial count <= 2
    le = LabelEncoder()
    encoded_columns = []

    if is_train : 
        for column in df :
            if df[column].dtype == 'object' :
                if len(list(df[column].unique())) <= 2:
                    # Training on the training data and avoid leakage
                    le.fit(df[column])
                    # Transform train data
                    df[column] = le.transform(df[column])
                    encoded_columns.append(column)

        # saving fitted encoder
        joblib.dump(le, "../data/encoder.save") 
    
    else : 
        le = joblib.load("../data/encoder.save") 
        for column in encoded_columns :
            if df[column].dtype == 'object' :
                if len(list(df[column].unique())) <= 2:
                    df[column] = le.transform(df[column])    


    # one-hot encoding of categorical variables
    df = pd.get_dummies(df)

    print(f'{df} features shape after encoding : ', df.shape)

    return df


def feature_eng(df) :
    """
    A feature engineering function to create new dataframes with business features
    
    Args: 
        pd.DataFrame: dataframe to enrich
    
    Returns:
        pd.DataFrame: dataframe with added business features
    """

    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df
