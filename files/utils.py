import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_data(URL) :
    """
    Helper to extract data from a csv file
    
    Args: 
        path to the csv file
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(URL, sep=",", header='infer')
    

def write_data(URL, df) : 
    """
    Helper write a dataframe to csv
    
    Args: 
        Path: path to the csv file
        pd.DataFrame: dataframe to save as a csv file
    
    Returns:
        CSV file
    """
    return df.to_csv(URL, sep=";", index=False, header=True)


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


def feature_eng(train_df, val_df, test_df):
    """
    A feature engineering function to create new dataframes with business features
    
    Args: 
        pd.DataFrame: train dataset
        pd.DataFrame: validation dataset
        pd.DataFrame: test dataset
    
    Returns:
        pd.DataFrame: three train, validation and test datasets with added business features
    """

    train_df['CREDIT_INCOME_PERCENT'] = train_df['AMT_CREDIT'] / train_df['AMT_INCOME_TOTAL']
    train_df['ANNUITY_INCOME_PERCENT'] = train_df['AMT_ANNUITY'] / train_df['AMT_INCOME_TOTAL']
    train_df['CREDIT_TERM'] = train_df['AMT_ANNUITY'] / train_df['AMT_CREDIT']
    train_df['DAYS_EMPLOYED_PERCENT'] = train_df['DAYS_EMPLOYED'] / train_df['DAYS_BIRTH']

    val_df['CREDIT_INCOME_PERCENT'] = val_df['AMT_CREDIT'] / val_df['AMT_INCOME_TOTAL']
    val_df['ANNUITY_INCOME_PERCENT'] = val_df['AMT_ANNUITY'] / val_df['AMT_INCOME_TOTAL']
    val_df['CREDIT_TERM'] = val_df['AMT_ANNUITY'] / val_df['AMT_CREDIT']
    val_df['DAYS_EMPLOYED_PERCENT'] = val_df['DAYS_EMPLOYED'] / val_df['DAYS_BIRTH']

    test_df['CREDIT_INCOME_PERCENT'] = test_df['AMT_CREDIT'] / test_df['AMT_INCOME_TOTAL']
    test_df['ANNUITY_INCOME_PERCENT'] = test_df['AMT_ANNUITY'] / test_df['AMT_INCOME_TOTAL']
    test_df['CREDIT_TERM'] = test_df['AMT_ANNUITY'] / test_df['AMT_CREDIT']
    test_df['DAYS_EMPLOYED_PERCENT'] = test_df['DAYS_EMPLOYED'] / test_df['DAYS_BIRTH']

    return train_df, val_df, test_df
