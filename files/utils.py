import pandas as pd


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
