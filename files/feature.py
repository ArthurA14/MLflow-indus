import data_prep as dp
import utils


# TRAIN_DF_PATH = r'..\\data\\train_df.csv'
# VAL_DF_PATH = r'..\\data\\val_df.csv'
# TEST_DF_PATH = r'..\\data\\test_df.csv' 

# try :
#     # get data
#     train_df = utils.get_data(TRAIN_DF_PATH)
# #     val_df = utils.get_data(VAL_DF_PATH)
# #     test_df = utils.get_data(TEST_DF_PATH)
# except Exception as e :
#     dp.logger.exception("Unable to download training & test CSV. Error: %s", e)

train_df, val_df, test_df = dp.train_df, dp.val_df, dp.test_df


def feature_prep(train_df, val_df, test_df) :
    """
    Function to process feature engineering on data
    
    Args: 
        pd.DataFrame of train raw , val raw and test raw
    
    Returns:
        pd.DataFrame of train, val and test feature engineering
    """  

    train_enrich = train_df.copy()
    val_enrich = val_df.copy()
    test_enrich = test_df.copy()


    ############################ ADDING BUSINESS FEATURES ###########################

    train_enrich = utils.feature_eng(train_enrich)
    val_enrich = utils.feature_eng(val_enrich)
    test_enrich = utils.feature_eng(test_enrich)


    ################## RECORDING OF THE NEW DATAFRAMES IN A HARD COPY ##################

    TRAIN_DF_PATH = r'..\\data\\train_enrich.csv'
    utils.write_data(TRAIN_DF_PATH, train_enrich)

    VAL_DF_PATH = r'..\\data\\val_enrich.csv'
    utils.write_data(VAL_DF_PATH, val_enrich)

    TEST_DF_PATH = r'..\\data\\test_enrich.csv'
    utils.write_data(TEST_DF_PATH, test_enrich)


    return train_enrich, val_enrich, test_enrich


# if __name__ == "__main__" :
train_enrich, val_enrich, test_enrich = feature_prep(train_df, val_df, test_df)
