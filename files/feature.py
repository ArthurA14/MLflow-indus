# import data_prep as dp
# import utils


# def feature_prep(TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH) :
#     """
#     Function to process feature engineering on data
    
#     Args: 
#         pd.DataFrame of train raw , val raw and test raw
    
#     Returns:
#         pd.DataFrame of train, val and test feature engineering
#     """  

#     ########################## GET DATA FROM DATA_PREP FILE #########################
    
#     try :
#         # get data
#         train_df = utils.get_data(TRAIN_DF_PATH)
#         val_df = utils.get_data(VAL_DF_PATH)
#         test_df = utils.get_data(TEST_DF_PATH)
#     except Exception as e :
#         dp.logger.exception("Unable to download training & test CSV. Error: %s", e)

#     # train_df, val_df, test_df = dp.data_prep(TRAIN_DF_PATH, TEST_DF_PATH)

#     train_enrich = train_df.copy()
#     val_enrich = val_df.copy()
#     test_enrich = test_df.copy()


#     ############################ ADDING BUSINESS FEATURES ###########################

#     train_enrich = utils.feature_eng(train_enrich)
#     val_enrich = utils.feature_eng(val_enrich)
#     test_enrich = utils.feature_eng(test_enrich)


#     ################## RECORDING OF THE NEW DATAFRAMES IN A HARD COPY ##################

#     TRAIN_DF_PATH = r'..\\data\\train_enrich.csv'
#     utils.write_data(TRAIN_DF_PATH, train_enrich)

#     VAL_DF_PATH = r'..\\data\\val_enrich.csv'
#     utils.write_data(VAL_DF_PATH, val_enrich)

#     TEST_DF_PATH = r'..\\data\\test_enrich.csv'
#     utils.write_data(TEST_DF_PATH, test_enrich)


#     return train_enrich, val_enrich, test_enrich



# # if __name__ == "__main__" :
# TRAIN_DF_PATH = r'..\\data\\train_df.csv'
# VAL_DF_PATH = r'..\\data\\val_df.csv'
# TEST_DF_PATH = r'..\\data\\test_df.csv'
# train_enrich, val_enrich, test_enrich = feature_prep(TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH)
