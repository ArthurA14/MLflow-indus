import data_prep as dp
import utils


train_enrich = dp.train_df.copy()
val_enrich = dp.val_df.copy()
test_enrich = dp.test_df.copy()


############################ ADDING BUSINESS FEATURES ###########################

train_enrich = utils.feature_eng(train_enrich)
val_enrich = utils.feature_eng(val_enrich)
test_enrich = utils.feature_eng(test_enrich)
