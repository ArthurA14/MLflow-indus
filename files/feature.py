from data_prep import *
import utils 


train_enrich = train.copy()
val_enrich = val.copy()
test_enrich = test.copy()
# useful later
features = list(test_enrich.columns)

train_enrich, val_enrich, test_enrich = utils.feature_eng(train_enrich, val_enrich, test_enrich)


