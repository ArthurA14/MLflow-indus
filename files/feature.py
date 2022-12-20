from data_prep import *


# if __name__ == "__main__" :
train_enrich = train.copy()
val_enrich = val.copy()
test_enrich = test.copy()

train_enrich['CREDIT_INCOME_PERCENT'] = train_enrich['AMT_CREDIT'] / train_enrich['AMT_INCOME_TOTAL']
train_enrich['ANNUITY_INCOME_PERCENT'] = train_enrich['AMT_ANNUITY'] / train_enrich['AMT_INCOME_TOTAL']
train_enrich['CREDIT_TERM'] = train_enrich['AMT_ANNUITY'] / train_enrich['AMT_CREDIT']
train_enrich['DAYS_EMPLOYED_PERCENT'] = train_enrich['DAYS_EMPLOYED'] / train_enrich['DAYS_BIRTH']

val_enrich['CREDIT_INCOME_PERCENT'] = val_enrich['AMT_CREDIT'] / val_enrich['AMT_INCOME_TOTAL']
val_enrich['ANNUITY_INCOME_PERCENT'] = val_enrich['AMT_ANNUITY'] / val_enrich['AMT_INCOME_TOTAL']
val_enrich['CREDIT_TERM'] = val_enrich['AMT_ANNUITY'] / val_enrich['AMT_CREDIT']
val_enrich['DAYS_EMPLOYED_PERCENT'] = val_enrich['DAYS_EMPLOYED'] / val_enrich['DAYS_BIRTH']

test_enrich['CREDIT_INCOME_PERCENT'] = test_enrich['AMT_CREDIT'] / test_enrich['AMT_INCOME_TOTAL']
test_enrich['ANNUITY_INCOME_PERCENT'] = test_enrich['AMT_ANNUITY'] / test_enrich['AMT_INCOME_TOTAL']
test_enrich['CREDIT_TERM'] = test_enrich['AMT_ANNUITY'] / test_enrich['AMT_CREDIT']
test_enrich['DAYS_EMPLOYED_PERCENT'] = test_enrich['DAYS_EMPLOYED'] / test_enrich['DAYS_BIRTH']
