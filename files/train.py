from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from data_prep import *
from feature import *
from mlflow.tracking import MlflowClient
from argparse import ArgumentParser


# Initialize the Parser
parser = ArgumentParser(description ='Select a machine learning model: logistic or lgbm.')

# Adding Argument
parser.add_argument('model',
                    metavar ='N',
                    type = str,
                    nargs ='+',
                    help ='a ML model to be used: logistic or lgbm.')

# return a list
args = parser.parse_args()

                    
# if __name__ == "__main__" :
with mlflow.start_run() :
    
    # useful later
    y_train_enrich = train_enrich.TARGET
    y_val_enrich = val_enrich.TARGET
    X_train_enrich = train_enrich.drop('TARGET', axis=1)
    X_val_enrich = val_enrich.drop('TARGET', axis=1)
    features = list(X_train_enrich.columns)

    # imputation  
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(X_train_enrich) 
    X_train_enrich = imputer.transform(X_train_enrich)
    X_val_enrich = imputer.transform(X_val_enrich)
    test_enrich = imputer.transform(test_enrich)

    # scaling
    scaler = StandardScaler()
    scaler.fit(X_val_enrich)
    X_train_enrich = scaler.transform(X_train_enrich)
    X_val_enrich = scaler.transform(X_val_enrich)
    test_enrich = scaler.transform(test_enrich)


    if args.model[0] == 'lgbm': 
        print("process lgbm")
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                    class_weight = 'balanced', learning_rate = 0.05, 
                                    reg_alpha = 0.1, reg_lambda = 0.1, 
                                    subsample = 0.8, n_jobs = -1, random_state = 50)

        model.fit(X_train_enrich, y_train_enrich, eval_metric = 'auc',
                    eval_set = [(X_val_enrich, y_val_enrich), (X_train_enrich, y_train_enrich)],
                    eval_names = ['valid', 'train'],
                    early_stopping_rounds = 500, verbose = 200)
        print("process lgbm")
        mlflow.log_param("class_weight", 'balanced')
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("n_estimators", 10000)


    else:
        model = LogisticRegression(class_weight='balanced', C=0.001)
        model.fit(X_train_enrich, y_train_enrich)

        mlflow.log_param("class_weight", 'balanced')
        mlflow.log_param("C", 0.001)


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file" :
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name=f"{args.model}_model")
    else :
        mlflow.sklearn.log_model(model, "model")
