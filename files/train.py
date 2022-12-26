from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from argparse import ArgumentParser
import joblib
import utils
from data_prep import logger



TRAIN_URL = r'../data/train_enrich.csv'
VAL_URL = r'../data/val_enrich.csv'
TEST_URL = r'../data/test_enrich.csv' 


# Create experiment
EXPERIMENT_NAME = "mlflow-demo"
try:
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
except Exception as e:
    logger.exception("Experiment ID already set. Error: %s", e)


try :
    # get data
    train_enrich = utils.get_data(TRAIN_URL)
    val_enrich = utils.get_data(VAL_URL)
    test_enrich = utils.get_data(TEST_URL)
except Exception as e :
    logger.exception("Unable to download training & test CSV. Error: %s", e)

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
with mlflow.start_run(run_name="PARENT_RUN") :
    
    # separate features from target
    y_train_enrich = train_enrich.TARGET
    y_val_enrich = val_enrich.TARGET
    X_train_enrich = train_enrich.drop('TARGET', axis=1)
    X_val_enrich = val_enrich.drop('TARGET', axis=1)

    # imputation  
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(X_train_enrich) 
    X_train_enrich = imputer.transform(X_train_enrich)
    X_val_enrich = imputer.transform(X_val_enrich)
    test_enrich = imputer.transform(test_enrich)
    # saving fitted scaler
    imputer_filename = "imputer.save"
    joblib.dump(imputer, imputer_filename) 

    # scaling
    scaler = StandardScaler()
    scaler.fit(X_val_enrich)
    X_train_enrich = scaler.transform(X_train_enrich)
    X_val_enrich = scaler.transform(X_val_enrich)
    test_enrich = scaler.transform(test_enrich)
    # saving fitted scaler
    scaler_filename = "std_scaler.save"
    joblib.dump(scaler, scaler_filename) 


    if args.model[0] == 'lgbm': 

        best_model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                    class_weight = 'balanced', learning_rate = 0.05, 
                                    reg_alpha = 0.1, reg_lambda = 0.1, 
                                    subsample = 0.8, n_jobs = -1, random_state = 50)

        best_model.fit(X_train_enrich, y_train_enrich, eval_metric = 'auc',
                    eval_set = [(X_val_enrich, y_val_enrich), (X_train_enrich, y_train_enrich)],
                    eval_names = ['valid', 'train'],
                    early_stopping_rounds = 500, verbose = 200)

        # log params
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("n_estimators", 10000)

        y_pred = best_model.predict(X_val_enrich)  

        # get metrics 
        f1score, accuracy = f1_score(y_val_enrich, y_pred), accuracy_score(y_val_enrich, y_pred)
        print('f1_score: ', f1_score(y_val_enrich, y_pred), 'accuracy_score: ', accuracy_score(y_val_enrich, y_pred))  
        mlflow.log_metric("f1_score", f1score)
        mlflow.log_metric("accuracy", accuracy)


    else:
        # Selecting a parameter range to try out
        C = [10, 1, 0.1, 0.01, 0.001, 0.0001]
        # For each value of C, running a child run
        best_model = 0
        best_acc = 0
        for param_value in C:
            with mlflow.start_run(run_name="CHILD_RUN", nested=True):
                # Instantiating and fitting the model
                model = LogisticRegression(class_weight='balanced', C=param_value, max_iter=1000)            
                model.fit(X=X_train_enrich, y=y_train_enrich)
                
                # Logging the current value of C
                mlflow.log_param(key="C", value=param_value)
                
                y_pred = model.predict(X_val_enrich)  

                # get metrics 
                c, f1score, accuracy = param_value, f1_score(y_val_enrich, y_pred), accuracy_score(y_val_enrich, y_pred)
                print('c: ', param_value, ': ', 'f1_score: ', f1_score(y_val_enrich, y_pred), 'accuracy_score: ', accuracy_score(y_val_enrich, y_pred))  
                mlflow.log_metric("f1_score", f1score)
                mlflow.log_metric("accuracy", accuracy)

                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model = model


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file" :
        # There are other ways to use the Model Registry, cf docs:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"{args.model[0]}_model")
    else :
        mlflow.sklearn.log_model(best_model, "model")
