from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from data_prep import *
from feature import *
from mlflow.tracking import MlflowClient


# if __name__ == "__main__" :
with mlflow.start_run() :
    # useful later
    y_train_enrich = train_enrich.TARGET
    y_val_enrich = val_enrich.TARGET
    X_train_enrich = train_enrich.drop('TARGET', axis=1)
    X_val_enrich = val_enrich.drop('TARGET', axis=1)
    features = list(X_train_enrich.columns)

    # imputation naive  
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

    print(X_train_enrich.shape)
    print(X_val_enrich.shape)
    print(test_enrich.shape)


    log_reg = LogisticRegression(class_weight='balanced', C=0.001)
    log_reg.fit(X_train_enrich, y_train_enrich)

    train = train.drop('TARGET', axis=1)
    features = list(train.columns)

    # coef_table = pd.DataFrame(list(features))
    # coef_table.insert(len(coef_table.columns),"Coefs",log_reg.coef_.transpose())

    # coef_table.sort_values(by=['Coefs'])

    mlflow.log_param("class_weight", 'balanced')
    mlflow.log_param("C", 0.001)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file" :
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(log_reg, "model", registered_model_name="LogisticRegressionModel")
    else :
        mlflow.sklearn.log_model(log_reg, "model")