from data_prep import *
from feature import *
from train import *
import utils


EXPERIMENT_NAME = "mlflow-demo"
client = MlflowClient()

# Retrieve Experiment information
EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# Retrieve Runs information (parameter 'depth', metric 'accuracy')
# ALL_RUNS_INFO = client.list_run_infos(EXPERIMENT_ID)
# ALL_RUNS_ID = [run.run_id for run in ALL_RUNS_INFO]
# ALL_PARAM = [client.get_run(run_id).data.params["class_weight"] for run_id in ALL_RUNS_ID]
# ALL_PARAM = [client.get_run(run_id).data.params["C"] for run_id in ALL_RUNS_ID]
# ALL_METRIC = [client.get_run(run_id).data.metrics["f1_score"] for run_id in ALL_RUNS_ID]
# ALL_METRIC = [client.get_run(run_id).data.metrics["accuracy"] for run_id in ALL_RUNS_ID]

if __name__ == "__main__" :
    with mlflow.start_run() :

        y_pred = model.predict(X_val_enrich)  
        c, f1score, accuracy = 0.001, f1_score(y_val_enrich, y_pred), accuracy_score(y_val_enrich, y_pred)
        print('c: ', 0.001, ': ', 'f1_score: ', f1_score(y_val_enrich, y_pred), 'accuracy_score: ', accuracy_score(y_val_enrich, y_pred)) # accuracy_score(y_test, y_pred)

        y_val_enrich = y_val_enrich.values.reshape(y_val_enrich.shape[0], 1)
        y_pred = y_pred.reshape(y_pred.shape[0], 1)

        # Create the dataframe from numpy.ndarray
        X_val_enrich = val_enrich.drop('TARGET', axis=1) # X_test
        features = list(X_val_enrich.columns)

        X_val_enrich_df = pd.DataFrame(X_val_enrich, columns=features)
        y_val_enrich_df = pd.DataFrame(y_val_enrich, columns=['y_test'])
        y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

        # # Add y_test & y_pred columns to the X_test dataset
        test_df = X_val_enrich_df.join(y_val_enrich_df).join(y_pred_df)


        URL = r'../../data/final_test_df.csv'
        utils.write_data(URL, test_df)

        mlflow.log_metric("f1_score", f1score)
        mlflow.log_metric("accuracy", accuracy)

        # # Delete runs (DO NOT USE UNLESS CERTAIN)
        # for run_id in ALL_RUNS_ID :
        #     client.delete_run(run_id)

        # # Delete experiment (DO NOT USE UNLESS CERTAIN)
        # client.delete_experiment(EXPERIMENT_ID)
