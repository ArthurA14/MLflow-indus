import joblib
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow
import utils
import feature as ft


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


# HERE: put the path of the best model (found on MLflow ui)
logged_model = 'runs:/f67e89851aed4bc686c997b78507e4bc/model'

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)


def predict_(data: pd.DataFrame):
    """
    Helper to make predictions on data
    
    Args: 
        pd.DataFrame of given features to perform prediction on
    
    Returns:
        np.array with model predictions
    """

    # load necessary trained artefacts
    imputer = joblib.load('../data/imputer.save') 
    scaler = joblib.load('../data/std_scaler.save') 
    data = imputer.transform(data)
    data = scaler.transform(data)

    return loaded_model.predict(data)


if __name__ == "__main__" :
    with mlflow.start_run() :

        y_pred = predict_(ft.test_enrich) # y_pred = loaded_model.predict(X_test)
        y_pred = y_pred.reshape(y_pred.shape[0], 1)

        # Create the dataframe from numpy.ndarray
        test_enrich_df = pd.DataFrame(ft.test_enrich, columns=list(ft.test_enrich.columns))
        y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

        # Add y_pred column to the X_test dataset
        test_df = test_enrich_df.join(y_pred_df)


        PATH = r'..\\data\\final_test_df.csv'
        utils.write_data(PATH, test_df)



        # # Delete runs (DO NOT USE UNLESS CERTAIN)
        # for run_id in ALL_RUNS_ID :
        #     client.delete_run(run_id)

        # # Delete experiment (DO NOT USE UNLESS CERTAIN)
        # client.delete_experiment(EXPERIMENT_ID)
