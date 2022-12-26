# mlflow_project

Using MLflow to log and industrialize our ML project

MLflow consists of the following four main components:

- MLflow Tracking — facilitates the recording of experiments, including the tracking of used models, hyperparameters, and artifacts.
- MLflow Projects — allows teams to package data science code in a reproducible format.
- MLflow Models — allows teams to export and deploy machine learning models in various environments.
- MLflow Registry — enables model storage, versioning, staging, and annotation.

## Package Usage

### Training 

* lbgm model: 
```bash
python3 train.py lgbm
```
* logistic model: 
```bash
python3 train.py logistic
```

### Predicting on test data

#### Warning : provide the model path in the script

![image](https://user-images.githubusercontent.com/57401552/209138374-4ed4009c-23a9-47dc-9a87-9c6bb4edef77.png)

```bash
python3 predict.py
```

## API Usage

### Predict endpoint

```bash
python3 main.py
```

Then provide a csv to the endpoint URL (CF code in notebook test_api)
