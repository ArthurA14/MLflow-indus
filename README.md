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
```bash
python3 predict.py
```
