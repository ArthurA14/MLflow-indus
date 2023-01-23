# mlflow_project

## Using MLflow to log and industrialize our ML project

MLflow consists of the following four main components:

- MLflow Tracking — facilitates the recording of experiments, including the tracking of used models, hyperparameters, and artifacts.
- MLflow Projects — allows teams to package data science code in a reproducible format.
- MLflow Models — allows teams to export and deploy machine learning models in various environments.
- MLflow Registry — enables model storage, versioning, staging, and annotation.

## Goals

The goal of this project is to apply some concepts & tools seen in the 3 sessions of this course, this
project is organized into 3 parts :

- Part 1 : Building Classical ML projects with respect to basic ML Coding best practices <br>
    ● Use GIT for team collaboration, code & model versioning <br>
    ● Separate your ML project workflow into different scripts (data preparation, feature
    engineering, models training, predict) <br>
    ● Use a template cookie cutter or adapt/define your own (Example :
    https://drivendata.github.io/cookiecutter-data-science/ ) <br>
    ● Use a conda environment for all your libraries (or any other package/environnement
    management like poetry) <br>
    ● Use a documentation library (Sphinx recommended) : despite several attempts, we couldn't make Sphinx work on this project.  
    
- Part 2 : Integrate MLFlow to your project <br>
    ● Install MLFlow in your python environment (don’t forget to add it to your lib requirements) <br>
    ● Track parameters & metrics of your model and display the results in your local mlflow UI
    (multiple runs) <br>
    ● Package your code in a reusable and reproducible model format with ML Flow projects <br>
    ● Deploy your model into a local REST server that will enable you to score predictions
    (Optional) : we build a REST API for this purpose.
    
- Part 3 : Integrate ML Interpretability to your project <br>
    ● Install SHAP in your python environment (don’t forget to add it to your lib requirements) <br>
    ● Use it to explain your model predictions : <br>
        - Build a TreeExplainer and compute Shaplay Values, <br>
        - Visualize explanations for a specific point of your data set, <br>
        - Visualize explanations for all points of your data set at once, <br>
        - Visualize a summary plot for each class on the whole dataset. <br>

## Package Usage

Intall the required packages:

```bash
pip install -r requirements.txt
```
#### Warning: This step is really important. By running this command, many requirements, which are located in the subdirectory *'opt/conda/'*, or in the directory *'tmp/'*, and in other directories related to the operating system used, are written in this file.
In this repository, we have simply put the main dependencies required to complete the project.

### Data preparation

#### Warning: it is necessary to download the application_train.csv and application_test.csv files on Kaggle and store them in a data/ folder in the root.

```bash
python3  data_prep.py
```

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

#### Warning : provide the model path in the script, you can retrieve it in MLflow UI.

![image](https://user-images.githubusercontent.com/57401552/209138374-4ed4009c-23a9-47dc-9a87-9c6bb4edef77.png)

```bash
python3 predict.py
```

### MLflow UI
Launch ***mlflow ui*** command in the **"../mlflow_project/files"** directory
```bash
cd files
mlflow ui
```

## API Usage

### Predict endpoint

```bash
python3 main.py
```

Then provide a csv to the endpoint URL (cf. code in notebook *test_api.ipynb*)


## SHAP Library

### Let's get some explanations from visualizations

* Let's generate and save the model with the *train.py* file (*joblib* inside)
* Let's load the model and get some visualizations on the test set (cf. code in notebook *shap_viz.ipynb*)
