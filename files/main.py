from flask import Flask, request 
from predict import predict
import json, io, joblib
import pandas as pd
from data_prep import data_prep, logger
import utils


app = Flask(__name__)
TRAIN_URL = r'../data/application_train.csv'

try :
    # get data
    train = utils.get_data(TRAIN_URL) 
except Exception as e :
    logger.exception("Unable to download training & test CSV. Error: %s", e)


@app.route('/api/', methods=['POST'])
def get_preds():

    f = request.files['file'] 
    if f and f.filename.rsplit('.', 1)[1].lower() == 'csv':
        
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)    
        stream.seek(0)
        # result = stream.read()
        # io.StringIO(result)
        df = pd.read_csv(stream)


    # preprocessing pipeline
    _, _, df = data_prep(train, df)

    
    # get preds
    preds = predict(df)
    preds = preds.tolist()
    json_str = json.dumps(preds)

    return json_str


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5004')