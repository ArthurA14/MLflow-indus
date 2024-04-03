from flask import Flask, request 
import json, io, joblib
import pandas as pd
import data_prep as dp
import utils
from predict import predict_

app = Flask(__name__)
TRAIN_DF_PATH = r'..\\data\\application_train.csv'

try :
    # get data
    train = utils.get_data(TRAIN_DF_PATH) 
except Exception as e :
    dp.logger.exception("Unable to download training & test CSV. Error: %s", e)


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
    _, _, df = dp.data_prep(train, df)

    # get preds
    preds = predict_(df)
    preds = preds.tolist()
    json_str = json.dumps(preds)

    return json_str

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5004')
