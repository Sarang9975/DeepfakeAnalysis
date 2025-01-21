import sys
import os
#!flask/bin/python
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import pickle
from flask_cors import CORS
#import python module
# from src.cleaning import process_text
from src.prediction import get_predictions

# Add the src folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from cleaning import *  # Or specific imports as needed

app = Flask(__name__)
Bootstrap(app)

CORS(app)

# define the path for model and feature transformer
model_path="models/lr_final_model.pkl"
transformer_path="models/transformer.pkl"
# load the model and feature transformer with pickle
loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
        
@app.route('/predict', methods=['POST'])
def predict():
    """
    Collect the input and predict the outcome
    Returns:
        Results.html with prediction
    """
    if request.method == 'POST':
        # get input statement
        namequery = request.form['namequery']
        data = [namequery]
        # get the clean data
        clean_data = process_text(str(data))
        test_features = loaded_transformer.transform([" ".join(clean_data)])
        my_prediction = get_predictions(loaded_model,test_features)
    return render_template('results.html', prediction=my_prediction, name = namequery)

if __name__ == '__main__':
    app.run(debug=True)