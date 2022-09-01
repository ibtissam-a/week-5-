from flask import Flask, render_template, request
import pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patch
import tensorflow
app = Flask(__name__)
model = pickle.load(open('Titanic_Model (1).pkl', 'rb'))


@app.route('/')
def home():

    return render_template('tita.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_feaures = [int(x) for x in request.form.values()]
    final_features = [np.array(int_feaures)]
    prediction_t = model.predict(final_features)
    p = round(prediction_t[0], 2)

    return render_template('tita.html', prediction = p )


if __name__ == "__main__":
    app.run(port=8000, debug=True)

