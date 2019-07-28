from flask import Blueprint,jsonify,request
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json
import Consumerpreprocessing

model_predictor= Blueprint('model_predictor',__name__)

@model_predictor.route('/predict', methods=["POST"])
def predict_value():
    json_file = open('Consumermodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Consumermodel.h5")
    print("Loaded model from disk")

    print(request.is_json)
    content = request.get_json()
    print(content['input'])

    print("post")
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model)
    predstring = content["input"]
    datas = list()
    datas.append(predstring)
    pred = model.predict(Consumerpreprocessing.preprocessing(datas))
    print("PREDICTIONS :")
    print(pred)
    return jsonify(
        {
            "review" : str(pred[0])
        }
    )

