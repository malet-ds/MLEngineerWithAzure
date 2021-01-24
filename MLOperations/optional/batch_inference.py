import io
import pickle
import argparse
import numpy as np
from azureml.core.model import Model

def init():
    global batch_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest="model_name", required=True)
    args = parser.parse_args()
    model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as model_file:
        batch_model = pickle.load(model_file)

def run(input_data):
    results_list = []
    for row in input_data:
        pred = batch_model.predict(row)
        if pred > 0.5:
            results_list.append("Risk")
        else:
            results_list.append("No Risk")
    return results_list
