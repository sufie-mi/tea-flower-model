import os
import pandas as pd
import torch
import numpy as np
# -*- coding: utf-8 -*-
# pytorch mlp for multiclass classification
from torch import Tensor
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Softmax, Module, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_

# Read data line by line, make predictions with the model, and write to a new column

class MLP(Module):  # Specify Module n_inputs when instantiating MLP
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 16)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(16, 14)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(14, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = ReLU()
        # self.act3 = Softmax(dim=1) # No softmax here as CrossEntropyLoss expects raw logits
        # 4
        self.hidden4 = Linear(10, 14)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act4 = ReLU()
        # 5
        self.hidden5 = Linear(14, 6)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # 6
        self.hidden6 = Linear(6, 5)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act6 = ReLU()
        # 7
        self.hidden7 = Linear(5, 5)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act7 = Softmax(dim=1)
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        # X = self.act3(X) Softmax is removed because CrossEntropyLoss handles logits and softmax calculation internally
        X = self.hidden4(X)
        X = self.act4(X)
        # 5th layer
        X = self.hidden5(X)
        X = self.act5(X)
        # 6
        X = self.hidden6(X)
        X = self.act6(X)
        # 7
        X = self.hidden7(X)
        X = self.act7(X)
        return X

def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

def predictions(input_file_path, output_file_path, model):
    df = pd.read_csv(input_file_path)
    # Iterate through each row to make predictions
    for index, row in df.iterrows():
        input_data = [row['buds'], row['flowers'], row['old flower'], row['time']]
        print(input_data)
        m = predict(input_data, model)
        print(m)

def add_predictions_to_csv(input_file_path, output_file_path, model):
    df = pd.read_csv(input_file_path)

    # Create a new column to store prediction results
    df['predict_stages'] = None

    # Iterate through each row to make predictions
    for index, row in df.iterrows():
        input_data = [row['buds'], row['flowers'], row['old flower'], row['time']]
        prediction = predict(input_data, model)
        # df.at[index, 'predict_stages'] = prediction[0]  # Assuming the return is an array
        # Get the index of the maximum value to write into csv
        max_index = np.argmax(prediction[0])  # prediction[0] is the prediction result
        df.at[index, 'predict_stages'] = max_index
    
    # Write results to a new CSV file
    df.to_csv(output_file_path, index=False)



input_file_path= r"flower_stage_data_raw.csv"
output_file_path= r"flower_stage_data_predict.csv"

model = MLP(4)
# load model weights
model.load_state_dict(torch.load(r"model_weights.pth"))
add_predictions_to_csv(input_file_path, output_file_path, model)