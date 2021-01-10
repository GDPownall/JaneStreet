#!/usr/bin/env python

from Utils.load_data import Data
from Models.LSTM_NN import LSTM, train
import torch
import pandas as pd

## Training data
data = Data.from_csv(short=True) # Use path argument to state where data comes from
model = LSTM(data,n_lstm_layers=2)
train(model,log_file='log.csv')
model.my_save('model')
del model

## Loading model and testing
## Dataframe is provided 

test_prediction = pd.read_csv('input/example_test.csv')
model = torch.load('model')
def predict():
    for i in range(len(test_prediction)):
        df = test_prediction.iloc[[i],1:]
        #print(df)
        y = model.kaggle_predict(df)
        #print(y)
    return

import timeit
print(timeit.timeit(stmt=predict, number=5))
