#!/usr/bin/env python

from Utils.load_data import Data
from Models.LSTM_NN import LSTM, train
import torch
import pandas as pd

## Training data
data = Data.from_csv(short=True) # Use path argument to state where data comes from
model = LSTM(data,n_lstm_layers=2,reg_first_layer_only=False)
train(model,log_file='log.csv')
model.my_save('model')
for param in list(model.parameters()):
    print(param.size())
del model

## Loading model and testing
## Dataframe is provided 

test_prediction = pd.read_csv('input/example_test.csv')
# convert to what is given on kaggle
test_prediction = test_prediction[['weight']+['feature_'+str(i) for i in range(130)]+['date']]
model = torch.load('model')
def predict():
    for i in range(len(test_prediction)):
        df = test_prediction.iloc[[i],:]
        print(df) 
        #print(df)
        y = model.kaggle_predict(df)
        #print(y)
    return

import timeit
print(timeit.timeit(stmt=predict, number=5))
