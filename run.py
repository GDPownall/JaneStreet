#!/usr/bin/env python

from Utils.load_data import Data
from Models.LSTM_NN import LSTM, train
import torch

data = Data(short=True) # Use path argument to state where data comes from
model = LSTM(data)
train(model)
torch.save(model,'model')
