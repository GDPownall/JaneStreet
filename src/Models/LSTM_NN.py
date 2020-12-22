#!/usr/bin/env python

from Utils.load_data import Data 
from torch import nn
import torch

data = Data()

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer_size = 100
        self.lstm = nn.LSTM(1, self.hidden_layer_size)
        
