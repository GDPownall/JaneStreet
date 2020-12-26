#!/usr/bin/env python

from Utils.load_data import Data 
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

data = Data()

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        ## Sizes of various things
        self.hidden_layer_size = 100 # number of hidden states
        n_features = data.n_features()
        output_size = 1
        self.n_layers = 1
        self.seq_len = 5

        ## layers
        self.lstm = nn.LSTM(
                input_size = n_features,
                hidden_size = self.hidden_layer_size,
                num_layers = self.n_layers,
                batch_first = True)
        self.linear = nn.Linear(self.hidden_layer_size*self.seq_len, output_size)
        #self.reset_hidden_cell()

    def reset_hidden_cell(self, batch_size):
        self.hidden_cell = (torch.zeros(self.n_layers,batch_size,self.hidden_layer_size),
                            torch.zeros(self.n_layers,batch_size,self.hidden_layer_size))

    def forward(self, input_seq):
        ## Define function to calculate predictions
        batch_size, seq_len, _ = input_seq.size()

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.linear(x)


def train():
    model = LSTM()
    model.train()
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)
    
    epochs = 10
    batch_size = 10
    train_x, train_y, test_x, test_y = data.train_test()

    for i in range(epochs):
        for b in range(0,len(train_x),batch_size):
            x = train_x[b:b+batch_size,:,:]
            y = train_y[b:b+batch_size]
            optimiser.zero_grad()
            model.reset_hidden_cell(torch.FloatTensor(x).size(0))
            y_pred = model(torch.FloatTensor(x))

            single_loss = loss_function(y_pred, torch.FloatTensor([y]))
            single_loss.backward()
            optimiser.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    return model

def plot_predictions(model):
    train_x, train_y, test_x, test_y = data.train_test()
    y_pred = []
    for x in train_x:
        print(x)
        model.reset_hidden_cell()
        y_pred.append(model(torch.FloatTensor(x)))
    y_full = np.concatenate([train_y,test_y])
    horizontal = range(len(y_full))
    horizontal_pred = range(len(train_y),len(y_full))

    plt.plot(horizontal,y_full)
    print(horizontal_pred)
    print(y_pred)
    plt.plot(horizontal_pred, y_pred)
    plt.show()

model = train()
#plot_predictions(model)
