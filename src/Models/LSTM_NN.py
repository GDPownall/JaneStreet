#!/usr/bin/env python

from Utils.load_data import Data 
from torch import nn
import torch

data = Data()

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        ## Sizes of various things
        self.hidden_layer_size = 100
        n_features = data.n_features()
        output_size = 1

        ## layers
        self.lstm = nn.LSTM(6, self.hidden_layer_size)
        self.linear = nn.Linear(self.hidden_layer_size, output_size)
        self.reset_hidden_cell()

    def reset_hidden_cell(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        ## Define function to calculate predictions

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq),-1))
        return predictions[-1]


def train():
    model = LSTM()
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)
    
    epochs = 10
    train_x, train_y, test_x, test_y = data.train_test()

    for i in range(epochs):
        for x,y in zip(train_x, train_y):
            optimiser.zero_grad()
            model.reset_hidden_cell()
            y_pred = model(torch.FloatTensor(x))

            single_loss = loss_function(y_pred, torch.FloatTensor([y]))
            single_loss.backward()
            optimiser.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

train()
