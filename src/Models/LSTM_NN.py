#!/usr/bin/env python

from Utils.load_data import Data, split_sequences
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

class LSTM(nn.Module):
    def __init__(self, data = None):
        super().__init__()
        ## Sizes of various things
        self.data = data
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


def train(model):
    model.train()
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)

    ## Add on rows of zeros at start
    to_add = np.zeros((model.seq_len-1, model.data.train_x[0].size))
    model.data.train_x = np.vstack((to_add,model.data.train_x))

    epochs = 10
    batch_size = 30

    for i in range(epochs):
        for b in range(0,len(model.data.train_x),batch_size):
            print(b/len(model.data.train_x))
            x = split_sequences(model.data.train_x,model.seq_len,[b,b+batch_size])
            y = model.data.train_y[b:b+batch_size]
            optimiser.zero_grad()
            model.reset_hidden_cell(torch.FloatTensor(x).size(0))
            y_pred = model(torch.FloatTensor(x))
            
            single_loss = loss_function(y_pred, torch.FloatTensor([y]).T)
            single_loss.backward()
            optimiser.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

def plot_predictions(model):
    train_x, train_y, test_x, test_y = model.data.train_test()
    y_pred = []
    model.reset_hidden_cell(torch.FloatTensor(test_x).size(0))
    y_pred = model(torch.FloatTensor(test_x))
    y_full = np.concatenate([train_y,test_y])
    horizontal = range(len(y_full))
    horizontal_pred = range(len(train_y),len(y_full))

    plt.plot(horizontal,y_full)
    print(horizontal_pred)
    print(y_pred)
    plt.plot(horizontal_pred, y_pred.detach().numpy())
    plt.show()

if __name__ == '__main__':
    data = Data()
    model = LSTM(data)
    train(model)
    plot_predictions(model)
