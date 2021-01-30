#!/usr/bin/env python

from Utils.load_data import Data, split_sequences
from Utils.Loss import custom_loss
from torch import nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, data = None, hidden_layer_size = 100, linear_sizes = [30,15], n_lstm_layers = 1, seq_len = 5, dropout = 0.2, reg_first_layer_only = False):
        '''
        Class for the LSTM model of this analysis. Overloads the pyTorch nn.Module class.

        data:                   input data (must be an instance of the Data class in Utils/load_data.py)
        hidden_layer_size:      size of the hidden layer in the lstm instance.
        linear_sizes:           variable-length list of sizes of linear layers.
        n_lstm_layers:          number of lstm layers (stacked). Second layer and beyond take size hidden_layer_size.
        seq_len:                sequence length used for training and evaluation.
        dropout:                dropout for lstm layers (between 0 and 1 as usual).
        reg_first_layer_only:   boolean. If true, regularise only the parameters of the first linear layer.
        '''
        super().__init__()
        ## Sizes of various things
        self.data = data
        self.ffill = self.data.ffill
        self.hidden_layer_size = hidden_layer_size # number of hidden states
        n_features = data.n_features()
        output_size = 1
        self.n_layers = n_lstm_layers
        self.seq_len = seq_len
        self.dropout = dropout
        self.linear_sizes = linear_sizes
        self.reg_first_layer_only = reg_first_layer_only
        
        linear = OrderedDict()
        ## layers
        self.lstm = nn.LSTM(
                input_size = n_features,
                hidden_size = self.hidden_layer_size,
                num_layers = self.n_layers,
                dropout=dropout,
                batch_first = True)
        self.dropout       = nn.Dropout(dropout)
        
        layer_sizes = [hidden_layer_size] + linear_sizes
        for l in range(len(layer_sizes)-1):
            linear['linear'+str(l)] = nn.Linear(layer_sizes[l], layer_sizes[l+1])
            linear['relu'+str(l)]   = nn.ReLU()
        linear['output']            = nn.Linear(layer_sizes[-1], 1, bias=False)
        self.linear = nn.Sequential(linear)
        #self.reset_hidden_cell()

    def reset_hidden_cell(self, batch_size):
        '''
        Reset the hidden cell of the lstm layers.
        '''
        self.hidden_cell = (torch.zeros(self.n_layers,batch_size,self.hidden_layer_size).to(device),
                            torch.zeros(self.n_layers,batch_size,self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        ## Define function to calculate predictions
        batch_size, seq_len, _ = input_seq.size()

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        x = lstm_out.contiguous()
        #x = x.view(batch_size,-1)
        x = x[:,-1,:] # This one takes only output from last lstm layer
        x = self.dropout(x)
        x = self.linear(x)
        prob = torch.sigmoid(x)
        return prob 

    def get_linear_params(self):
        '''
        Get the parameters of the linear layers of the model for regularisation purposes.
        '''
        if not self.reg_first_layer_only:
            params = torch.reshape(list(self.parameters())[-1], (-1,))
            for i in range(len(self.linear_sizes)):
                params = torch.cat(
                        (
                            params,
                            torch.reshape(list(self.parameters())[-1-2*(i+1)], (-1,))
                        )
                )
        else:
            params = torch.reshape(list(self.parameters())[-1-2*(len(self.linear_sizes))], (-1,))

        return params

    def add_zeros_to_data(self):
        to_add = np.zeros((self.seq_len-1, self.data.train_x[0].size))
        self.data.train_x = np.vstack((to_add,self.data.train_x))
        self.data.test_x  = np.vstack((to_add,self.data.test_x))

    def my_save(self,path):
        '''
        Function to save the model. 
        Deletes the data (so that it isn't saved to an extra file) and stores whatever else is needed for evaluation, for example what to replace nans with.
        '''
        self.data_nans      = self.data.nans
        self.data_nans_np   = np.array([self.data_nans[key] for key in self.data_nans.keys()])
        self.data           = None
        self.predict_from   = np.zeros((self.seq_len, len(self.data_nans.keys()))) 
        torch.save(self, path)

    def kaggle_predict(self, row):
        '''
        Make a prediction given a single of a dataframe, as is required by Kaggle.
        Stores this row and previous rows up to the sequence length, deletes rows going further back.
        Fills the nans as previously defined.
        '''
        row_vals = row.values[:,1:-1]
        if self.ffill:
            row_vals = np.where(np.isnan(row_vals), self.predict_from[-1,:], row_vals)
        elif np.isnan(row_vals.sum()):
            #print('================')
            #print(row_vals)
            #print(self.data_nans_np)
            row_vals = np.where(np.isnan(row_vals), self.data_nans_np , row_vals)
            #print(row_vals)
        self.predict_from = np.vstack((self.predict_from,row_vals))[1:,:]
        x = split_sequences(self.predict_from, self.seq_len)
        self.reset_hidden_cell(torch.FloatTensor(x).size(0))
        x_tens = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x_tens = x_tens.cuda() 

        y_pred = self(x_tens)[-1]
        return y_pred

def train(model, lr=0.0001, epochs=10, batch_size=300, log_file=None, reg_lambda = 0.01, lin_reg_lambda = 0.01):

    model.train()
    loss_function = custom_loss#nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_lambda)

    print(model)

    ## Add on rows of zeros at start
    model.add_zeros_to_data()

    if np.isnan(model.data.train_x).any():
        print('nan found in features')
        print(np.argwhere(np.isnan(model.data.train_x)))
        raise ValueError('nan detected in features')

    for i in range(epochs):
        epoch_loss = 0.
        test_loss  = 0.
        for b in range(0,len(model.data.train_x),batch_size):
            #Set up x and y
            x = split_sequences(model.data.train_x,model.seq_len,[b,b+batch_size])
            y = model.data.train_y[b:b+batch_size]
            weight = model.data.train_weight[b:b+batch_size]
            if np.isnan(x).any(): raise ValueError('nan detected in features')
            #Set up model
            optimiser.zero_grad()
            x_tensor = torch.FloatTensor(x).to(device)
            model.reset_hidden_cell(x_tensor.size(0))
            #Produce prediction
            y_pred = model(x_tensor)
            #single_loss = loss_function(y_pred, torch.FloatTensor([y]).T)
            regularise_params = model.get_linear_params()             
            single_loss = loss_function(y_pred, 
                    torch.FloatTensor([y]).to(device).T, 
                    torch.FloatTensor([weight]).to(device).T,
                    lin_reg_lambda*regularise_params)
            single_loss.backward()
            optimiser.step()
            for param in model.parameters():
                if torch.isnan(param.data).any(): raise ValueError('nan weight detected epoch '+str(i)+' and batch '+str(b/len(model.data.train_x)))
            epoch_loss += single_loss.item()
        print(f'epoch: {i:3} loss: {epoch_loss:10.10f}')
        if model.data.train_full: continue
        # Calculate loss for test set
        # Use same batch size for memory efficiency
        model.eval()
        for b in range(0,len(model.data.test_x),batch_size):
            x_test = split_sequences(model.data.test_x,model.seq_len,[b,b+batch_size])
            y_test = model.data.test_y[b:b+batch_size]
            test_weight = model.data.test_weight[b:b+batch_size]
            model.reset_hidden_cell(torch.FloatTensor(x_test).size(0))
            y_test_pred = model(torch.FloatTensor(x_test).to(device))
            test_loss += loss_function(y_test_pred, 
                    torch.FloatTensor([y_test]).to(device).T, 
                    torch.FloatTensor([test_weight]).to(device).T).item()
        print(f'epoch: {i:3} test loss: {test_loss:10.10f}')
        model.train()
        if log_file != None:
            if not os.path.isfile(log_file):
                with open(log_file,'a') as ifile:
                    ifile.write('epoch,lr,batch_size,hidden_layer_size,n_layers,seq_len,epoch_loss,test_loss\n')
            log = ','.join([str(a) for a in [
                i,
                lr,
                batch_size,
                model.hidden_layer_size,
                model.n_layers, 
                model.seq_len,
                epoch_loss,
                test_loss
                ]])
            log += '\n'
            with open(log_file,'a') as ifile:
                ifile.write(log)
        #for param in model.parameters():
        #    print(param.data)

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
