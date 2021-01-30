# JaneStreet
Jane street kaggle competition

## Getting started

This repository is set up to be run using python 3.7.

In a bash shell, to initialise the modules in this repository, execute

```bash
source setup.sh
```

Alternatively, in a python script, execute:

```python
import sys
sys.path.append('<repository_location>/src/')
```

The code is stored primarily in the src/ directory. 
Utils contains the utilities (load data, that sort of thing), Models contains the models.

## What is the JaneStreet Markets Kaggle challenge?

Given 130 unknown time-series features, a "response" and a weight for each event, optimise the sum of the response multiplied by the weight multiplied by your own input "action", which can be 1 or 0 on the submitted set. 

## Functionality

### Model

At the time of writing, there is just one model written: an LSTM written in torch stored in the src/Models/LSTM_NN.py file. To understand how to use it, try

```python
from Models.LSTM_NN import LSTM
help(LSTM)
```

It contains an instance of a pytorch LSTM, where the output of the final timestep is interfaced to a linear neural network. The activation functions of the linear network is the ReLU function. 

In the final step, it is interfaced to a sigmoid function for classification purposes. This approximates the 1 or 0 action we are looking for, and has the advantage that the model does not overfit noise in datapoints for which the action is clear. It was found that switching from regression of the response to classification of the action (with the custom loss function discussed below) improved the score by a factor of between two and three.

In the same script, there is a train function

```python
from Models.LSTM_NN import train
help(train)
```

This will use stochastic gradient descent to optimise the model to the training set of data. 

Loss functions are defined in the src/Utils/Loss.py file. Currently, just one loss function is used, which optimises the score which will be taken in the competition. This is important so that the model does not overfit to low-weight events. It also includes functionality for L1 regularisation of input parameters - at the moment, the parameters of the linear layers are regularised. In future, more loss functions with different methods of regularisation could be used.

### Data
