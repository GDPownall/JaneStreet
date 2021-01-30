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

It contains an instance of a pytorch LSTM, where the output of the final timestep is interfaced to a linear neural network (note that this is different to most tutorials which use outputs from all timesteps considered). The activation functions of the linear network is the ReLU function. 

In the final step, it is interfaced to a sigmoid function for classification purposes. This approximates the 1 or 0 action we are looking for, and has the advantage that the model does not overfit noise in datapoints for which the action is clear. It was found that switching from regression of the response to classification of the action (with the custom loss function discussed below) improved the score by a factor of between two and three.

In the same script, there is a train function

```python
from Models.LSTM_NN import train
help(train)
```

This will use stochastic gradient descent to optimise the model to the training set of data. 

Loss functions are defined in the src/Utils/Loss.py file. Currently, just one loss function is used, which optimises the score which will be taken in the competition. This is important so that the model does not overfit to low-weight events. It also includes functionality for L1 regularisation of input parameters - at the moment, the parameters of the linear layers are regularised. In future, more loss functions with different methods of regularisation could be used.

### Data

The data is read into a class, rather than a simple dataframe, and the model takes things from that class. This makes for cleaner code.

```python
from Utils.load_data import Data
help(Data)
help(Data.from_csv)
```

The module itself is initiated from a pandas dataframe, however there is a class method from_csv which will load directly from a csv file.

## Notes on findings

### Handling the size of the data

This competition uses a relatively large amount of data for hobby programmers. This has required a few changes to the standard LSTM tutorials you might see on the internet.

Firstly, rather than converting the data into sequences immediately, the program converts to sequences on the fly. Converting to sequences immediately requires storage of the sequence length multiplied by the data, which gets enormous quickly. This takes longer to process, but is far more memory efficient.

NaN replacement in the kaggle competition reading is done through the numpy.where function, which was found to be faster than the pandas fillna function by two orders of magnitude. There is also functionality to transfer everything to a GPU.

Even with the memory efficiency, this can take up large amounts of space on a personal computer, so it is recommended to run through Google Colab or similar. The colab notebook being used is saved in the lead directory of this repository.

### Suggestions on Kaggle

There have been a number of suggestions on the Kaggle forums which have not been implemented. 

Firstly, significant improvement in the leaderboard score was found by using the data between 0.4 and 0.6 of the way through the training sample, and especially by removing data from the first 85 days. This has not been implemented as I believe this to be overtraining to the leaderboard test set.

Secondly, one popular suggestion is to one-hot encode feature_0 as is it takes two discrete values, ie. it is a binary feature. I believe this to be already one-hot encoded, and that doing this would only serve to create an additional feature which is directly anticorrelated to the first. Any improvements seen is surely overfitting.
