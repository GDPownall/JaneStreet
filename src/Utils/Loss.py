#!/usr/bin/env python

'''
Contains custom loss functions
'''
import torch

def custom_loss(action, resp, weight):
    rounded = torch.round(action)
    loss = torch.sum(-1*rounded*resp*weight)
    return loss
