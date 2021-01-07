#!/usr/bin/env python

'''
Contains custom loss functions
'''
import torch

def custom_loss(action, resp, weight):
    loss = torch.sum(-1*action*resp*weight)
    return loss
