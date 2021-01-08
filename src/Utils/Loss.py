#!/usr/bin/env python

'''
Contains custom loss functions
'''
import torch

def custom_loss(action, resp, weight, regularise=None):
    loss = torch.sum(-1*action*resp*weight)
    if regularise != None:
        loss += torch.sum(regularise)
    #if regularise != None: print(torch.sum(-1*action*resp*weight),torch.sum(torch.abs(regularise)))
    return loss
