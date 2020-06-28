import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationNet(nn.Module):
    '''
    NN for the classification tasks
    Utilises ReLU as activation function
    Dropout can be adjusted
    '''
    def __init__(self, Layers):
        super(ClassificationNet, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.dropout(F.relu(linear_transform(activation)), p=0.5)
            else:
                activation = linear_transform(activation)
        return activation


class RegressionNet(nn.Module):
    '''
    NN for the regression task
    Utilises ReLU as activation function
    Dropout can be adjusted
    '''
    def __init__(self, Layers):
        super(RegressionNet, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.dropout(F.relu(linear_transform(activation)), p=0)
            else:
                activation = linear_transform(activation)
        return activation