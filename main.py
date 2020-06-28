import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

import data
import model
import train

torch.manual_seed(1)

def accuracy(mdl, ds):
    '''
    Metric used for the classification tasks
    '''
    _, yhat = torch.max(mdl(ds.x), 1)
    return (yhat == ds.y).numpy().mean()


def mae(mdl, ds):
    '''
    Metric used for the regression task
    '''
    total = 0
    for i in range(len(ds.x)):
        total += F.l1_loss(mdl(ds.x[i]), ds.y[i]).item()
    return total / len(ds.x)


def classification1():
    '''
    For the first classification task
    Layers, batch_size and epochs can be adjusted
    '''
    train_ds, val_ds, test_ds = data.read_split('data/secondMost.csv', 'classification1')
    train_loader = DataLoader(dataset=train_ds, batch_size=256, shuffle=True)
    mdl = model.ClassificationNet([75, 100, 100, 50, 50, 5])
    optimizer = torch.optim.Adam(mdl.parameters())
    criterion = nn.CrossEntropyLoss()
    train_loss, val_loss, train_metric, val_metric = \
        train.train(mdl, criterion, train_loader, train_ds, val_ds, accuracy, optimizer, epochs=100)
    print(accuracy(mdl, train_ds))
    print(accuracy(mdl, val_ds))
    print(accuracy(mdl, test_ds))


def classification2():
    '''
    For the second classification task
    Layers, batch_size and epochs can be adjusted
    '''
    train_ds, val_ds, test_ds = data.read_split('data/secondMost.csv', 'classification2')
    train_loader = DataLoader(dataset=train_ds, batch_size=256, shuffle=True)
    mdl = model.ClassificationNet([75, 100, 100, 50, 50, 2])
    optimizer = torch.optim.Adam(mdl.parameters())
    criterion = nn.CrossEntropyLoss()
    train_loss, val_loss, train_metric, val_metric = \
        train.train(mdl, criterion, train_loader, train_ds, val_ds, accuracy, optimizer, epochs=100)
    print(accuracy(mdl, train_ds))
    print(accuracy(mdl, val_ds))
    print(accuracy(mdl, test_ds))


def regression():
    '''
    For the regression task
    Layers, batch_size and epochs can be adjusted
    '''
    train_ds, val_ds, test_ds = data.read_split('data/secondMost.csv', 'regression')
    train_loader = DataLoader(dataset=train_ds, batch_size=256, shuffle=True)
    mdl = model.RegressionNet([75, 100, 100, 100, 100, 100, 50, 1])
    optimizer = torch.optim.Adam(mdl.parameters())
    criterion = nn.SmoothL1Loss()
    train_loss, val_loss, train_metric, val_metric = \
        train.train(mdl, criterion, train_loader, train_ds, val_ds, mae, optimizer, epochs=100)
    print(mae(mdl, train_ds))
    print(mae(mdl, val_ds))
    print(mae(mdl, test_ds))


def main(args):
    if args.task == 'classification1':
        classification1()
    elif args.task == 'classification2':
        classification2()
    elif args.task == 'regression':
        regression()
    else:
        print('Invalid task.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    args = parser.parse_args()
    main(args)