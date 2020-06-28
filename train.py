import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import data
import model


def train(mdl, criterion, train_loader, train_ds, val_ds, metric, optimizer, epochs=100):
    '''
    Trains the model and records train loss and validation loss at end of each epoch
    Also records the train accuracy/MAE and validation accuracy/MAE at end of each epoch
    '''
    train_loss = []
    val_loss = []
    train_metric = []
    val_metric = []
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = mdl(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
        train_loss.append(criterion(mdl(train_ds.x), train_ds.y).item())
        val_loss.append(criterion(mdl(val_ds.x), val_ds.y).item())
        train_metric.append(metric(mdl, train_ds))
        val_metric.append(metric(mdl, val_ds))
    
    return train_loss, val_loss, train_metric, val_metric