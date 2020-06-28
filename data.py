import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


def createTrain(X_train, y_train, task):
    '''
    Given training features and labels, returns Dataset depending on the task
    '''
    X_train = torch.tensor(X_train).float()
    mean = torch.mean(X_train, dim=0)
    std = torch.std(X_train, dim=0)
    X_train = (X_train - mean) / std

    indices = []
    if len(torch.where(torch.isnan(X_train[0]))[0].numpy()) != 0:
        indices = torch.where(torch.isnan(X_train[0]))[0].numpy()
        for idx in indices:
            X_train[:, idx] = 0.0
    
    if (task == 'classification1') or (task == 'classification2'):
        y_train = torch.tensor(y_train).long()
    elif task == 'regression':
        y_train = torch.tensor(y_train).reshape(-1, 1).float()

    return Dataset(X_train, y_train), [mean, std, indices]


def createValTest(X_val, X_test, y_val, y_test, parameters, task):
    '''
    Given validation and test features, and  validation and test labels, returns Dataset depending on the task
    '''
    X_val = (torch.tensor(X_val).float() - parameters[0]) / parameters[1]
    X_test = (torch.tensor(X_test).float() - parameters[0]) / parameters[1]

    if len(parameters[2]) != 0:
        for idx in parameters[2]:
            X_val[:, idx] = 0.0
            X_test[:, idx] = 0.0
    
    if (task == 'classification1') or (task == 'classification2'):
        y_val = torch.tensor(y_val).long()
        y_test = torch.tensor(y_test).long()
    elif task == 'regression':
        y_val = torch.tensor(y_val).reshape(-1, 1).long()
        y_test = torch.tensor(y_test).reshape(-1, 1).float()

    return Dataset(X_val, y_val), Dataset(X_test, y_test)


def read_split(file, task):
    '''
    Given filename of data and task, returns the train, validation and test Datasets
    '''
    df = pd.read_csv(file)
    if task == 'classification1':
        X = df.iloc[:, 2:-2].values
        y = df.iloc[:, -2].values
    elif task == 'classification2':
        draftOrUndraft = np.zeros(shape=(df.shape[0], 1))
        for i, val in enumerate(df['DraftCategory'].values):
            if val == 4:
                draftOrUndraft[i] = 1
            else:
                draftOrUndraft[i] = 0
        df['DraftCategory'] = draftOrUndraft
        X = df.iloc[:, 2:-2].values
        y = df.iloc[:, -2].values
    elif task == 'regression':
        df = df[df['DraftCategory'] != 4]
        X = df.iloc[:, 2:-2].values
        y = df.iloc[:, -1].values
    else:
        return 0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    train_ds, parameters = createTrain(X_train, y_train, task)
    val_ds, test_ds = createValTest(X_val, X_test, y_val, y_test, parameters, task)
    return train_ds, val_ds, test_ds