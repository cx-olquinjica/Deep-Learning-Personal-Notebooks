#! /Users/admin/miniconda3/envs/d2l/bin/python

import time
import numpy as np
import torch
from d2l import torch as d2l

# this class saves all arguments in a class's __init__ method as class attributes

class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented 

# the base class of all models

class ProgressBoard(d2l.HyperParameters):
    """The Board that plots data points in animation."""
    pass

class Module(nn.Module, d2l.HyperParameters):
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError
    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural Network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        pass 


    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

        l = self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError
    #! /Users/admin/miniconda3/envs/d2l/bin/python

### the code above constitute the basic implementation of the  Module class


class DataModule(d2l.HyperParameters):
    """The base class of data."""
        def validation_step(self, batch):
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)


### Here we have the basic implementation of the DataModule class

class Trainer(d2l.HyperParameters):
    """The base calss for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    from torch import nn
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model 

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optmizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            # I could add some print statement here with a counter to confirm how many times the fit_epoch() method is called

    self.plot('loss', l, train=False)
    def fit_epoch(self):
        raise NotImplementedError
    # this will method will be overwritten every time I am working with a new model
    def get_dataloader(self, train):
        pass






