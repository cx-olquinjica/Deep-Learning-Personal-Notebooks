#! /Users/admin/miniconda3/envs/d2l/bin/python
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


# data preparation

training_data = datasets.FashionMNIST(
        root="data",
        train=True, 
        download=True, 
        transform=ToTensor()
)


test_data= datasets.FashionMNIST(
        root="data",
        train=False, 
        download=True, 
        transform=ToTensor()

)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# get device for training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# defing the class

class LeNet(nn.Module):
    """Defining LeNet."""
    def __ini__(self, lr=0.01, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2), 
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(), 
                nn.AvgPool2d(kernel_size=2, stide=2),
                nn.Flatten(), 
                nn.LazyLinear(120), nn.Sigmoid(), 
                nn.LazyLinear(84), nn.Sigmoid(), 
                nn.LazyLinear(num_classes))




