import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as tf
import matplotlib.pyplot as plt
import datetime

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(tf.relu(self.conv1(x)))
        x = self.pool(tf.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = tf.relu(self.fc1(x))
        x = tf.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
path = './final'
saved_model = GarmentClassifier()
saved_model.load_state_dict(t.load(path))