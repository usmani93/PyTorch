import model as m
import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as tf
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
import numpy as np

    
custom_model = m.MPL()
custom_model.load_state_dict(t.load('.\custom_model'))
custom_model.eval()
