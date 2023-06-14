import torch as t

class MPL(t.nn.Module):
    def __init__(self, num_classes=10):
        super(MPL, self).__init__()
        self.fc1 = t.nn.Linear(64*64*1, 100)
        self.fc2 = t.nn.Linear(100, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 64*64*1)
        x = self.fc1(x)
        x = t.relu(x)
        x = self.fc2(x)
        return x