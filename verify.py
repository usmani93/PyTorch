import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as tf
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
import numpy as np

batch_size = 10

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

#transform all images to same size
transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))])

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

validation_set = tv.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
validation_loader = t.utils.data.DataLoader(validation_set, batch_size, shuffle=False)

# get some random training images
dataiter = iter(validation_loader)
images, labels = next(dataiter)

# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
# show images
imshow(tv.utils.make_grid(images))

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
saved_model.eval()

outputs = saved_model(images)

_, predicted = t.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))