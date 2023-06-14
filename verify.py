import model as m
import torch as t
import numpy as np
import torchvision as tv
import load_single_image as lsi
import matplotlib.pyplot as plt

#to show image
def imshow(sample_element, shape = (64, 64)):
    plt.imshow(sample_element[0].numpy().reshape(shape), cmap='gray')
    plt.show()

#transformation for images
transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
        tv.transforms.Resize(size=(64,64)),
        tv.transforms.Grayscale(1)])

#load model for evaluation
custom_model = m.MPL()
custom_model.load_state_dict(t.load('.\custom_model'))
print('Model loaded ')
custom_model.eval()

#load image(s) for prediction
single_image = lsi.LoadImages(main_dir='.\Single', transform=transform)
load_single_image = t.utils.data.DataLoader(single_image, shuffle = False)
# image = next(iter(load_single_image))
print('Image(s) loaded ')

#load image into model and get output as prediction
# outputs = custom_model(image)

#load original dataset to get classes
dataset_training = tv.datasets.ImageFolder(root='.\Pictures', transform=transform)
classes = dataset_training.classes
print('Classes loaded ')

count = 0
for image in load_single_image:
    print(single_image.total_images[count])
    count += 1
    outputs = custom_model(image)
    _, predicted = t.max(outputs, 1)
    #range(1) for two classes
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))