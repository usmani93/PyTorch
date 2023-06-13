import os
import torch as t
import torchvision as tv
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class LoadImages(Dataset):
    def __init__(self, main_dir, transform):
        #set the loading directory
        self.main_dir = main_dir
        self.transform = transform

        #list all images in folder and count them
        all_imgs = os.listdir(main_dir)
        print(main_dir)
        self.total_images = natsorted(all_imgs)
        print(self.total_images)

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
# transform = tv.transforms.Compose(
#     [tv.transforms.ToTensor(),
#         tv.transforms.Normalize((0.5,), (0.5,)),
#         tv.transforms.Grayscale(1)])
# dataset_training = tv.datasets.ImageFolder(root='.\Pictures', transform=transform)
# dataloader_training = t.utils.data.DataLoader(dataset_training, shuffle = True)

# for i, j in enumerate(dataloader_training):
#     print(j)
