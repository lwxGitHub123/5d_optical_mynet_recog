from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2


# -----------------ready the dataset--------------------------
def default_loader(path):
    # img = cv2.imread(path)

    img = Image.open(path).convert('RGB')
    # print("  img.size =  ")
    # print(img.size)

    return img
    #return Image.open(path).convert('RGB')
    # return Image.open(path).convert('L')

class MyDatasetCuda(Dataset):
    def __init__(self,root, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.root = root


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        if label == 80 :
            label = 0
        elif label == 120 :
            label = 1

        fn = self.root + fn
        img = self.loader(fn)
        width = 224
        height = 224
        img = img.resize((width, height), Image.ANTIALIAS)  # resize image with high-quality
        if self.transform is not None:
            img = self.transform(img)

        # img = np.array(img)
        # img = img.squeeze()

        return img,label

    def __len__(self):
        return len(self.imgs)




