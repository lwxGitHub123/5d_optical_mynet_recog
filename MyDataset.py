from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch


# -----------------ready the dataset--------------------------
def default_loader(path):

    return Image.open(path).convert('RGB')
    # return Image.open(path).convert('L')

class MyDataset(Dataset):
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
        if label == 50 :
            label = 0
        elif label == 70 :
            label = 1

        fn = self.root + fn
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)




