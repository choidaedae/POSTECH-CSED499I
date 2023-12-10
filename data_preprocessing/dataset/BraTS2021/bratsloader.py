import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from PIL import Image
import nibabel as nib
import numpy as np
import os

# set directory


# 사용자 정의 데이터셋 클래스 정의
class Brats2DDataset(Dataset):
    def __init__(self, transform=None):
        
        self.path = '/root/daehyeonchoi/csed491/wavediffseg/data_process/dataset/BraTS2021/normalized_image_arrays'
        self.datas = os.listdir(self.path)
        self.image_dir = '/root/daehyeonchoi/csed491/wavediffseg/data_process/dataset/BraTS2021/normalized_image_arrays'
        self.label_dir = '/root/daehyeonchoi/csed491/wavediffseg/data_process/dataset/BraTS2021/label_arrays'
        self.transform = transform 
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        
        images = np.load(os.path.join(self.image_dir, self.datas[idx]))
        label = np.load(os.path.join(self.label_dir, self.datas[idx]))
        

        if self.transform:
            images = self.transform(images)
            label = self.transform(label)
            
        images = np.transpose(images, (1, 2, 0))
        
        return images, label


if  __name__== "__main__":
        
    brats = Brats2DDataset()
    sampler = RandomSampler(brats)
    
    len = brats.__len__()
    image, label = brats[0]
    
    print(len)
    print(image.shape)
    print(label.shape)
    
    batch_size = 32
    data_loader = DataLoader(brats, sampler = sampler, batch_size=batch_size)
