import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import nibabel as nib
import numpy as np
import os

# 이미지 파일들이 저장된 디렉토리 경로 설정
data_directory = './data_process/dataset/BraTS2021/data/'
type_list = ['flair','t1','t1ce','t2','seg']
img_id_list = []
train_path = '/root/daehyeonchoi/brats/brats-unet/data_process/dataset/BraTS2021/train.txt'
    
with open(train_path, 'r') as file:
    for line in file:
        img_id_list.append(line.strip()[10:15])  

'''
transforms.Compose([
        transforms.Resize((240, 240)),  # 이미지 크기 조절
        transforms.ToTensor(),  # 이미지를 Tensor로 변환
        transforms.Normalize()])
        
        '''

# 사용자 정의 데이터셋 클래스 정의
class Brats2DDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transform 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_id = img_id_list[idx]
        img_id_dir = f'BraTS2021_{img_id}'
        images = {}
        
        for type in type_list:
            
    
            file_name = f'BraTS2021_{img_id}_{type}.nii.gz'
            file_path = os.path.join(self.data_dir, img_id_dir)
            file_path = os.path.join(file_path, file_name)
            image=nib.load(file_path).get_fdata()
            title=file_path.rsplit("_",1)[1].split(".",1)[0]
            
            if title == 'flair':
                images['flair'] = image # (240, 240, 155)
            elif title == 't1':
                images['t1'] = image # (240, 240, 155)
            elif title == 't1ce':
                images['t1ce'] = image # (240, 240, 155)
            elif title == 't2':
                images['t2'] = image # (240, 240, 155)
            elif title == 'seg':
                label = image # (240, 240, 155)

        if self.transform:
            images = self.transform(image)
        
        return images, label 


if  __name__== "__main__":
        
    brats = Brats2DDataset(data_directory)
    
    image, label = brats[0]
    
    print(image['flair'].shape)
    

    # DataLoader를 사용하여 데이터 로딩
    batch_size = 32
    data_loader = DataLoader(brats, batch_size=batch_size, shuffle=True)

    # DataLoader를 통해 데이터를 반복(iterate)할 수 있습니다.
    
    '''
    for batch in data_loader:
        # 이 부분에 원하는 작업을 수행하십시오.
        # batch는 이미지 데이터의 배치입니다. (batch_size, channels, height, width)
        print("Batch shape:", batch.shape)
        
        '''