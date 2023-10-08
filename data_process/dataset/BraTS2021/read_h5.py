import h5py
import numpy as np
import matplotlib.pyplot as plt 
import os

p = '/root/daehyeonchoi/brats/brats-unet/data_process/dataset/BraTS2021/dataset/BraTS2021_00000_mri_norm2.h5'

output_directory = '/root/daehyeonchoi/brats/brats-unet/data_process/dataset/BraTS2021/preprocessed/'

h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]

print('image shape:',image.shape,'\t','label shape',label.shape)
print('label set:',np.unique(label))

flair_array = image[0, :, :, :]
t1_array = image[1, :, :, :]
t1ce_array = image[2, :, :, :]
t2_array = image[3, :, :, :]
seg_array = label

print(np.max(t1_array))
print(np.min(t2_array))
print(np.max(seg_array))
print(np.min(seg_array))

