import h5py
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2

p = '/root/daehyeonchoi/brats/brats-unet/data_process/dataset/BraTS2021/dataset/BraTS2021_00000_mri_norm2.h5'
input_directory = '/root/daehyeonchoi/brats/brats-unet/data_process/dataset/BraTS2021/dataset/'
file_names = os.listdir(input_directory)
file_names = [file for file in file_names if os.path.isfile(os.path.join(input_directory, file))]
output_directory = '/root/daehyeonchoi/brats/brats-unet/data_process/dataset/BraTS2021/preprocessed/'

h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]

for t in range(80, 81):

    image_t = image[:, :, :, t]

    flair_array = image_t[0, :, :]
    print(flair_array.shape)
    t1_array = image_t[1, :, :]
    t1ce_array = image_t[2, :, :]
    t2_array = image_t[3, :, :]
    seg_array = label

    if flair_array.max() - flair_array.min() != 0: 
        flair_array = ((flair_array - flair_array.min()) / (flair_array.max() - flair_array.min())*255).astype(np.uint8)

    else: flair_array = flair_array.astype(np.uint8)

    if t1_array.max() - t1_array.min() != 0: 
        t1_array = ((t1_array - t1_array.min()) / (t1_array.max() - t1_array.min())*255).astype(np.uint8)

    else: flair_array = flair_array.astype(np.uint8)

    if t1ce_array.max() - t1ce_array.min() != 0: 
        t1ce_array = ((t1ce_array - t1ce_array.min()) / (t1ce_array.max() - t1ce_array.min())*255).astype(np.uint8)

    else: t1ce_array = t1ce_array.astype(np.uint8)

    if t2_array.max() - t2_array.min() != 0: 
        t2_array = ((t2_array - t2_array.min()) / (t2_array.max() - t2_array.min())*255).astype(np.uint8)

    else: t2_array = t2_array.astype(np.uint8)
    

    image_list = [flair_array, t1_array, t1ce_array, t2_array]
    
    image_name_list = ["flair.jpg", "t1.jpg", "t1ce.jpg", "t2.jpg"]
    
    for i in range(4):
        output_path = os.path.join(output_directory, image_name_list[i])
        cv2.imwrite(output_path, image_list[i])





