# wavelet-diffusion-segmentation

Brain Tumor Segmentation via Noised Multi-level Wavelet Feature Extraction.
- POSTECH CSED499A - Research Project 1 
- in MIV Lab @ POSTECH (Under the supervision by Wonhwa Kim, Hyuna Cho)
- **(As of 12/15)** All codes and docs are completed version.

  
### Summary
- Implemented ‘WaveUNet’, that is able to extract useful features in image, with multi-level wavelet transform and various diffusion noise scales
- Proposed ‘Symmetric Contrastive Loss’, simple but strong logic
- Based on AttentionUNet, added a symmetric contrastive loss and had a better performance for **tumor cores (NCR, ET)**.
- Feature extractor’s feature matching loss is hard to converge, need to modify the architecture or loss function in later

### 2D Haar Wavelet Transform 
![그림3](https://github.com/choidaedae/wavelet-diffusion-segmentation/assets/105369646/b0fbc94f-16cc-47cf-a6df-37b3fcc63c21)
- Simple variation of wavelet transform, involving Discrete Wavelet Transform
(DWT) & Discrete Inverse Wavelet Transform (IWT)
- $L = {1\over\sqrt(2)} [1  \\  \\ 1], H = {1\over\sqrt(2)} [-1 \\  \\ 1]$ represent low-pass & high-pass filters, construct 4 kernels ($LL^T, LH^T, HL^T, HH^T$)
- Decompose the input 𝑋 ∈ $R^{H \times W}$ into 4 subbands ($X_{ll}, X_{hl}, X_{hl}, X_{hh}$) with dimensions $R^{{H\over 2} \times {W\over 2}}$
- Accurate reconstruction of the original signals 𝑋 from frequency components through IWT


### Method 1: Multi-level Noised Wavelet Feature Extractor (WaveUNet)
#### WaveUNet
1. Input image is decomposed by Multi-level DWT, first low-frequency subbands 𝑋1,𝑙𝑙 gets noise and goes into WaveUNet
2. Higher level’s low-frequency subbands are concatenated residually into WaveUNet’s layers
3. WaveUNet is trained to mimics each level’s low frequency subbands, returns to image domain by IWT
#### Feature Voter
- Can apply any general segmentation models (MLP, CNN, …)
- WaveUNet gives extracted multi-level feature map to Feature Voter, makes a per class score map
#### Model Architecture
![image](https://github.com/choidaedae/wavelet-diffusion-segmentation/assets/105369646/6d100eb0-b16b-451a-aa6c-1461d74046d8)

#### Loss Function
**1. Feature matching Loss**
- Feature extractor is learned to reduce the L2 norm of multilevel wavelet inputs & feature maps it extracts
**2. Segmentation Loss**
- Weighted multi-class cross-entropy loss
- Class distribution was highly imbalanced, weights as reciprocal of the class distribution for stable learning



### Method 2: Symmetric Contrastive Loss
- Tumor regions are almost **asymmetric**.
- If there is a tumor in one region based on the x-axis, there is very likely to no tumor in the opposite.
- Penalize not to be similar model’s prediction to opposite region’s label (Just tumor region)
![image](https://github.com/choidaedae/wavelet-diffusion-segmentation/assets/105369646/eb14e8ba-58b7-466f-bf8a-23980ede7f1d)



### Experiments & Results
#### Experiment 1.
- Used baseline model (AttentionUNet) learns to compare the effects by adding symmetric contrastive loss
- Each model trained for 100 epochs, hyperparameter 𝝀 = 0, 0.1, 0.3, 0.5
- Trained on 12,510 train datasets, and measure Dice Score for each classes with 400 validation images.
#### Experiment 2.
- Feature extraction with WaveUNet, give feature maps to AttentionUNet, λ=0.1 -> Not going well…

#### Experiment Overview
![그림2](https://github.com/choidaedae/wavelet-diffusion-segmentation/assets/105369646/2c2da684-432f-48f2-9383-894d0bb737bc)


### Dataset
- **BraTS 2021**

![image](https://github.com/choidaedae/wavelet-diffusion-segmentation/assets/105369646/6be8b32a-5be3-4ac4-88db-950da52daa04)

#### BraTS Challenge
- Challenge in MICCAI
- Evaluate state-of-the-art methods for the tumor segmentation in mpMRI scans
#### Preprocessing
- Sliced 10 timesteps(70-79) to solve 2D Brain Tumor Segmentation task
- Sliced data have 240×240 resolution with 4 modalities (t1, t1ce, t2, flair),
depending on whether a contrast agent is administered or not
- Each pixels are labeled one of 4 classes (label 0: Background, label 1: NCR, label 2: Edematous, label 4: ET)
- Preprocessed with min-max normalization to range each pixel values 0-1
- No data augmentation


  
### How to Use 
- TBD 



### Project Poster 
[Final_Poster_DaehyeonChoi.pdf](https://github.com/choidaedae/wavelet-diffusion-segmentation/files/13676733/_._.pdf)

