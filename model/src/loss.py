import torch
from torch import nn

class FMatchingLoss(nn.Module):
    def __init__(self, wavelet_level):
        super().__init__()
        self.Lfunc = nn.MSELoss()
        self.wavelet_level = wavelet_level
        
    def forward(self, wavelets, features):
        
        loss = 0.
        
        for timestep in range(len(wavelets)):
            for wavelet, feature in zip(wavelets[timestep], features[timestep]):
                loss += self.Lfunc(wavelet[0], feature) # LL

        return loss / (self.wavelet_level * len(wavelets))
        
    
class SymmetricContLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, b_out, b_label):
        b_size = b_out.shape[0]
        epr = b_out[:, :, :, :] * b_label[:, :, :, :]
        loss = torch.sum(epr, dim=(0, 1, 2, 3)) / b_size
        loss = loss / (b_out.shape[-1]*b_out.shape[-2]) # per pixel mean 
        
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, pred, target, smooth = 1e-5):
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
        # dice coefficient
        dice = 2.0 * (intersection + smooth) / (union + smooth)
    
        # dice loss
        dice_loss = 1.0 - dice
        
        return dice_loss.sum()
    
class SymmetricContLoss_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.CE = nn.CrossEntropyLoss()
        pass
    
    def forward(self, b_out, b_label):
        ce = self.CE((b_out, b_label))
        loss = 1 / ce + 1e-8
        
        return loss
    