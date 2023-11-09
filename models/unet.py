""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from wavelets import DWT_2D, IDWT_2D


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    

    def forward(self, x): # subbands: N numbers of (B, C, H, W) tensors 
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) # 이 과정에서 channel 2배로 줄어듦 
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WaveletUNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet_levels, bilinear=True):
        super(WaveletUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.wavelet_levels = wavelet_levels
        self.subbands = []
        self.DWT = DWT_2D('haar')
        self.IDWT = IDWT_2D('haar')

        self.inc = DoubleConv(n_channels, 16) #240x240
        self.down1 = Down(16, 64) # 120x120
        self.down2 = Down(64, 128) # 60x60
        self.down3 = Down(128, 256) # 30x30
        
        self.up3 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up1 = Up(64 + 16, 40)
        self.last = DoubleConv(32, n_classes)
    
    def forward(self, x):
        
        x1 = self.inc(x)  
        x2 = self.down1(x1, self.subbands[0]) 
        x3 = self.down2(x2, self.subbands[1]) 
        x4 = self.down3(x3, self.subbands[2]) 
        x5 = self.up3(x4, x3) 
        x6 = self.up2(x5, x2) 
        x = self.up1(x6, x1) 
        x = self.last(x) 
        return x 
    
if __name__ == "__main__":
    
    device = torch.device("cuda:6")
    
    x = torch.randn(4, 4, 240, 240) # 4 channels image (flair, t1, t1ce, t2)
    y_label = torch.randn(4, 240, 240)
    model = WaveletUNet(n_channels=4, n_classes=4, wavelet_levels=3, bilinear=False)
    
    y_pred = model(x)
    
    print(x.shape)
    
    