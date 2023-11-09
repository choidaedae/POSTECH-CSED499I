""" Parts of the Wavelet U-Net model """

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


class WaveDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    

    def forward(self, x, subbands): # subbands: N numbers of (B, C, H, W) tensors 
        wave_x = torch.cat(subbands, dim =1) # wave_x = 1 tensor with shape (B, NC, H, W)
        x = self.maxpool(x)
        x = torch.cat([x, wave_x], dim = 1) # concat or elementwise add 
        x = self.double_conv(x)
        return x


class WaveUp(nn.Module):
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

        self.inc = DoubleConv(n_channels, 16) # 240x245
        # add 1st level wavelet subband (with 16 channels), 64+16 = 80 channels 
        self.down1 = WaveDown(16+16, 64) # 120x120
        # add 2nd level wavelet subband (with 16 channels), 160+16 = 176 channels 
        self.down2 = WaveDown(64+16, 160) # 60x60
        # add 3rd level wavelet subband (with 16 channels), 352+16 = 368 channels 
        self.down3 = WaveDown(160+16, 352) # 30x30
        
        self.up3 = WaveUp(160 + 352, 256)
        self.up2 = WaveUp(256 + 64, 160)
        self.up1 = WaveUp(160 + 16, 88)
        self.dec = DoubleConv(88, 32)
        self.last = DoubleConv(32, n_classes)
    
    def forward(self, x):
        
        xll = x 
        self.device = x.device
        for _ in range(self.wavelet_levels):
            xll, xhl, xlh, xhh = self.DWT(xll)
            self.subbands.append([xll, xhl, xlh, xhh])
        # x shape = (b, 4, 240, 240)
        x1 = self.inc(x)  # x1 shape = (b, 16, 240, 240)
        x2 = self.down1(x1, self.subbands[0]) # x2 shape = (b, 64, 120, 120)
        x3 = self.down2(x2, self.subbands[1]) # x3 shape = (b, 160, 60, 60)
        x4 = self.down3(x3, self.subbands[2]) # x4 shape = (b, 352, 30, 30) -> IWT -> Matching 
        x5 = self.up3(x4, x3) # x shape = (b, 256, 60, 60) -> IWT -> Matching 
        x6 = self.up2(x5, x2) # x shape = (b, 160, 120, 120) -> IWT -> Matching 
        x = self.up1(x6, x1) # x shape = (b, 88, 240, 240) 
        x = self.dec(x) # x shape = (b, 16, 240, 240)
        x = self.last(x) # x shape = (b, 4, 240, 240) -> Matching 
        return x 
    
if __name__ == "__main__":
    
    device = torch.device("cuda:6")
    
    x = torch.randn(4, 4, 240, 240) # 4 channels image (flair, t1, t1ce, t2)
    y_label = torch.randn(4, 240, 240)
    model = WaveletUNet(n_channels=4, n_classes=4, wavelet_levels=3, bilinear=False)
    
    y_pred = model(x)
    
    print(x.shape)
    
    