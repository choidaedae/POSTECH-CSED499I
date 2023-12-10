import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from src.feature_extractors
from src.pixel_classfier import pixel_classifier
from PIL import Image


class DiffSeg(nn.Module):
    def __init__(self, args):
        super(DiffSeg, self).__init__()
        self.model, self.diffusion = create_model_and_diffusion(args)
        self.segmentation = pixel_classifier(args['num_classes'])
        self.steps = args['num_timesteps']

    def feature_extractor_forward(self, x, noise):  # for feature_extraction 
        activations = [] 
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        size = x.shape
        for feats in activations:
            feats = nn.functional.interpolate(feats, size=size, mode = 'bilinear')
            resized_activations.append(feats)
        
        features = torch.cat(resized_activations, dim=0)
        return features
        
    def segmentation_forward(self, features): # for segmentation 
        return self.segmentation(x)
    
    def forward(self, x, noise):
        features = feature_extractor_forward(x, noise)
        out = segmentation_forward(features)
        return out # output: N(Classes), H, W 

if __name__=='__main__':
    
    x = torch.randn((4, 4, 240, 240))
    
    Model = DiffSeg(args)
    
    out = Model(x)   
    print(out.shape)

