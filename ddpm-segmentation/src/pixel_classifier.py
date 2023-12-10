import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, in_channels, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

            
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
class UNetforFeature(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(48, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class FeatureVoter(nn.Module):
    def __init__(self, numpy_class, dim):
        super(FeatureVoter, self).__init__()
        if numpy_class < 30:
            self.layer1 = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU()
            )
            self.layer3 = nn.Linear(32, numpy_class)
                
            self.bn1 = nn.BatchNorm2d(num_features=128)
            self.bn2 = nn.BatchNorm2d(num_features=32)
        
        else:
            
            self.layer1 = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU()
            ),
            self.layer2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU()
            ),
            self.layer3 = nn.Linear(128, numpy_class),
                
            self.bn1 = nn.BatchNorm2d(num_features=256)
            self.bn2 = nn.BatchNorm2d(num_features=128)


    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.einsum('bchw->bhwc', self.bn1(torch.einsum('bhwc->bchw', x)))
        x = self.layer2(x)
        x = torch.einsum('bchw->bhwc', self.bn2(torch.einsum('bhwc->bchw', x)))
        return self.layer3(x)


def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            preds = models[MODEL_NUMBER](features.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k


"""
def save_predictions(args, image_paths, preds):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)
        np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', filename + '.jpg')
        )
"""
        
def save_predictions(args, preds, epoch):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions', str(epoch)), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations', str(epoch)), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = f'visualize_{i}'
        pred = np.squeeze(pred)
        flattened_pred = pred.flatten()
        counter_result = Counter(flattened_pred)
            
        np.save(os.path.join(args['exp_dir'], 'predictions', str(epoch), filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', str(epoch), filename + '.jpg')
        )



def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
            
    return np.array(ious).mean()

def compute_dice(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    dices = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        dice = 2*intersections[target_num] / (1e-8 + unions[target_num] + intersections[target_num])
        dices.append(dice)
        if print_per_class_ious:
            print(f"DICE Score for {class_names[target_num]} {dice:.4}")
            
    return np.array(dices)

def compute_miou_and_dice(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    dices = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        dice = 2*intersections[target_num] / (1e-8 + unions[target_num] + intersections[target_num])
        dices.append(dice)
        ious.append(iou)
        if print_per_class_ious:
            print(f"DICE Score for {class_names[target_num]} {dice:.4}")
            print(f"IOU for {class_names[target_num]} {iou:.4}")
            
    return np.array(dices).mean(), np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(FeatureVoter(args["number_class"], args['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models
