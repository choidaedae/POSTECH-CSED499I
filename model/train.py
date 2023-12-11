import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import sys
import random
import wandb 
from torch.utils.data import DataLoader
from torchvision.transforms.functional import vflip
import numpy as np
import argparse
from src.utils import setup_seed, multi_acc
from src.attention_unet import AttentionUNet
from src.pixel_classifier import load_ensemble, compute_iou, compute_dice, compute_miou_and_dice, predict_labels, save_predictions,  FeatureVoter
from src.brats import Brats2DDataset
from src.feature_extractors import create_feature_extractor, collect_features, _collect_features
from src.loss import FMatchingLoss, SymmetricContLoss
from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev, dev_num
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from wavelets.DWT_IDWT_Layers import DWT_2D, IDWT_2D



def train(args):
    
    train_args = args['train'] 
    train_dir = args['train_dir']
    
    train_data = Brats2DDataset(train_dir)
    train_loader = DataLoader(dataset=train_data, batch_size=train_args['batch_size'], shuffle = True, drop_last = True, num_workers=16) # brats dataset loader 

    run_device = train_args['run_device']
    print(f"Number of data: {len(train_loader)}")
    wandb.log({'run_device': f"cuda:{run_device}"})
    nSamples = [54667, 275, 1677, 981]
    
    normedWeights = []
    for x in nSamples:
        normedWeights.append(1/x)
    normedWeights = torch.FloatTensor(normedWeights)
    normedWeights = (normedWeights / torch.sum(normedWeights)).to(f"cuda:{run_device}")
    
    L_seg = nn.CrossEntropyLoss(weight = normedWeights) # or Dice Loss 
    L_match = FMatchingLoss(args['wavelet_level']) # to be implemented 
    L_contrast = SymmetricContLoss() 
    feature_extractor = create_feature_extractor(**args)
    model, _ = feature_extractor.model, feature_extractor.diffusion
    seg_net = AttentionUNet(img_ch = 48, output_ch = args['number_class']).to(dev_num(run_device))
    
    wandb.watch(model)
    wandb.watch(seg_net)
    
    best_dice = 0
    
    _lambda = train_args['lambda'] # hyperparameter
    wandb.log({'lambda': _lambda})

    optimizer = torch.optim.Adam(params = [{'params': model.parameters(), 'lr': 0.001},
        {'params': seg_net.parameters(), 'lr': 0.001}], lr=0.001)
    
    for epoch in range(train_args['epochs']): # epoch 
    
        for i, (image, label) in enumerate(train_loader): # iteration 
            
            if i == train_args['train_iters']: break
            image, label = image.to(dev_num(run_device)), label.to(dev_num(run_device))
            optimizer.zero_grad()
        
            wavelets, freq_features = feature_extractor(image)
            features = feature_extractor.collect_wavelet_features(args, freq_features) # using IWT, collect featuremaps
            out = seg_net(features)
            p_out = torch.softmax(out, dim=1)
            
            # Frequency Level Feature Matching Loss for the Feature extractor
            l_match = L_match(wavelets, freq_features) 
            
            # Segmentation Loss for the FeatureVoter
            label = torch.squeeze(label.long())
            l_seg = L_seg(p_out, label) # Multi-level Cross entropy Loss
            ohot_label = torch.einsum('bhwc->bchw', F.one_hot(label, args['number_class']))
            
            # Contrastive Loss needs binary prediction, whether the region is tumor or not 
            b_label = torch.stack((ohot_label[:, 0, :, :], ohot_label[:, 1:, :, :].sum(dim=1)), dim=1)
            b_out = torch.stack((p_out[:, 0, :, :], p_out[:, 1:, :, :].sum(dim=1)), dim=1)
            b_label_flip = vflip(b_label) # vertical flip 
            l_contrast = L_contrast(b_out, b_label_flip)
            
            loss = l_seg + l_contrast*_lambda + l_match # l_contrast: dot product
            if i % 1 == 0: print(f'[Train Mode] Current Epoch: {epoch}, Current Iteration: {i}, Train Loss: {loss}')
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': loss})
            
            
        if i % train_args['val_freq'] == 0: 
            _, dice = validation(args, epoch, feature_extractor, seg_net)
            wandb.log({"Epoch": epoch})
            wandb.log({"mean Dice Score": dice.mean()})
            wandb.log({"Dice Score for NCR": dice[1]})
            wandb.log({"Dice Score for Edema": dice[2]})
            wandb.log({"Dice Score for Enhan Tumor": dice[3]})
            
            if(dice.mean() > best_dice):
                best_dice = dice.mean()
                torch.save(seg_net.state_dict(), os.path.join(train_args['save_path'], f'feature_voter_e{epoch}.pt'))

            
def validation(args, epoch, feature_extractor, seg_net):
    val_args = args['validation']
    val_dir = args['val_dir']
    
    val_data = Brats2DDataset(val_dir)
    val_loader = DataLoader(dataset=val_data, batch_size=val_args['batch_size'], shuffle = False, drop_last = True) # brats dataset loader 
    mIoU = 0
    
    run_device = val_args['run_device']
    with torch.no_grad():
        feature_extractor.model.eval()
        seg_net.eval()
        preds = []
        labels = []
        for i, (image, label) in enumerate(val_loader): # iteration 
            
            if (i == val_args['val_iters']): break

            print(f"[Validation mode] Current Iteration: {i}")
            
            image, label = image.to(dev_num(run_device)), label.to(dev_num(run_device))
        
            wavelets, freq_features = feature_extractor(image)
            features = feature_extractor.collect_wavelet_features(args, freq_features) # using IWT, collect featuremaps
            out = seg_net(features)
            p_out = torch.softmax(out, dim=1)
            
            image, label = image.to(dev_num(run_device)), label.to(dev_num(run_device))
            _, freq_features = feature_extractor(image)
            features = feature_extractor.collect_wavelet_features(args, freq_features) # using IWT, collect featuremaps
            pred = torch.argmax(seg_net(features), dim=1).to('cpu').numpy() 
            label = torch.squeeze(label.long()).to('cpu').numpy() 
            preds.append(pred)
            labels.append(label)
            
        # calculate mIoU
        save_predictions(args, preds, epoch)
            
        # calculate mIoU
        mIoU = compute_iou(args, preds, labels)
        dice = compute_dice(args, preds, labels)
        print(f'[Validation Mode] Current epoch: {epoch}, epoch IoU: {mIoU}, epoch DICE: {dice.mean()}')
        
        return mIoU, dice 
            
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    default_json_dir = "/root/daehyeonchoi/POSTECH-CSED491/wavelet-diffusion-segmentation/model/experiments/ddpm.json"

    
    parser.add_argument('--exp', type=str, default=default_json_dir)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]
    
    config = {
        'model_name' : 'ddpm-segmentation-w/oFMatchingLoss/-wContLoss',
        'batch_size' : 4,
        'epoch' : opts['train']['epochs'],
        'criterion' : 'ContLoss',
        'optimizer' : 'adam',

    }
    
    wandb.init(reinit=True, project='wavediffseg-test', config=config)

    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = 'WaveUNet'
        suffix += '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    train(opts)
    
    
    
    

        