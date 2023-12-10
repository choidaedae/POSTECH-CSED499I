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
    train_loader = DataLoader(dataset=train_data, batch_size=train_args['batch_size'], shuffle = True, drop_last = True) # brats dataset loader 

    run_device = train_args['run_device']
    print(f"Number of data: {len(train_loader)}")
    wandb.log({'run_device': f"cuda:{run_device}"})
    nSamples = [54667, 275, 1677, 981]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(f"cuda:{run_device}")
    
    L_seg = nn.CrossEntropyLoss(ignore_index = 0, weight = normedWeights) # or Dice Loss 
    L_match = FMatchingLoss(args['wavelet_level']) # to be implemented 
    L_contrast = SymmetricContLoss() if args['L_contrast'] else None
    feature_extractor = create_feature_extractor(**args)
    model, _ = feature_extractor.model, feature_extractor.diffusion
    seg_net = FeatureVoter(args['number_class'], args['dim'][-1]).to(dev_num(run_device))
    seg_net.init_weights()
    
    wandb.watch(model)
    wandb.watch(seg_net)
    best_mIoU = 0
    #model_parameters = list(model.parameters()) + list(seg_net.parameters())
    # parameter5
    optimizer = torch.optim.Adam(params = [{'params': model.parameters(), 'lr': 0.001},
        {'params': seg_net.parameters(), 'lr': 0.001}], lr=0.001)
    
    for epoch in range(train_args['epochs']): # epoch 
        
        for i, (image, label) in enumerate(train_loader): # iteration 
            image, label = image.to(dev_num(run_device)), label.to(dev_num(run_device))
            optimizer.zero_grad()
        
            wavelets, freq_features = feature_extractor(image)
            features = feature_extractor.collect_wavelet_features(args, freq_features) # using IWT, collect featuremaps
            features = torch.einsum('bchw->bhwc', features)
            out = torch.einsum('bhwc->bchw', seg_net(features))
            p_out = torch.softmax(out, dim=1)
            
            # Frequency Level Feature Matching Loss for the Feature extractor
            l_match = L_match(wavelets, freq_features) 
            
            # Segmentation Loss for the FeatureVoter
            label = torch.squeeze(label.long())
            l_seg = L_seg(out, label) # Multi-level Cross entropy Loss
            ohot_label = torch.einsum('bhwc->bchw', F.one_hot(label, args['number_class']))
            
            # Contrastive Loss needs binary prediction, whether the region is tumor or not 
            b_label = torch.stack((ohot_label[:, 0, :, :], ohot_label[:, 1:, :, :].sum(dim=1)), dim=1)
            b_out = torch.stack((p_out[:, 0, :, :], p_out[:, 1:, :, :].sum(dim=1)), dim=1)
            b_label_flip = vflip(b_label) # vertical flip 
            l_contrast = L_contrast(b_out, b_label_flip)
            
            _lambda = train_args['lambda'] # hyperparameter
            wandb.log({'lambda': _lambda})
            #loss = l_match
            loss = l_seg + l_contrast + l_match# l_contrast: dot product
            print(f'[Train Mode] Current Epoch: {epoch}, Current Iteration: {i}, Train Loss: {loss}')
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            
            wandb.log({'train_loss': loss})
        
            
        if i % train_args['val_freq'] == 0: 
            mIoU = validation(args, epoch, feature_extractor, seg_net)
            wandb.log({"mean IoU": mIoU})
            if(mIoU > best_mIoU):
                best_mIoU = mIoU
                torch.save(feature_extractor.model.state_dict(), os.path.join(train_args['save_path'], f'feature_extractor_e{epoch}.pt'))
                torch.save(seg_net.state_dict(), os.path.join(train_args['save_path'], f'feature_voter_e{epoch}.pt'))
                
        
        if i % train_args['sampling_freq'] == 0:
            # get_visualize
            # sampling
            pass

            
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
            _, freq_features = feature_extractor(image)
            features = feature_extractor.collect_wavelet_features(args, freq_features) # using IWT, collect featuremaps
            features = torch.einsum('bchw->bhwc', features)
            pred = torch.einsum('bhwc->bchw', seg_net(features))
            pred = torch.argmax(torch.einsum('bhwc->bchw', seg_net(features)), dim=1).to('cpu').numpy()
            label = torch.squeeze(label.long()).to('cpu').numpy() 
            preds.append(pred)
            labels.append(label)
            
        # calculate mIoU
        mIoU = compute_iou(args, preds, labels)
        dice = compute_dice(args, preds, labels)
        print(f'[Validation Mode] Current epoch: {epoch}, epoch IoU: {mIoU}, epoch DICE: {dice}')
        return mIoU 
            
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    default_json_dir = "/root/daehyeonchoi/csed491/wavediffseg/ddpm-segmentation/experiments/brats/ddpm.json"

    
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
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    train(opts)
    
    
    
    

        