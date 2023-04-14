from __future__ import print_function, division
import argparse
from fileinput import filename
from operator import truediv
import os
import random
from tkinter.tix import Tree
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
from datasets.data_io import get_transform, read_all_lines

import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc



cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Bidirectional-Semi-Dualbranch-CNN (BSDual-CNN)')
parser.add_argument('--model', default='Bidirectional-DualBranch-CNN', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--filelist', default='./filenames/kitti12_test.txt', help='testing list')
parser.add_argument('--dataset', default='kitti', help='dataset name')

parser.add_argument('--loadckpt', default= './')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--savepath', default='./')
parser.add_argument('--datapath', default='./', help='data path')

parser.add_argument('--CV',action='store_true', help='conduct Cross-Validation experiments')
parser.add_argument('--fold_num', type=int, default=1, help='choose the fold number in cross-validation')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# model, optimizer
model = __models__[args.model](args.maxdisp)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
model = nn.DataParallel(model)
model.cuda()


# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'], strict= True)


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    #transforms.Resize(512),
    
    
    if args.CV:
        assert args.fold_num is not None
        
        lines = read_all_lines(args.filelist)
        splits = [line.split() for line in lines]
        if args.dataset == 'scared' :
            fold2scared = [1,2,3,6,7] # In scared dataset, we only use d1, d2, d3, d6, d7 for 5-fold cross-validation
            file_name = 'd'+ str(fold2scared[args.fold_num]) 
        else:
            file_name = 'Experiment_' + str(args.fold_num)


        test_left_img = [x[0] for x in splits if x[0].find(file_name) > -1]
        test_right_img = [x[1] for x in splits if x[0].find(file_name) > -1]

    else:
        lines = read_all_lines(args.filelist)
        splits = [line.split() for line in lines]
        
        test_left_img = [x[0] for x in splits ]
        test_right_img = [x[1] for x in splits ]


    print(test_left_img)

    for inx in range(len(test_left_img)):

        imgL_o = Image.open(os.path.join(args.datapath, test_left_img[inx])).convert('RGB')
        imgR_o = Image.open(os.path.join(args.datapath, test_right_img[inx])).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        # pad to width and hight to 16 times
        if args.dataset == 'kitti' :
            w, h = imgL_o.size
            top_pad = 384 - h
            right_pad = 1248 - w
            
            str_inx = test_left_img[inx].split('/')[-1]
            str_inx = str_inx.split('.')[0]

        elif args.dataset == 'servct' :
            top_pad = 0
            right_pad = 16    

            str_inx = str((args.fold_num-1)*8+inx+1).zfill(3)

        else:
            top_pad = 0
            right_pad = 0   

            str_inx = test_left_img[inx].replace('/','-')
            str_inx = str_inx.split('-Left')[0]

        print(str_inx)

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)


        pred_disp, pred_conf = test_sample(imgL, imgR)
        

        if not os.path.exists(args.savepath):
            os.makedirs(args.savepath)

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
            conf = pred_conf[top_pad:,:-right_pad]
        else:
            img = pred_disp
            conf = pred_conf
        
        img = np.array(img)
        
        img = 256 * img
        img = img.astype(np.uint16)
        img = Image.fromarray(img)
        
        img_path = args.savepath + str_inx + '.png'
        img.save(img_path)

        conf = np.array(conf)
        conf = conf * 255
        conf = Image.fromarray(conf).convert('L')
        
        conf_path = args.savepath + '/conf' + str_inx + '.png'
        conf.save(conf_path)


# test one sample
@make_nograd_func
def test_sample(imgL, imgR):
    model.eval()

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    disp_ests_l, confmap_l = model(imgL, imgR, step = 1)
    disp_ests_r, confmap_r = model(imgL, imgR, step = 2)

    if torch.sum(confmap_l[0]) >= torch.sum(confmap_r[0]):
        confmap = torch.squeeze(confmap_l[0])
        disp = disp_ests_l[0]
    else:
        confmap = torch.squeeze(confmap_r[0])
        disp = disp_ests_r[0]

    confmap = torch.squeeze(confmap).data.cpu()
    disp = torch.squeeze(disp).data.cpu()

    return disp, confmap


if __name__ == '__main__':
    main()
