import torch
import torch.nn as nn
import numpy as np

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dimensions',default=32, type=int,help='channel expansion in Unet')  # odd
parser.add_argument('--gpu_id', type=int, default=0, help='gpu number in a cluster')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int,help='training batch size')
parser.add_argument('--log', default="./log",help='log dump')
parser.add_argument('--epoch', default=50, type=int, help='max epoch')
parser.add_argument('--root', default="./data/", type=str,help='data root directory')
parser.add_argument('--resume_training_from', default=None, help='address of the last saved ckpt')

args = parser.parse_args()