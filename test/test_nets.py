from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from dataset_surreal import *
from model import *
from utils import *
import os
import json
import datetime
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default = 'trained_models/sup_human_network_last.pth',  help='your path to the trained model')
parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
# ========================================================== #

# =============Get data and template======================================== #
if not os.path.exists("./data/datas_surreal_train.pth"):
    os.system('./data/download_data.sh')
if not os.path.exists("./data/template/template.ply"):
    os.system('./data/download_template.sh')
# ========================================================== #



# =============DEFINE stuff for logs ======================================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

val_loss_L2_SURREAL = AverageValueMeter()

# ===================CREATE DATASET================================= #
dataset_SURREAL_test = SURREAL(train=False)
dataloader_SURREAL_test = torch.utils.data.DataLoader(dataset_SURREAL_test, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
# ========================================================== #

# ===================CREATE network================================= #
network = AE_AtlasNet_Humans()
network.cuda()  # put network on GPU
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #


# ========================================================== #

# =============start of the learning loop ======================================== #

def test_trained_nets():
    with torch.no_grad():
        #val on SURREAL data
        network.eval()
        val_loss_L2_SURREAL.reset()
        for i, data in enumerate(dataloader_SURREAL_test, 0):
            points, fn,_,  idx = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            pointsReconstructed = network(points)  # forward pass
            loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
            val_loss_L2_SURREAL.update(loss_net.item())  

        print("test loss: ", val_loss_L2_SURREAL.avg)       
    assert(val_loss_L2_SURREAL.avg < 0.0002)

test_trained_nets()