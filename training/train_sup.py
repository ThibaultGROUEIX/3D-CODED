from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
sys.path.append("/app/python/")

from datasetSMPL2 import *
from model import *
from utils import *
from ply import *
import os
import json
import datetime
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
vis = visdom.Visdom(port=8888, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = 1 #random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
L2curve_train_smpl = []
L2curve_val_smlp = []

# meters to record stats on learning
train_loss_L2_smpl = AverageValueMeter()
val_loss_L2_smpl = AverageValueMeter()
tmp_val_loss = AverageValueMeter()
# ========================================================== #


# ===================CREATE DATASET================================= #
dataset = SMPL(train=True, regular = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_smpl_test = SMPL(train=False)
dataloader_smpl_test = torch.utils.data.DataLoader(dataset_smpl_test, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = AE_AtlasNet_Humans()
network.cuda()  # put network on GPU
network.apply(weights_init)  # initialization of the weight
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')
# ========================================================== #

# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    if epoch==80:
        lrate = lrate/10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)
    if epoch==90:
        lrate = lrate/10.0  # learning rate scheduled decay
        optimizer = optim.Adam(network.parameters(), lr=lrate)

    # TRAIN MODE
    train_loss_L2_smpl.reset()
    network.train()
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        points, idx,_ = data
        points = points.transpose(2, 1).contiguous()
        points = points.cuda()
        pointsReconstructed = network.forward_idx(points, idx)  # forward pass
        loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
        loss_net.backward()
        train_loss_L2_smpl.update(loss_net.item())
        optimizer.step()  # gradient update

        # VIZUALIZE
        if i % 100 == 0:
            vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
            win = 'Train_input',
            opts = dict(
                title="Train_input",
                markersize=2,
            ),
            )
            vis.scatter(X=pointsReconstructed[0].data.cpu(),
            win = 'Train_output',
            opts = dict(
                title="Train_output",
                markersize=2,
            ),
            )

        print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / 32,  loss_net.item()))

    # Validation
    with torch.no_grad():
        #val on SMPL data
        network.eval()
        val_loss_L2_smpl.reset()
        for i, data in enumerate(dataloader_smpl_test, 0):
            points, fn, idx = data
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            pointsReconstructed = network(points)  # forward pass
            loss_net = torch.mean(
                (pointsReconstructed - points.transpose(2, 1).contiguous()) ** 2)
            val_loss_L2_smpl.update(loss_net.item())
            # VIZUALIZE
            if i % 10 == 0:
                vis.scatter(X=points.transpose(2, 1).contiguous()[0].data.cpu(),
                            win='Test_smlp_input',
                            opts=dict(
                                title="Test_smlp_input",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsReconstructed[0].data.cpu(),
                            win='Test_smlp_output',
                            opts=dict(
                                title="Test_smlp_output",
                                markersize=2,
                            ),
                            )
            print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset / 32, loss_net.item()))
      

        # UPDATE CURVES
        L2curve_train_smpl.append(train_loss_L2_smpl.avg)
        L2curve_val_smlp.append(val_loss_L2_smpl.avg)

        vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smlp)))),
                 Y=np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smlp))),
                 win='loss',
                 opts=dict(title="loss", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smlp" + opt.env,]))

        vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smlp)))),
                 Y=np.log(np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smlp)))),
                 win='log',
                 opts=dict(title="log", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smlp" + opt.env,]))

        # dump stats in log file
        log_table = {
            "val_loss_L2_smpl": val_loss_L2_smpl.avg,
            "train_loss_L2_smpl": train_loss_L2_smpl.avg,
            "epoch": epoch,
            "lr": lrate,
            "env": opt.env,
        }
        print(log_table)
        with open(logname, 'a') as f:  # open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')
        #save latest network
        torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
