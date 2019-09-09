
from __future__ import print_function
import sys
sys.path.append('./auxiliary/')
sys.path.append('/app/python/')
sys.path.append('./')
import my_utils
print("fixed seed")
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from datasetSMPL2 import *
from model import *
from ply import *
import os
import json
import datetime
import correspondences
import parser
parser = argparse.ArgumentParser()
print(sys.path)
# ========
parser.add_argument('--model_path', type=str, default = 'trained_models/sup_human_network_last.pth',  help='your path to the trained model')
parser.add_argument('--dataset_path', type=str, default = './export/',  help='your path to the trained model')
parser.add_argument('--id', type=str, default = '1',  help='your path to the trained model')
parser.add_argument('--LR_input', type=int, default=1, help='Use high Resolution template for better precision in the nearest neighbor step ?')
parser.add_argument('--randomize', type=int, default=0)

opt = parser.parse_args()
opt.LR_input = my_utils.int_2_boolean(opt.LR_input)
opt.randomize = my_utils.int_2_boolean(opt.randomize)
my_utils.plant_seeds(randomized_seed=opt.randomize)


#
# =====DEFINE CHAMFER LOSS======================================== #
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
now = datetime.datetime.now()
save_path = now.isoformat()
save_path = opt.id + save_path
dir_name = os.path.join('log_inference', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
inf = correspondences.Inference(model_path = opt.model_path, save_path=dir_name, LR_input=opt.LR_input)


if not os.path.exists("log_inference"):
    print("Creating log_inference folder")
    os.mkdir("log_inference")




inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_006.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_021.ply"), path = "006_021.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_011.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_107.ply"), path = "011_107.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_012.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_043.ply"), path = "012_043.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_015.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_098.ply"), path = "015_098.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_028.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_079.ply"), path = "028_079.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_033.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_080.ply"), path = "033_080.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_038.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_196.ply"), path = "038_196.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_039.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_142.ply"), path = "039_142.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_045.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_114.ply"), path = "045_114.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_046.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_128.ply"), path = "046_128.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_055.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_191.ply"), path = "055_191.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_056.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_077.ply"), path = "056_077.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_061.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_104.ply"), path = "061_104.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_065.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_165.ply"), path = "065_165.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_067.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_019.ply"), path = "067_019.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_078.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_171.ply"), path = "078_171.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_084.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_183.ply"), path = "084_183.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_089.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_134.ply"), path = "089_134.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_090.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_068.ply"), path = "090_068.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_093.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_010.ply"), path = "093_010.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_094.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_133.ply"), path = "094_133.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_095.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_117.ply"), path = "095_117.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_101.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_152.ply"), path = "101_152.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_115.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_199.ply"), path = "115_199.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_124.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_173.ply"), path = "124_173.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_129.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_053.ply"), path = "129_053.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_131.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_174.ply"), path = "131_174.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_136.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_002.ply"), path = "136_002.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_137.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_072.ply"), path = "137_072.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_146.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_025.ply"), path = "146_025.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_149.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_057.ply"), path = "149_057.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_151.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_184.ply"), path = "151_184.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_155.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_112.ply"), path = "155_112.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_156.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_014.ply"), path = "156_014.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_158.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_187.ply"), path = "158_187.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_161.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_103.ply"), path = "161_103.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_166.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_037.ply"), path = "166_037.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_167.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_044.ply"), path = "167_044.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_178.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_180.ply"), path = "178_180.txt")
inf.forward(inputA=os.path.join(opt.dataset_path,"test_scan_198.ply"), inputB=os.path.join(opt.dataset_path,"test_scan_029.ply"), path = "198_029.txt")
os.system(f"cd {dir_name}; zip -r base{opt.id}.zip *_*.txt")
# os.system(f"cd {dir_name}; zip -r base{opt.id}.zip *_*.txt ; cd ../../; rm -r {dir_name}")