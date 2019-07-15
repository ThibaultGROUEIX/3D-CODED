from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import sys
sys.path.append('./auxiliary/')
# from datasetFaust import *
from model import *
from utils import *
from ply import *
import reconstruct
import time
from sklearn.neighbors import NearestNeighbors
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
import visdom
import global_variables
import trimesh
import os
import _thread as thread
thread.start_new_thread(os.system, ('visdom -p 8888 > /dev/null 2>&1',))

def compute_correspondances(source_p, source_reconstructed_p, target_p, target_reconstructed_p):
    """
    Given 2 meshes, and their reconstruction, compute correspondences between the 2 meshes through neireast neighbors
    :param source_p: path for source mesh
    :param source_reconstructed_p: path for source mesh reconstructed
    :param target_p: path for target mesh
    :param target_reconstructed_p: path for target mesh reconstructed
    :return: None but save a file with correspondences
    """
    # inputs are all filepaths
    with torch.no_grad():
        source = trimesh.load(source_p, process=False)
        source_reconstructed = trimesh.load(source_reconstructed_p, process=False)
        target = trimesh.load(target_p, process=False)
        target_reconstructed = trimesh.load(target_reconstructed_p, process=False)

        # project on source_reconstructed
        neigh.fit(source_reconstructed.vertices)
        idx_knn = neigh.kneighbors(source.vertices, return_distance=False)

        #correspondances throught template
        closest_points = target_reconstructed.vertices[idx_knn]
        closest_points = np.mean(closest_points, 1, keepdims=False)


        # project on target
        if global_variables.opt.project_on_target:
            print("projection on target...")
            neigh.fit(target.vertices)
            idx_knn = neigh.kneighbors(closest_points, return_distance=False)
            closest_points = target.vertices[idx_knn]
            closest_points = np.mean(closest_points, 1, keepdims=False)

        # save output
        mesh = trimesh.Trimesh(vertices=closest_points, faces=source.faces, process=False)
        mesh.export("results/correspondences.ply")
        np.savetxt("results/correspondences.txt", closest_points, fmt='%1.10f')
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--HR', type=int, default=1, help='Use high Resolution template for better precision in the nearest neighbor step ?')
    parser.add_argument('--nepoch', type=int, default=3000, help='number of epochs to train for during the regression step')
    parser.add_argument('--model', type=str, default = 'trained_models/sup_human_network_last.pth',  help='your path to the trained model')
    parser.add_argument('--inputA', type=str, default =  "data/example_0.ply",  help='your path to mesh 0')
    parser.add_argument('--inputB', type=str, default =  "data/example_1.ply",  help='your path to mesh 1')
    parser.add_argument('--num_points', type=int, default = 6890,  help='number of points fed to poitnet')
    parser.add_argument('--num_angles', type=int, default = 300,  help='number of angle in the search of optimal reconstruction. Set to 1, if you mesh are already facing the cannonical direction as in data/example_1.ply')
    parser.add_argument('--env', type=str, default="CODED", help='visdom environment')
    parser.add_argument('--clean', type=int, default=1, help='if 1, remove points that dont belong to any edges')
    parser.add_argument('--scale', type=int, default=1, help='if 1, scale input mesh to have same volume as the template')
    parser.add_argument('--project_on_target', type=int, default=0, help='if 1, projects predicted correspondences point on target mesh')


    opt = parser.parse_args()
    global_variables.opt = opt
    vis = visdom.Visdom(port=8888, env=opt.env)

    distChamfer =  ext.chamferDist()

    # =============Get data and template======================================== #
    if not os.path.exists("./trained_models/sup_horse_network_last.pth"):
        os.system('./trained_models/download_models.sh')
    if not os.path.exists("./data/template/template.ply"):
        os.system('./data/download_template.sh')
    # ========================================================== #




    # load network
    global_variables.network = AE_AtlasNet_Humans(num_points=opt.num_points)
    global_variables.network.cuda()
    global_variables.network.apply(weights_init)
    if opt.model != '':
        print("using model: ", opt.model)
        global_variables.network.load_state_dict(torch.load(opt.model))
    global_variables.network.eval()

    neigh = NearestNeighbors(1, 0.4)
    opt.manualSeed = random.randint(1, 10000) # fix seed
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    start = time.time()
    print("computing correspondences for " + opt.inputA + " and " + opt.inputB)

    # Reconstruct meshes
    reconstruct.reconstruct(opt.inputA)
    reconstruct.reconstruct(opt.inputB)

    # Compute the correspondences through the recontruction
    compute_correspondances(opt.inputA, opt.inputA[:-4] + "FinalReconstruction.ply", opt.inputB, opt.inputB[:-4] + "FinalReconstruction.ply")
    end = time.time()
    print("ellapsed time is ", end - start, " seconds !")