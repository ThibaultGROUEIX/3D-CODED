from __future__ import print_function
import sys

sys.path.append('./auxiliary/')
sys.path.append('./')
import numpy as np
import torch.optim as optim
import torch.nn as nn
import model
import ply
import time
from sklearn.neighbors import NearestNeighbors

sys.path.append("./extension/")
sys.path.append("./training/")
sys.path.append("/app/python/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()
import trimesh
import torch
import pandas as pd
import os
from mpl_toolkits import mplot3d
import argument_parser
import my_utils
import trainer
import matplotlib.pyplot as plt


class Inference(object):
    def __init__(self, HR=1, reg_num_steps=3000, model_path='trained_models/sup_human_network_last.pth', num_points=6890,
                 num_angles=100, clean=1, scale=1, project_on_target=0, save_path=None, LR_input=True, network = None):
        self.LR_input = LR_input
        self.HR = HR
        self.reg_num_steps = reg_num_steps
        self.model_path = model_path
        self.num_points = num_points
        self.num_angles = num_angles
        self.clean = clean
        self.scale = scale
        self.project_on_target = project_on_target
        self.distChamfer = ext.chamferDist()

        # load network
        if network is None:
            self.network = model.AE_AtlasNet_Humans(num_points=self.num_points)
            self.network.cuda()
            self.network.apply(my_utils.weights_init)
            if self.model_path != '':
                print("Reload weights from : ", self.model_path)
                self.network.load_state_dict(torch.load(self.model_path))
        else:
            self.network = network
        self.network.eval()

        self.neigh = NearestNeighbors(1, 0.4)
        self.mesh_ref = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_ref_LR = trimesh.load("./data/template/template.ply", process=False)

        # load colors
        self.red_LR = np.load("./data/template/red_LR.npy").astype("uint8")
        self.green_LR = np.load("./data/template/green_LR.npy").astype("uint8")
        self.blue_LR = np.load("./data/template/blue_LR.npy").astype("uint8")
        self.red_HR = np.load("./data/template/red_HR.npy").astype("uint8")
        self.green_HR = np.load("./data/template/green_HR.npy").astype("uint8")
        self.blue_HR = np.load("./data/template/blue_HR.npy").astype("uint8")
        self.save_path = save_path

    def compute_correspondances(self, source_p, source_reconstructed_p, target_p, target_reconstructed_p, path):
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
            self.neigh.fit(source_reconstructed.vertices)
            idx_knn = self.neigh.kneighbors(source.vertices, return_distance=False)

            # correspondances throught template
            closest_points = target_reconstructed.vertices[idx_knn]
            closest_points = np.mean(closest_points, 1, keepdims=False)

            # project on target
            if self.project_on_target:
                print("projection on target...")
                self.neigh.fit(target.vertices)
                idx_knn = self.neigh.kneighbors(closest_points, return_distance=False)
                closest_points = target.vertices[idx_knn]
                closest_points = np.mean(closest_points, 1, keepdims=False)

            # save output
            if not os.path.exists("results"):
                print("Creating results folder")
                os.mkdir("results")

            if path is None:
                np.savetxt("results/correspondences.txt", closest_points, fmt='%1.10f')
            else:
                np.savetxt(os.path.join(self.save_path, path), closest_points, fmt='%1.10f')
            mesh = trimesh.Trimesh(vertices=closest_points, faces=source.faces, process=False)
            mesh.export("results/correspondences.ply")

    def forward(self, inputA="data/example_0.ply", inputB="data/example_1.ply", path=None):
        print("computing correspondences for " + inputA + " and " + inputB)
        start = time.time()

        # Reconstruct meshes
        self.reconstruct(inputA)
        self.reconstruct(inputB)

        # Compute the correspondences through the recontruction
        if self.save_path is None:
            self.compute_correspondances(inputA, inputA[:-4] + "FinalReconstruction.ply", inputB,
                                         inputB[:-4] + "FinalReconstruction.ply", path)
        else:
            self.compute_correspondances(inputA,
                                         os.path.join(self.save_path, inputA[-8:-4] + "FinalReconstruction.ply"),
                                         inputB,
                                         os.path.join(self.save_path, inputB[-8:-4] + "FinalReconstruction.ply"), path)

        end = time.time()
        print("ellapsed time is ", end - start, " seconds !")

    def regress(self, points):
        """
        search the latent space to global_variables. Optimize reconstruction using the Chamfer Distance
        :param points: input points to reconstruct
        :return pointsReconstructed: final reconstruction after optimisation
        """
        points = points.data
        latent_code = self.network.encoder(points)
        lrate = 0.001  # learning rate
        # define parameters to be optimised and optimiser
        input_param = nn.Parameter(latent_code.data, requires_grad=True)
        self.optimizer = optim.Adam([input_param], lr=lrate)
        loss = 10
        i = 0

        # learning loop
        while np.log(loss) > -9 and i < self.reg_num_steps:
            self.optimizer.zero_grad()
            pointsReconstructed = self.network.decode(input_param)  # forward pass
            pointsReconstructed = pointsReconstructed.view(pointsReconstructed.size(0), -1, 3)

            dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            loss_net.backward()
            self.optimizer.step()
            loss = loss_net.item()
            i = i + 1
        with torch.no_grad():
            if self.HR:
                pointsReconstructed = self.network.decode_full(input_param)  # forward pass
                pointsReconstructed = pointsReconstructed.view(pointsReconstructed.size(0), -1, 3)

            else:
                pointsReconstructed = self.network.decode(input_param)  # forward pass
                pointsReconstructed = pointsReconstructed.view(pointsReconstructed.size(0), -1, 3)


        print(f"loss reg : {loss} after {i} iterations")
        return pointsReconstructed

    def run(self, input, scalefactor, path):
        """
        :param input: input mesh to reconstruct optimally.
        :return: final reconstruction after optimisation
        """

        input, translation = my_utils.center(input)
        if not self.HR:
            mesh_ref = self.mesh_ref_LR
        else:
            mesh_ref = self.mesh_ref

        ## Extract points and put them on GPU
        points = input.vertices
        # TODO : remove random here
        random_sample = np.random.choice(np.shape(points)[0], size=10000)

        points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
        points = points.transpose(2, 1).contiguous()
        points = points.cuda()

        # Get a low resolution PC to find the best reconstruction after a rotation on the Y axis
        if self.LR_input:
            print("Using a Low_res input")
            points_LR = torch.from_numpy(input.vertices[random_sample].astype(np.float32)).contiguous().unsqueeze(0)
        else:
            print("Using a High_res input")
            points_LR = torch.from_numpy(input.vertices.astype(np.float32)).contiguous().unsqueeze(0)

        input_LR_mesh = trimesh.Trimesh(vertices=(points_LR.squeeze().data.cpu().numpy() + translation) / scalefactor,
                                        faces=np.array([1, 2, 3]), process=False)
        if self.save_path is None:
            input_LR_mesh.export(path[:-4] + "DownsampledInput.ply")
        else:
            input_LR_mesh.export(os.path.join(self.save_path, path[-8:-4] + "DownsampledInput.ply"))

        points_LR = points_LR.transpose(2, 1).contiguous()
        points_LR = points_LR.cuda()

        theta = 0
        bestLoss = 10
        pointsReconstructed = self.network(points_LR)
        pointsReconstructed = pointsReconstructed.view(pointsReconstructed.size(0), -1, 3)
        dist1, dist2 = distChamfer(points_LR.transpose(2, 1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        print("loss without rotation: ", loss_net.item(),
              0)  # ---- Search best angle for best reconstruction on the Y axis---

        x = np.linspace(-np.pi / 2, np.pi / 2, self.num_angles)
        y = np.linspace(-np.pi / 4, np.pi / 4, self.num_angles // 4)

        THETA, PHI = np.meshgrid(x, y)
        Z = np.ndarray([THETA.shape[0], THETA.shape[1]])
        for j in range(THETA.shape[1]):
            for i in range(THETA.shape[0]):
                if self.num_angles == 1:
                    theta = 0
                    phi = 0
                theta = THETA[i, j]
                phi = PHI[i, j]

                #  Rotate mesh by theta and renormalise
                rot_matrix = np.array(
                    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0, np.cos(theta)]])
                rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
                rot_matrix = torch.matmul(torch.from_numpy(np.array(
                    [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1], ])).float().cuda(),
                                          rot_matrix)
                points2 = torch.matmul(rot_matrix, points_LR)
                mesh_tmp = trimesh.Trimesh(process=False, use_embree=False,
                                           vertices=points2[0].transpose(1, 0).data.cpu().numpy(),
                                           faces=self.network.template[0].mesh.faces)
                # bbox
                bbox = np.array([[np.max(mesh_tmp.vertices[:, 0]), np.max(mesh_tmp.vertices[:, 1]),
                                  np.max(mesh_tmp.vertices[:, 2])],
                                 [np.min(mesh_tmp.vertices[:, 0]), np.min(mesh_tmp.vertices[:, 1]),
                                  np.min(mesh_tmp.vertices[:, 2])]])
                norma = torch.from_numpy((bbox[0] + bbox[1]) / 2).float().cuda()

                norma2 = norma.unsqueeze(1).expand(3, points2.size(2)).contiguous()
                points2[0] = points2[0] - norma2

                # reconstruct rotated mesh
                pointsReconstructed = self.network(points2)
                pointsReconstructed = pointsReconstructed.view(pointsReconstructed.size(0), -1, 3)
                dist1, dist2 = distChamfer(points2.transpose(2, 1).contiguous(), pointsReconstructed)

                loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
                Z[i, j] = loss_net.item()
                if loss_net < bestLoss:
                    print(theta, phi, loss_net)
                    bestLoss = loss_net
                    best_theta = theta
                    best_phi = phi
                    # unrotate the mesh
                    norma3 = norma.unsqueeze(0).expand(pointsReconstructed.size(1), 3).contiguous()
                    pointsReconstructed[0] = pointsReconstructed[0] + norma3
                    rot_matrix = np.array(
                        [[np.cos(-theta), 0, np.sin(-theta)], [0, 1, 0], [- np.sin(-theta), 0, np.cos(-theta)]])
                    rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
                    rot_matrix = torch.matmul(rot_matrix, torch.from_numpy(np.array(
                        [[np.cos(-phi), np.sin(-phi), 0], [-np.sin(-phi), np.cos(-phi), 0],
                         [0, 0, 1], ])).float().cuda())
                    pointsReconstructed = torch.matmul(pointsReconstructed, rot_matrix.transpose(1, 0))
                    bestPoints = pointsReconstructed

        try:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(THETA, PHI, -Z, rstride=1, cstride=1,
                            cmap='magma', edgecolor='none', alpha=0.8)

            ax.set_xlabel('THETA', fontsize=20)
            ax.set_ylabel('PHI', fontsize=20)
            ax.set_zlabel('CHAMFER', fontsize=20)
            ax.scatter(best_theta, best_phi, -bestLoss.item(), marker='*', c="red", s=100, alpha=1)
            ax.scatter(best_theta, best_phi, np.min(-Z), marker='*', c="red", s=100, alpha=1)
            ax.view_init(elev=45., azim=45)
            plt.savefig("3Dcurve.png")
            if self.save_path is not None:
                plt.savefig(os.path.join(self.save_path, path[-8:-4] + "3Dcurve.png"))
            else:
                plt.savefig(path[:-4] + "3Dcurve.png")

        except:
            pass

        print("best loss and theta and phi : ", bestLoss.item(), best_theta, best_phi)

        if self.HR:
            faces_tosave = self.network.template[0].mesh_HR.faces
        else:
            faces_tosave = self.network.template[0].mesh.faces

        # create initial guess
        mesh = trimesh.Trimesh(vertices=(bestPoints[0].data.cpu().numpy() + translation) / scalefactor,
                               faces=self.network.template[0].mesh.faces, process=False)
        try:
            plt.plot(X, Y)
            plt.savefig("curve.png")
        except:
            pass
        # START REGRESSION
        print("start regression...")

        # rotate with optimal angle
        rot_matrix = np.array(
            [[np.cos(best_theta), 0, np.sin(best_theta)], [0, 1, 0], [- np.sin(best_theta), 0, np.cos(best_theta)]])
        rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        points2 = torch.matmul(rot_matrix, points)
        mesh_tmp = trimesh.Trimesh(vertices=points2[0].transpose(1, 0).data.cpu().numpy(),
                                   faces=self.network.template[0].mesh.faces, process=False)
        bbox = np.array(
            [[np.max(mesh_tmp.vertices[:, 0]), np.max(mesh_tmp.vertices[:, 1]), np.max(mesh_tmp.vertices[:, 2])],
             [np.min(mesh_tmp.vertices[:, 0]), np.min(mesh_tmp.vertices[:, 1]), np.min(mesh_tmp.vertices[:, 2])]])
        norma = torch.from_numpy((bbox[0] + bbox[1]) / 2).float().cuda()
        norma2 = norma.unsqueeze(1).expand(3, points2.size(2)).contiguous()
        points2[0] = points2[0] - norma2
        pointsReconstructed1 = self.regress(points2)
        # unrotate with optimal angle
        norma3 = norma.unsqueeze(0).expand(pointsReconstructed1.size(1), 3).contiguous()
        rot_matrix = np.array(
            [[np.cos(-best_theta), 0, np.sin(-best_theta)], [0, 1, 0], [- np.sin(-best_theta), 0, np.cos(-best_theta)]])
        rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        pointsReconstructed1[0] = pointsReconstructed1[0] + norma3
        pointsReconstructed1 = torch.matmul(pointsReconstructed1, rot_matrix.transpose(1, 0))

        # create optimal reconstruction
        meshReg = trimesh.Trimesh(vertices=(pointsReconstructed1[0].data.cpu().numpy() + translation) / scalefactor,
                                  faces=faces_tosave, process=False)

        print("... Done!")
        return mesh, meshReg

    def save(self, mesh, mesh_color, path, red, green, blue):
        """
        Home-made function to save a ply file with colors. A bit hacky
        """
        to_write = mesh.vertices
        b = np.zeros((len(mesh.faces), 4)) + 3
        b[:, 1:] = np.array(mesh.faces)
        try:
            points2write = pd.DataFrame({
                'lst0Tite': to_write[:, 0],
                'lst1Tite': to_write[:, 1],
                'lst2Tite': to_write[:, 2],
                'lst3Tite': red,
                'lst4Tite': green,
                'lst5Tite': blue,
            })
            ply.write_ply(filename=path, points=points2write, as_text=True, text=False, faces=pd.DataFrame(b.astype(int)),
                          color=True)
        except:
            points2write = pd.DataFrame({
                'lst0Tite': to_write[:, 0],
                'lst1Tite': to_write[:, 1],
                'lst2Tite': to_write[:, 2],
            })
            ply.write_ply(filename=path, points=points2write, as_text=True, text=False, faces=pd.DataFrame(b.astype(int)),
                          color=False)
    def reconstruct(self, input_p):
        """
        Recontruct a 3D shape by deforming a template
        :param input_p: input path
        :return: None (but save reconstruction)
        """
        print("Reconstructing ", input_p)
        input = trimesh.load(input_p, process=False)
        scalefactor = 1.0
        if self.scale:
            input, scalefactor = my_utils.scale(input,
                                                self.mesh_ref_LR)  # scale input to have the same volume as mesh_ref_LR
        if self.clean:
            input = my_utils.clean(input)  # remove points that doesn't belong to any edges
        my_utils.test_orientation(input)
        mesh, meshReg = self.run(input, scalefactor, input_p)

        if not self.HR:
            red = self.red_LR
            green = self.green_LR
            blue = self.blue_LR
            mesh_ref = self.mesh_ref_LR
        else:
            blue = self.blue_HR
            red = self.red_HR
            green = self.green_HR
            mesh_ref = self.mesh_ref

        if self.save_path is None:
            self.save(mesh, self.mesh_ref_LR, input_p[:-4] + "InitialGuess.ply", self.red_LR, self.green_LR,
                      self.blue_LR)
            self.save(meshReg, mesh_ref, input_p[:-4] + "FinalReconstruction.ply", red, green, blue)
        else:
            self.save(mesh, self.mesh_ref_LR, os.path.join(self.save_path, input_p[-8:-4] + "InitialGuess.ply"),
                      self.red_LR, self.green_LR, self.blue_LR)
            self.save(meshReg, mesh_ref, os.path.join(self.save_path, input_p[-8:-4] + "FinalReconstruction.ply"), red,
                      green, blue)

        # Save optimal reconstruction


if __name__ == '__main__':
    if not os.path.exists("learning_elementary_structure_trained_models/0point_translation/network.pth"):
        os.system("chmod +x ./inference/download_trained_models.sh")
        os.system("./inference/download_trained_models.sh")
    opt = argument_parser.parser()
    my_utils.plant_seeds(randomized_seed=opt.randomize)
    trainer = trainer.Trainer(opt)
    trainer.build_network()
    my_utils.plant_seeds(randomized_seed=opt.randomize)
    inf = Inference(HR=opt.HR, reg_num_steps=opt.reg_num_steps, num_points=opt.number_points,
                    num_angles=opt.num_angles, clean=opt.clean, scale=opt.scale,
                    project_on_target=opt.project_on_target, LR_input=opt.LR_input, save_path=opt.dir_name, network=trainer.network)

    inf.forward(opt.inputA, opt.inputB)
