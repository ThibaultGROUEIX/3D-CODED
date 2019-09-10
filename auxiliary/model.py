from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from my_utils import sampleSphere
import trimesh
import pointcloud_processor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

"""
Template Discovery -> Patch deform
Tamplate learning -> Point translation 
"""


class PointNetfeat(nn.Module):
    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze()).unsqueeze(-1)))
        return x.squeeze()


class patchDeformationMLP(nn.Module):
    """ Deformation of a 2D patch into a 3D surface """

    def __init__(self, patchDim=2, patchDeformDim=3, tanh=True):

        super(patchDeformationMLP, self).__init__()
        layer_size = 128
        self.tanh = tanh
        self.conv1 = torch.nn.Conv1d(patchDim, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, patchDeformDim, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size)
        self.bn2 = torch.nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        print("bottleneck_size", bottleneck_size)
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2 * self.th(self.conv4(x))
        return x


class GetTemplate(object):
    def __init__(self, start_from, dataset_train=None):
        if start_from == "TEMPLATE":
            self.init_template()
        elif start_from == "SOUP":
            self.init_soup()
        elif start_from == "TRAININGDATA":
            self.init_trainingdata(dataset_train)
        else:
            print("select valid template type")

    def init_template(self):
        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        point_set = mesh.vertices
        point_set, _, _ = pointcloud_processor.center_bounding_box(point_set)

        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set_HR = mesh_HR.vertices
        point_set_HR, _, _ = pointcloud_processor.center_bounding_box(point_set_HR)

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"
        self.prop = torch.from_numpy(self.prop).cuda().unsqueeze(0).float()
        print(f"Using template to initialize template")

    def init_soup(self):
        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh  # Load this anyway to keep access to edge information
        self.vertex = torch.FloatTensor(6890, 3).normal_().cuda()
        self.vertex_HR = self.vertex.clone()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(f"Using Random soup to initialize template")

    def init_trainingdata(self, dataset_train=None):
        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh  # Load this anyway to keep access to edge information
        index = np.random.randint(len(dataset_train))
        points = dataset_train.datas[index].squeeze().clone()
        self.vertex = points
        self.vertex_HR = self.vertex.clone()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(f"Using training data number {index} to initialize template")


class PrimSelect(nn.Module):
    def __init__(self):
        super(PrimSelect, self).__init__(bottleneck_size=1024, nb_primitives=1)
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.prim_network = nn.Sequential(
                nn.Linear(bottleneck_size, self.bottleneck_size // 2),
                nn.BatchNorm1d(self.bottleneck_size // 2),
                nn.ReLU(),
                nn.Linear(bottleneck_size // 2, nb_primitives),
                nn.BatchNorm1d(nb_primitives),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.prim_network(x)


class AE_AtlasNet_Humans(nn.Module):
    def __init__(self, num_points=6890, bottleneck_size=1024, nb_primitives=1, point_translation=False, dim_template=3,
                 patch_deformation=False, dim_out_patch=3, start_from="TEMPLATE", dataset_train=None, primitive_selection=False):
        super(AE_AtlasNet_Humans, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.point_translation = point_translation
        self.dim_template = dim_template
        self.patch_deformation = patch_deformation
        self.dim_out_patch = dim_out_patch
        self.dim_before_decoder = 3
        self.count = 0
        self.start_from = start_from
        self.dataset_train = dataset_train
        self.primitive_selection = primitive_selection

        self.template = [GetTemplate(start_from, dataset_train) for i in range(0, self.nb_primitives)]
        if point_translation:
            if dim_template > 3:
                for i in range(0, self.nb_primitives):
                    self.template[i].vertex = torch.cat([self.template[i].vertex, torch.zeros(
                        (self.template[i].vertex.size(0), self.dim_template - 3)).cuda()], -1)
                self.dim_before_decoder = dim_template

            for i in range(0, self.nb_primitives):
                self.template[i].vertex = torch.nn.Parameter(self.template[i].vertex)
                self.register_parameter("template_vertex_" + str(i), self.template[i].vertex)

        if patch_deformation:
            self.dim_before_decoder = dim_out_patch
            self.templateDiscovery = nn.ModuleList(
                [patchDeformationMLP(patchDim=dim_template, patchDeformDim=dim_out_patch, tanh=True) for i in
                 range(0, self.nb_primitives)])

        if self.primitive_selection:
            self.prim_network = PrimSelect(bottleneck_size, nb_primitives)

        self.encoder = PointNetfeat(num_points, bottleneck_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.dim_before_decoder + self.bottleneck_size) for i in
             range(0, self.nb_primitives)])

    def morph_points(self, x, idx=None, prim=False):
        outs = []
        if not idx is None:
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
        for i in range(0, self.nb_primitives):
            rand_grid = self.template[i].vertex  # 6890, 3
            if not idx is None:
                rand_grid = rand_grid[idx, :]  # batch x 2500, 3
                rand_grid = rand_grid.view(x.size(0), -1, self.dim_template).transpose(1,
                                                                                       2).contiguous()  # batch , 2500, 3 -> batch, 6980, 3
            else:
                rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0).expand(x.size(0), self.dim_template,
                                                                                       -1)  # 3, 6980 -> 1,3,6980 -> batch, 3, 6980

            # rand_grid = Variable(rand_grid) I have removed this line t make sure gradients gets back to self.template.vertex even with forward idx
            if self.patch_deformation:
                rand_grid = self.templateDiscovery[i](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        if prim:
            return outs#, self.prim_network(x)
        else:
            return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def decode(self, x, idx=None):
        return self.morph_points(x, idx)

    def forward(self, x, idx=None):
        x = self.encoder(x)
        return self.decode(x, idx)

    def decode_full(self, x):
        """
        This function only works if nb_primitives == 1. TODO : make it generic
        :param x:
        :return:
        """
        outs = []
        div = 20
        batch = int(self.template.num_vertex_HR / div)
        for i in range(div - 1):
            rand_grid = self.template[0].vertex_HR[batch * i:batch * (i + 1)].view(x.size(0), batch,
                                                                                   self.dim_template).transpose(1,
                                                                                                                2).contiguous()
            if self.patch_deformation:
                rand_grid = self.templateDiscovery[i](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.template[0].vertex_HR[batch * i:].view(x.size(0), -1, self.dim_template).transpose(1,
                                                                                                            2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def get_points_translation_template(self):
        templates = []
        for i in range(0, self.nb_primitives):
            templates.append(self.template[i].vertex)
        return templates

    def get_patch_deformation_template(self):
        templates = []
        for i in range(0, self.nb_primitives):
            rand_grid = self.template[i].vertex.transpose(0, 1).contiguous().unsqueeze(0).expand(1, self.dim_template,
                                                                                                 -1)
            templates.append(self.templateDiscovery[i](rand_grid).squeeze().transpose(1, 0).contiguous())
        return templates

    def save_template_png(self, path):
        if self.point_translation:
            templates = self.get_points_translation_template()
            if self.dim_template == 3:
                for i in range(0, self.nb_primitives):
                    p1 = templates[i][:, 0].cpu().clone().detach().numpy()
                    p2 = templates[i][:, 1].cpu().clone().detach().numpy()
                    p3 = templates[i][:, 2].cpu().clone().detach().numpy()
                    fig = plt.figure(figsize=(20, 20), dpi=80)
                    fig.set_size_inches(20, 20)
                    ax = fig.add_subplot(111, projection='3d', facecolor='white')
                    # ax = fig.add_subplot(111, projection='3d',  facecolor='#202124')
                    ax.view_init(0, 30)
                    ax.set_xlim3d(-0.8, 0.8)
                    ax.set_ylim3d(-0.8, 0.8)
                    ax.set_zlim3d(-0.8, 0.8)
                    # ax.set_xlabel('X Label')
                    # ax.set_ylabel('Y Label')
                    # ax.set_zlabel('Z Label')
                    ax.scatter(p3, p1, p2, alpha=1, s=10, c='salmon', edgecolor='orangered')
                    plt.grid(b=None)
                    plt.axis('off')
                    fig.savefig(os.path.join(path, "points_" + str(i) + "_" + str(self.count)), bbox_inches='tight',
                                pad_inches=0)
            else:
                print("can't save png if dim template is not 3!")
        if self.patch_deformation:
            templates = self.get_patch_deformation_template()
            if self.dim_template == 3:
                for i in range(0, self.nb_primitives):
                    p1 = templates[i][:, 0].cpu().clone().detach().numpy()
                    p2 = templates[i][:, 1].cpu().clone().detach().numpy()
                    p3 = templates[i][:, 2].cpu().clone().detach().numpy()
                    fig = plt.figure(figsize=(20, 20), dpi=80)
                    fig.set_size_inches(20, 20)
                    ax = fig.add_subplot(111, projection='3d', facecolor='white')
                    # ax = fig.add_subplot(111, projection='3d',  facecolor='#202124')
                    ax.view_init(0, 30)
                    ax.set_xlim3d(-0.8, 0.8)
                    ax.set_ylim3d(-0.8, 0.8)
                    ax.set_zlim3d(-0.8, 0.8)
                    # ax.set_xlabel('X Label')
                    # ax.set_ylabel('Y Label')
                    # ax.set_zlabel('Z Label')
                    ax.scatter(p3, p1, p2, alpha=1, s=10, c='salmon', edgecolor='orangered')
                    plt.grid(b=None)
                    plt.axis('off')
                    fig.savefig(os.path.join(path, "deformation_" + str(i) + "_" + str(self.count)),
                                bbox_inches='tight', pad_inches=0)
            else:
                print("can't save png if dim template is not 3!")
        self.count += 1


if __name__ == '__main__':
    a = AE_AtlasNet_Humans()
