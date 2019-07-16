from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import trimesh

#UTILITIES
class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2*self.th(self.conv4(x))
        return x


class AE_AtlasNet_Humans(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        translation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - translation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        translation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - translation

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

    def forward2(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = np.random.randint(self.num_vertex, size= x.size(0) * self.num_points)
            rand_grid = self.vertex[idx, : ].view(x.size(0), self.num_points, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode_full(self, x):
        outs = []
        div = 20
        batch = int(self.num_vertex_HR/div)
        for i in range(div-1):
            rand_grid = self.vertex_HR[batch*i:batch*(i+1)].view(x.size(0), batch, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
        

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.vertex[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x):
        x = self.encoder(x)
        outs = []

        rand_grid = self.vertex[:int(self.num_vertex/2)].view(x.size(0), int(self.num_vertex/2), 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        rand_grid = self.vertex[int(self.num_vertex/2):].view(x.size(0), self.num_vertex  - int(self.num_vertex/2), 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class AE_AtlasNet_Animal(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Animal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template_hyppo.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_hyppo.ply", process=False)
        self.mesh_HR = mesh_HR

        point_set = mesh.vertices
        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        translation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - translation
        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        translation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - translation

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

    def forward2(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = np.random.randint(self.num_vertex, size= x.size(0) * self.num_points)
            rand_grid = self.vertex[idx, : ].view(x.size(0), self.num_points, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode_full(self, x):
        outs = []
        div = 20
        batch = int(self.num_vertex_HR/div)
        for i in range(div-1):
            rand_grid = self.vertex_HR[batch*i:batch*(i+1)].view(x.size(0), batch, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.vertex[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x):
        x = self.encoder(x)
        outs = []

        rand_grid = self.vertex[:int(self.num_vertex/2)].view(x.size(0), int(self.num_vertex/2), 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        rand_grid = self.vertex[int(self.num_vertex/2):].view(x.size(0), self.num_vertex  - int(self.num_vertex/2), 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


if __name__ == '__main__':
    print("test")
