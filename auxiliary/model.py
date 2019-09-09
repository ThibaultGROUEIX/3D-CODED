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
        batchsize = x.size()[0]
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
        batchsize = x.size()[0]
        # print(x.size())
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
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.prop = pointcloud_processor.get_vertex_normalised_area(mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"
        self.prop = torch.from_numpy(self.prop).cuda().unsqueeze(0).float()

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
        # print(grid.sizegrid())
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        rand_grid = self.vertex[int(self.num_vertex/2):].view(x.size(0), self.num_vertex  - int(self.num_vertex/2), 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        # print(grid.sizegrid())
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class TemplateDiscovery(nn.Module):
    """deformation of a 2D patch into a 3D surface"""
    def __init__(self,dim = 3,tanh=True):

        super(TemplateDiscovery, self).__init__()
        layer_size = 128
        self.tanh = tanh
        self.conv1 = torch.nn.Conv1d(3, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, dim, 1)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.tanh:
            x = self.th(self.conv3(x))
        else:
            x = self.conv3(x)
        return x

class AE_AtlasNet_Humans_Parameters(nn.Module):
    def __init__(self, dim=3,num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Parameters, self).__init__()
        self.num_points = num_points
        self.dim = dim
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size =  self.dim +self.bottleneck_size) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.template = torch.from_numpy(point_set).float()
        if dim > 3:
            self.template = torch.cat([self.template,torch.zeros((self.template.size(0), self.dim-3))],-1)

        self.template = torch.nn.Parameter(self.template)
        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.template.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.register_parameter("template", self.template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), self.dim,-1)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.template[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, self.dim).transpose(1,2).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x, deformation):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = deformation.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(rand_grid)
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

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
        rand_grid = self.templateDiscovery[0](rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

class AE_AtlasNet_Humans_Template_Discovery_Parameters(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Template_Discovery_Parameters, self).__init__()
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
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery(tanh=False) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.template = torch.from_numpy(point_set).float().cuda()
        self.bias_template = torch.nn.Parameter(torch.zeros(self.template.size()))
        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.template.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(self.bias_template.type())

        self.register_parameter("bias_template", self.bias_template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            # rand_grid += self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            bias = self.bias_template[idx,:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = self.template[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = self.templateDiscovery[i](rand_grid)# + bias
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x):

        x = self.encoder(x)
        outs = []

        templates = []
        transformation = []
        discovery = []
        bias  = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            templates.append(rand_grid)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            transformation.append(rand_grid.clone())
            rand_grid += self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            print(self.bias_template)
            bias.append(self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1))
            discovery.append(rand_grid.clone())
            # discovery.append(self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1))
            edges.append(self.edges)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))


        return templates, transformation, bias, discovery, edges

class AE_AtlasNet_Sphere_Template_Discovery_Parameters(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Sphere_Template_Discovery_Parameters, self).__init__()
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
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery() for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.template = torch.from_numpy(sampleSphere(num_points)).float().cuda()
        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.template.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        self.bias_template = torch.nn.Parameter(torch.zeros(self.template.size()))
        self.register_parameter("bias_template", self.bias_template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            bias = self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid) + bias
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            bias = self.bias_template[idx,:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = self.template[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = self.templateDiscovery[i](rand_grid) + bias
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x):

        x = self.encoder(x)
        outs = []

        templates = []
        transformation = []
        discovery = []
        bias  = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            templates.append(rand_grid)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            transformation.append(rand_grid.clone())
            rand_grid += self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            print(self.bias_template)
            bias.append(self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1))
            discovery.append(rand_grid.clone())
            # discovery.append(self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1))
            edges.append(self.edges)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))


        return templates, transformation, bias, discovery, edges

class AE_AtlasNet_Humans_Template_Discovery(nn.Module):
    def __init__(self, dim=3, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Template_Discovery, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = dim +self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery(dim,tanh=False) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

        self.bias_template = torch.nn.Parameter(torch.zeros(self.vertex.size()))
        self.register_parameter("bias_template", self.bias_template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):

            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid) + self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)

            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []

        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.vertex[idx,:]
            bias = self.bias_template[idx,:]
            bias = bias.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid) + bias
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            exit()
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
            rand_grid = self.templateDiscovery[0](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        rand_grid = self.templateDiscovery[0](rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

class AE_AtlasNet_Humans_Template2_Discovery(nn.Module):
    def __init__(self, dim=3, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Template2_Discovery, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = dim +self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery(dim,tanh=True) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)


    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):

            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []

        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.vertex[idx,:]
            bias = self.bias_template[idx,:]
            bias = bias.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid) + bias
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            exit()
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
            rand_grid = self.templateDiscovery[0](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        rand_grid = self.templateDiscovery[0](rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class AE_AtlasNet_Humans_Template_Discovery_Sphere(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Template_Discovery_Sphere, self).__init__()
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
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery() for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/sphere.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh_HR.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.template = torch.nn.Parameter(torch.from_numpy(sampleSphere(num_points)).float())
        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.template.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

        self.register_parameter("template", self.template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.template[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = self.templateDiscovery[i](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x, deformation):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = deformation.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

        def decode(self, x):
            outs = []
            for i in range(0,self.nb_primitives):
                rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
                rand_grid = Variable(rand_grid)
                rand_grid = self.templateDiscovery[i](rand_grid)
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
                rand_grid = self.templateDiscovery[0](rand_grid)
                y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
                y = torch.cat( (rand_grid, y), 1).contiguous()
                outs.append(self.decoder[0](y))
                torch.cuda.synchronize()
            i = div - 1
            rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[0](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
            return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class AE_AtlasNet_Humans_Template_Discovery_Sphere_Param(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Template_Discovery_Sphere_Param, self).__init__()
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

        mesh = trimesh.load("./data/template/sphere.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh_HR.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.template = torch.nn.Parameter(torch.from_numpy(sampleSphere(num_points)).float())
        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.template.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

        self.register_parameter("template", self.template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.template[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x, deformation):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = deformation.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

class AE_AtlasNet_Humans_Template_Discovery_Sphere_Param(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_Humans_Template_Discovery_Sphere_Param, self).__init__()
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

        mesh = trimesh.load("./data/template/sphere.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh_HR.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.template = torch.nn.Parameter(torch.from_numpy(sampleSphere(num_points)).float())
        self.template.data.uniform_(-1,1)
        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.template.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

        self.register_parameter("template", self.template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.template[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x, deformation):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = deformation.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

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
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation
        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation
        print(np.shape(mesh.vertices))
        print(np.shape(mesh_HR.vertices))

        self.vertex = torch.from_numpy(point_set).cuda().float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)
        print(self.num_vertex)
        print(self.num_vertex_HR)

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

class sphere_both(nn.Module):
    def __init__(self, dim=3, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(sphere_template, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = dim +self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery(dim,tanh=False) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex = torch.from_numpy(sampleSphere(num_points)).float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

        self.bias_template = torch.nn.Parameter(torch.zeros(self.vertex.size()))
        self.register_parameter("bias_template", self.bias_template)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):

            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid) + self.bias_template.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)

            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []

        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.vertex[idx,:]
            bias = self.bias_template[idx,:]
            bias = bias.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid) + bias
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            exit()
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
            rand_grid = self.templateDiscovery[0](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        rand_grid = self.templateDiscovery[0](rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

class sphere_template(nn.Module):
    def __init__(self, dim=3, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(sphere_template, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = dim +self.bottleneck_size) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex = torch.nn.Parameter(torch.from_numpy(sampleSphere(num_points)).float())
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

        self.register_parameter("template", self.vertex)

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

    def get_template(self, x):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

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

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            exit()
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
            rand_grid = self.templateDiscovery[0](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        rand_grid = self.templateDiscovery[0](rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


class sphere_transform(nn.Module):
    def __init__(self, dim=3, num_points = 6890, bottleneck_size = 1024, nb_primitives = 1):
        super(sphere_transform, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = dim +self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.templateDiscovery = nn.ModuleList([TemplateDiscovery(dim,tanh=False) for i in range(0,self.nb_primitives)])

        mesh = trimesh.load("./data/template/template.ply", process=False)
        self.mesh = mesh
        mesh_HR = trimesh.load("./data/template/template_dense.ply", process=False)
        self.mesh_HR = mesh_HR
        point_set = mesh.vertices
        edge_set = mesh.faces

        bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set = point_set - tranlation

        point_set_HR = mesh_HR.vertices
        bbox = np.array([[np.max(point_set_HR[:,0]), np.max(point_set_HR[:,1]), np.max(point_set_HR[:,2])], [np.min(point_set_HR[:,0]), np.min(point_set_HR[:,1]), np.min(point_set_HR[:,2])]])
        tranlation = (bbox[0] + bbox[1]) / 2
        point_set_HR = point_set_HR - tranlation

        self.edges = torch.from_numpy(edge_set).cuda().int()
        self.vertex = torch.from_numpy(sampleSphere(num_points)).float()
        self.vertex_HR = torch.from_numpy(point_set_HR).cuda().float()
        self.num_vertex = self.vertex.size(0)
        self.num_vertex_HR = self.vertex_HR.size(0)

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):

            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)

            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def get_template(self, x):
        x = self.encoder(x)
        outs = []

        templates = []
        deformations = []
        edges = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,-1)
            rand_grid = Variable(rand_grid)
            edges.append(self.edges)
            templates.append(rand_grid)
            deformations.append(self.templateDiscovery[i](rand_grid))
            # deformations.append(self.templateDiscovery[i](rand_grid))

        return templates, deformations, edges

    def forward_idx(self, x, idx):
        x = self.encoder(x)
        outs = []

        for i in range(0,self.nb_primitives):
            idx = idx.view(-1)
            idx = idx.numpy().astype(np.int)
            rand_grid = self.vertex[idx,:]
            rand_grid = rand_grid.view(x.size(0), -1, 3).transpose(1,2).contiguous()
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = self.vertex.transpose(0,1).contiguous().unsqueeze(0).expand(x.size(0), 3,self.num_points)
            rand_grid = Variable(rand_grid)
            rand_grid = self.templateDiscovery[i](rand_grid)
            exit()
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
            rand_grid = self.templateDiscovery[0](rand_grid)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[0](y))
            torch.cuda.synchronize()
        i = div - 1
        rand_grid = self.vertex_HR[batch*i:].view(x.size(0), -1, 3).transpose(1,2).contiguous()
        rand_grid = Variable(rand_grid)
        rand_grid = self.templateDiscovery[0](rand_grid)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs.append(self.decoder[0](y))
        torch.cuda.synchronize()
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()



if __name__ == '__main__':
    # print('testing our method...')
    # sim_data = Variable(torch.rand(1, 3, 400, 400))
    # model = PointNetAE_RNN_grid2mesh()
    # model.cuda()
    # out = model(sim_data.cuda())
    # print(out.size())

    # print('testing baseline...')
    # sim_data = Variable(torch.rand(1, 3, 400, 400))
    # model = PointNetAEBottleneck()
    # model.cuda()
    # out = model(sim_data.cuda())
    # print(out.size())

    print('testing PointSenGet...')
    sim_data = Variable(torch.rand(1, 4, 192, 256))
    model = Hourglass()
    # model.cuda()
    # out = model(sim_data.cuda())
    out = model(sim_data)
    print(out.size())
