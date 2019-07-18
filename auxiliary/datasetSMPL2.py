from __future__ import print_function
import torch.utils.data as data
import torch
from utils import *
import numpy as np
import pymesh


class SMPL(data.Dataset):
    def __init__(self, train,  npoints=2500, regular = False, rot= False):
        self.rot = rot # apply random rotations in the dataloader
        self.train = train
        self.regular_sampling = regular # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints
        # PATH  : YOU PROBABLY NEED TO EDIT THOSE
        if self.train:
            self.path = "./data/dataset-surreal/"
            self.path_2 = "./data/dataset-bent/"
            self.path_3 = "mypath" #you can add you own generated data if you want (edit len(dataset))
        else:
            self.path = "./data/dataset-surreal-val/"

        # template
        self.mesh = pymesh.load_mesh("./data/template/template.ply")
        a = self.mesh.vertices[self.mesh.faces[:, 0]]
        b = self.mesh.vertices[self.mesh.faces[:, 1]]
        c = self.mesh.vertices[self.mesh.faces[:, 2]]
        cross = np.cross(a - b, a - c)
        area = np.sqrt(np.sum(cross ** 2, axis=1))
        prop = np.zeros((6890))
        prop[self.mesh.faces[:, 0]] = prop[self.mesh.faces[:, 0]] + area
        prop[self.mesh.faces[:, 1]] = prop[self.mesh.faces[:, 1]] + area
        prop[self.mesh.faces[:, 2]] = prop[self.mesh.faces[:, 2]] + area
        prop = prop / np.sum(prop)
        self.prop = prop # prop is the sum of adjacent area of each vertex divided by total area.

    def __getitem__(self, index):
        try:
            if index < 200000:
                input = pymesh.load_mesh(self.path + str(index) + ".ply")
            elif index<230000:
                input = pymesh.load_mesh(self.path_2 + str(index-200000) + ".ply")
            else: #never happens
                print("Never happens")
                input = pymesh.load_mesh(self.path_3 + str(index%70000) + ".ply")
        except:
            print("error loading")
            print(index)
            print(self.path)
            input = pymesh.load_mesh(self.path + str(0) + ".ply")
        points = input.vertices
        if self.rot:
            # generate random sample on the sphere :
            theta = np.random.uniform(- np.pi, np.pi)
            rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0,  np.cos(theta)]])
            # Uncomment these lines to get a uniform 3D rotation of the sphere.
            # x = torch.Tensor(2)
            # x.uniform_()
            # p = torch.Tensor([[np.cos(np.pi  * 2 * x[0] )* np.sqrt(x[1]), (np.random.binomial(1, 0.5, 1)[0]*2 -1) * np.sqrt(1-x[1]), np.sin(np.pi  * 2 * x[0]) * np.sqrt(x[1])]])
            # z = torch.Tensor([[0,1,0]])
            # v = (p-z)/(p-z).norm()
            # H = torch.eye(3) - 2*torch.matmul( v.transpose(1,0), v)
            # rot_matrix = - H.numpy().dot( rot_matrix)
            points = points.dot(np.transpose(rot_matrix, (1, 0)))

        #end random rotation
        points = points - (input.bbox[0] + input.bbox[1]) / 2
        points = torch.from_numpy(points.astype(np.float32)).contiguous()
        if self.train:
            a = torch.FloatTensor(3)
            points = points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)
        random_sample = np.random.choice(6890, size=2500, p=self.prop)
        print(self.prop)
        if self.regular_sampling:
            points = points[random_sample]
        return points, random_sample, index


    def __len__(self):
        if self.train:
            return 230000
        else:
            return 200


if __name__ == '__main__':
    import random
    manualSeed = 1#random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    print('Testing Shapenet dataset')
    d = SMPL(train=True, regular=True)
    a,b,c,   = d[0]
    print(a,b,c)
    min_vals = torch.min(a, 0)[0]
    max_vals = torch.max(a, 0)[0]
    print(min_vals)
    print(max_vals)
