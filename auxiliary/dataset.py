from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import trimesh
import pointcloud_processor
import time
import os


def unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)


class SURREAL(data.Dataset):
    def __init__(self, train, npoints=2500, regular_sampling=False, normal=False, data_augmentation_Z_rotation=False,
                 data_augmentation_Z_rotation_range=360, data_augmentation_3D_rotation=False):

        self.data_augmentation_Z_rotation = data_augmentation_Z_rotation
        self.data_augmentation_Z_rotation_range = data_augmentation_Z_rotation_range
        self.data_augmentation_3D_rotation = data_augmentation_3D_rotation
        self.normal = normal
        self.train = train
        self.regular_sampling = regular_sampling  # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints

        self.datas = []
        start = time.time()
        if not os.path.exists("./data/datas_surreal_train.pth"):
            os.system("chmod +x ./data/download_dataset.sh")
            os.system("./data/download_dataset.sh")
            os.system("mv *.pth data/")

        if self.train:
            self.datas = torch.load("./data/datas_surreal_train.pth")
        else:
            self.datas = torch.load("./data/datas_surreal_test.pth")

        end = time.time()
        print("Ellapsed time to load dataset: ", end - start)
        # template
        if not os.path.exists("./data/template/template.ply"):
            os.system("chmod +x ./data/download_template.sh")
            os.system("./data/download_template.sh")

        self.mesh = trimesh.load("./data/template/template.ply", process=False)
        self.prop = pointcloud_processor.get_vertex_normalised_area(self.mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"



    def __getitem__(self, index):
        # LOAD a training sample
        points = self.datas[index].squeeze()

        # Clone it to keep the cached data safe
        points = points.clone()

        rot_matrix = 0
        # apply random rotation of Z axis
        if self.data_augmentation_Z_rotation:
            # Uniform random Rotation of axis Y
            points, rot_matrix = pointcloud_processor.uniform_rotation_axis(points, axis=1, normals=self.normal,
                                                                            range_rot=self.data_augmentation_Z_rotation_range)
            points, _, _ = pointcloud_processor.center_bounding_box(points)

        # apply random 3D rotation
        if self.data_augmentation_3D_rotation:
            #  Uniform random 3D rotation of the sphere.
            points, rot_matrix = pointcloud_processor.uniform_rotation_sphere(points, normals=self.normal)
            points, _, _ = pointcloud_processor.center_bounding_box(points)

        # Add small random translation
        if self.train:
            points = pointcloud_processor.add_random_translation(points, scale=0.03)

        # Resample according to triangles area
        random_sample = 0

        if self.regular_sampling:
            random_sample = np.random.choice(6890, size=self.npoints, p=self.prop)
            points = points[random_sample]

        return points, random_sample, rot_matrix, index


    def __len__(self):
        if self.train:
            return 230000
        else:
            return 200


if __name__ == '__main__':
    import random

    manualSeed = 1  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    print('Testing Shapenet dataset')
    d = SURREAL(train=True, regular_sampling=True, data_augmentation_3D_rotation=False)
    a, b, c, d = d[0]
    print(a, b, c, d)
    min_vals = torch.min(a, 0)[0]
    max_vals = torch.max(a, 0)[0]
    print(min_vals)
    print(max_vals)
