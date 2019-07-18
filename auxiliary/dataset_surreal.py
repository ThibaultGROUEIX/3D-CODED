from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import trimesh
import sys
import pointcloud_processor
from joblib import Parallel, delayed
import time
from collections import defaultdict
import joblib
import os 

def unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)


class SURREAL(data.Dataset):
    def __init__(self, train,  npoints=2500, regular_sampling = False, normal=False, data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range=360, data_augmentation_3D_rotation=False, cache=True):

        self.cache = cache
        self.data_augmentation_Z_rotation = data_augmentation_Z_rotation
        self.data_augmentation_Z_rotation_range = data_augmentation_Z_rotation_range
        self.data_augmentation_3D_rotation = data_augmentation_3D_rotation
        self.normal = normal
        self.train = train
        self.regular_sampling = regular_sampling # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints

        # PATH  : YOU MAY NEED TO EDIT THOSE IF YOU GENERATE YOUR OWN SYNTHETIC MODELS
        # Not used when flag "cache" is True
        if self.train:
            self.path = "./data/dataset-surreal/"
            self.path_2 = "./data/dataset-bent/"
            self.path_3 = "mypath" #you can add you own generated data if you want (edit len(dataset))
        else:
            self.path = "./data/dataset-surreal-val/"

        self.datas  = []
        if self.cache:
            start = time.time()
            if self.train and os.path.exists("./data/datas_surreal_train.pth"):
                self.datas = torch.load("./data/datas_surreal_train.pth")
            elif (not self.train) and os.path.exists("./data/datas_surreal_test.pth"):
                self.datas = torch.load("./data/datas_surreal_test.pth")
            else:
                class BatchCompletionCallBack(object):
                    completed = defaultdict(int)

                    def __init__(se, time, index, parallel):
                        se.index = index
                        se.parallel = parallel

                    def __call__(se, index):
                        BatchCompletionCallBack.completed[se.parallel] += 1
                        if BatchCompletionCallBack.completed[se.parallel] % 100 == 0 :
                            end = time.time()
                            etl = (end - start )*(self.__len__()/float(BatchCompletionCallBack.completed[se.parallel])) - (end - start)
                            print("Progress : %f %% " %
                                  float(BatchCompletionCallBack.completed[se.parallel] * 100 / self.__len__()) + "ETL %d seconds" % int(etl) , end='\r')
                        if se.parallel._original_iterator is not None:
                            se.parallel.dispatch_next()

                joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack
                self.datas = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(unwrap_self)(i) for i in zip([self] * self.__len__(), range(self.__len__())))
                print(" dataset : " + str(len(self.datas)))
                self.datas = torch.cat(self.datas, 0)
                print(self.datas.size())
                if self.train:
                    torch.save(self.datas, "./data/datas_surreal_train.pth")
                else:
                    torch.save(self.datas, "./data/datas_surreal_test.pth")
            end = time.time()
            print("Ellapsed time : " , end - start)
        # template
        self.mesh = trimesh.load("./data/template/template.ply", process=False)
        self.prop = pointcloud_processor.get_vertex_normalised_area(self.mesh)
        assert (np.abs(np.sum(self.prop)- 1)<0.001), "Propabilities do not sum to 1!)" 

    def _getitem(self, index):
        # Load a training sample et center the bounding box
        try:
            if index < 200000:
                input = trimesh.load(self.path + str(index) + ".ply", process=False)
            elif index<230000:
                input = trimesh.load(self.path_2 + str(index-200000) + ".ply", process=False)
            else: #never happens
                print("Should Never Happen")
                input = trimesh.load(self.path_3 + str(index%70000) + ".ply", process=False)
        except:
            print("error loading " + str(index))
            input = trimesh.load(self.path + str(0) + ".ply", process=False)

        points = input.vertices
        points, _, _ = pointcloud_processor.center_bounding_box(points)

        return torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)

    def __getitem__(self, index):
        # LOAD a training sample
        if self.cache:
            points = self.datas[index].squeeze()
        else:
            points = self._getitem(index).squeeze()

        # Clone it to keep the cached data safe
        points = points.clone()

        rot_matrix = 0
        # apply random rotation of Z axis
        if self.data_augmentation_Z_rotation:
            # Uniform random Rotation of axis Y
            points, rot_matrix = pointcloud_processor.uniform_rotation_axis(points, axis=1, normals = self.normal, range_rot = self.data_augmentation_Z_rotation_range)
            points, _, _ = pointcloud_processor.center_bounding_box(points)
 
        # apply random 3D rotation
        if self.data_augmentation_3D_rotation:
            #  Uniform random 3D rotation of the sphere.
            points, rot_matrix = pointcloud_processor.uniform_rotation_sphere(points, normals = self.normal)
            points, _, _ = pointcloud_processor.center_bounding_box(points)

        # Add small random translation
        if self.train:
            points = pointcloud_processor.add_random_translation(points, scale = 0.03)

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

    manualSeed = 1#random.randint(1, 10000)  # fix seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    print('Testing Shapenet dataset')
    d = SURREAL(train=True, cache=True, regular_sampling=True, data_augmentation_3D_rotation=False )
    a,b,c,d   = d[0]
    print(a,b,c,d)
    min_vals = torch.min(a, 0)[0]
    max_vals = torch.max(a, 0)[0]
    print(min_vals)
    print(max_vals)