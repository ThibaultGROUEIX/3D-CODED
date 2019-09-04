from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import numpy as np
from utils import *
import pymesh

mypath = ''
class FAUST(data.Dataset):
    def __init__(self, train, rootpc = "./data/MPI-FAUST/", npoints = 2500, correspondance=False):
        self.train = train
        self.rootpc = rootpc
        self.npoints = npoints
        self.datapath = []
        self.dataname = []
        self.datapathreg = []
        self.datapathregval = []
        self.datapathtxt = []
        self.datapathtxtval = []
        self.datanamereg = []
        self.datanameregval = []
        self.datanameregval = []
        self.correspondance = correspondance
        if not self.correspondance:
            if self.train:
                dir_ply  = os.path.join(self.rootpc, "training")
            else:
                dir_ply  = os.path.join(self.rootpc, "test")

            fns_ply = sorted(os.listdir(os.path.join(dir_ply, "scans_processed")))
            for fn in fns_ply:
                self.datapath.append(os.path.join(dir_ply, "scans_processed") + "/" + fn)
            for fn in fns_ply:
                self.dataname.append(fn)
            if self.train:
                fns_ply = sorted(os.listdir(os.path.join(dir_ply, "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathreg.append(os.path.join(dir_ply, "registrations") + "/" + fn)
                        self.datapathtxt.append(os.path.join(dir_ply, "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanamereg.append(fn)
                fns_ply = sorted(os.listdir(os.path.join(os.path.join(self.rootpc, "val"), "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathregval.append(os.path.join(os.path.join(self.rootpc, "val"), "registrations") + "/" + fn)
                        self.datapathtxtval.append(os.path.join(os.path.join(self.rootpc, "val"), "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanameregval.append(fn)
        else:
            if self.train:
                dir_ply  = os.path.join(self.rootpc, "training")
            else:
                dir_ply  = os.path.join(self.rootpc, "test")

            fns_ply = sorted(os.listdir(os.path.join(dir_ply, "scans_processed")))
            for fn in fns_ply:
                if (fn[0:6]=="tr_reg") and int(fn[7:10])>=80:
                    self.datapath.append(os.path.join(dir_ply, "scans_processed") + "/" + fn)
            for fn in fns_ply:
                if (fn[0:6]=="tr_reg") and int(fn[7:10])>=80:
                    self.dataname.append(fn)
            if self.train:
                fns_ply = sorted(os.listdir(os.path.join(dir_ply, "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathreg.append(os.path.join(dir_ply, "registrations") + "/" + fn)
                        self.datapathtxt.append(os.path.join(dir_ply, "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanamereg.append(fn)
                fns_ply = sorted(os.listdir(os.path.join(os.path.join(self.rootpc, "val"), "registrations")))
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datapathregval.append(os.path.join(os.path.join(self.rootpc, "val"), "registrations") + "/" + fn)
                        self.datapathtxtval.append(os.path.join(os.path.join(self.rootpc, "val"), "txt") + "/" + fn)
                for fn in fns_ply:
                    if fn.endswith(".ply"):
                        self.datanameregval.append(fn)


    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn) as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    try:
                        lenght = int(line.split()[2])
                    except ValueError:
                        lenght = -1
                    if lenght > 0:
                        break

                elif i == 3:
                    try:
                        print("trying line 3")
                        lenght = int(line.split()[2])
                    except ValueError:
                        print(fn)
                        print(line)
                        lenght = -1
                    break
        idx = np.random.randint(6890, size= self.npoints)
        for i in range(15):
            try:
                if self.dataname[index][3:6]=="reg":
                    point_set, idx = my_get_n_random_lines_reg(fn, n = self.npoints)
                else:
                    mystring = my_get_n_random_lines(fn, n = self.npoints)
                    point_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)
        point_set = torch.from_numpy(point_set)
        return point_set.contiguous(), self.dataname[index], idx




    def __len__(self):
        return len(self.datapath)



if __name__  == '__main__':
    d  =  FAUST(train=True, correspondance=True)
    a = len(d)
    print(a)
    a,b, idx = d[15]
    print(idx)
    print(b)
    print(a.size())
    d  =  FAUST(train=False)
    a =  len(d)
    print(a)
