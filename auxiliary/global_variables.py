from model import *

global network
global opt
global mesh_ref
global mesh_ref_LR

# =============Get data and template======================================== #
if not os.path.exists("./data/template/template.ply"):
    os.system('./data/download_template.sh')
# ========================================================== #

# load template at high and low resolution
mesh_ref = trimesh.load("./data/template/template_dense.ply", process=False)
mesh_ref_LR = trimesh.load("./data/template/template.ply", process=False)

#load colors
red_LR = np.load("./data/template/red_LR.npy").astype("uint8")
green_LR = np.load("./data/template/green_LR.npy").astype("uint8")
blue_LR = np.load("./data/template/blue_LR.npy").astype("uint8")
red_HR = np.load("./data/template/red_HR.npy").astype("uint8")
green_HR = np.load("./data/template/green_HR.npy").astype("uint8")
blue_HR = np.load("./data/template/blue_HR.npy").astype("uint8")
