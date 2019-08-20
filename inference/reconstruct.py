from __future__ import print_function
import numpy as np
import torch.utils.data
import sys
sys.path.append('./auxiliary/')
from model import *
from utils import *
from ply import *
import sys
import torch.optim as optim
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
import global_variables
import trimesh
val_loss = AverageValueMeter()



def regress(points):
    """
    search the latent space to global_variables. Optimize reconstruction using the Chamfer Distance
    :param points: input points to reconstruct
    :return pointsReconstructed: final reconstruction after optimisation
    """
    points = points.data
    latent_code = global_variables.network.encoder(points)
    lrate = 0.001  # learning rate
    # define parameters to be optimised and optimiser
    input_param = nn.Parameter(latent_code.data, requires_grad=True)
    global_variables.optimizer = optim.Adam([input_param], lr=lrate)
    loss = 10
    i = 0

    #learning loop
    while np.log(loss) > -9 and i < global_variables.opt.nepoch:
        global_variables.optimizer.zero_grad()
        pointsReconstructed = global_variables.network.decode(input_param)  # forward pass
        dist1, dist2 = distChamfer(points.transpose(2, 1).contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        global_variables.optimizer.step()
        loss = loss_net.item()
        i = i + 1
    with torch.no_grad():
        if global_variables.opt.HR:
            pointsReconstructed = global_variables.network.decode_full(input_param)  # forward pass
        else :
            pointsReconstructed = global_variables.network.decode(input_param)  # forward pass
    # print("loss reg : ", loss)
    return pointsReconstructed

def run(input, scalefactor):
    """
    :param input: input mesh to reconstruct optimally.
    :return: final reconstruction after optimisation
    """

    input, translation = center(input)
    if not global_variables.opt.HR:
        mesh_ref = global_variables.mesh_ref_LR
    else:
        mesh_ref = global_variables.mesh_ref

    ## Extract points and put them on GPU
    points = input.vertices
    random_sample = np.random.choice(np.shape(points)[0], size=10000)

    points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
    points = points.transpose(2, 1).contiguous()
    points = points.cuda()

    # Get a low resolution PC to find the best reconstruction after a rotation on the Y axis
    points_LR = torch.from_numpy(input.vertices[random_sample].astype(np.float32)).contiguous().unsqueeze(0)
    points_LR = points_LR.transpose(2, 1).contiguous()
    points_LR = points_LR.cuda()

    theta = 0
    bestLoss = 10
    pointsReconstructed = global_variables.network(points_LR)
    dist1, dist2 = distChamfer(points_LR.transpose(2, 1).contiguous(), pointsReconstructed)
    loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
    # print("loss : ",  loss_net.item(), 0)
    # ---- Search best angle for best reconstruction on the Y axis---
    for theta in np.linspace(-np.pi/2, np.pi/2, global_variables.opt.num_angles):
        if global_variables.opt.num_angles == 1:
            theta = 0
        #  Rotate mesh by theta and renormalise
        rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [- np.sin(theta), 0,  np.cos(theta)]])
        rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
        points2 = torch.matmul(rot_matrix, points_LR)
        mesh_tmp = trimesh.Trimesh(process=False, use_embree=False,vertices=points2[0].transpose(1,0).data.cpu().numpy(), faces=global_variables.network.mesh.faces)
        #bbox
        bbox = np.array([[np.max(mesh_tmp.vertices[:,0]), np.max(mesh_tmp.vertices[:,1]), np.max(mesh_tmp.vertices[:,2])], [np.min(mesh_tmp.vertices[:,0]), np.min(mesh_tmp.vertices[:,1]), np.min(mesh_tmp.vertices[:,2])]])
        norma = torch.from_numpy((bbox[0] + bbox[1]) / 2).float().cuda()

        norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
        points2[0] = points2[0] - norma2
        mesh_tmp = trimesh.Trimesh(process=False, use_embree=False,vertices=points2[0].transpose(1,0).data.cpu().numpy(), faces=np.array([[0,0,0]]))

        # reconstruct rotated mesh
        pointsReconstructed = global_variables.network(points2)
        dist1, dist2 = distChamfer(points2.transpose(2, 1).contiguous(), pointsReconstructed)


        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        if loss_net < bestLoss:
            bestLoss = loss_net
            best_theta = theta
            # unrotate the mesh
            norma3 = norma.unsqueeze(0).expand(pointsReconstructed.size(1), 3).contiguous()
            pointsReconstructed[0] = pointsReconstructed[0] + norma3
            rot_matrix = np.array([[np.cos(-theta), 0, np.sin(-theta)], [0, 1, 0], [- np.sin(-theta), 0,  np.cos(-theta)]])
            rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
            pointsReconstructed = torch.matmul(pointsReconstructed, rot_matrix.transpose(1,0))
            bestPoints = pointsReconstructed

    # print("best loss and angle : ", bestLoss.item(), best_theta)
    val_loss.update(bestLoss.item())

    if global_variables.opt.HR:
        faces_tosave = global_variables.network.mesh_HR.faces
    else:
        faces_tosave = global_variables.network.mesh.faces
    
    # create initial guess
    mesh = trimesh.Trimesh(vertices=(bestPoints[0].data.cpu().numpy() + translation)/scalefactor, faces=global_variables.network.mesh.faces, process = False)


    #START REGRESSION
    print("start regression...")
    
    # rotate with optimal angle
    rot_matrix = np.array([[np.cos(best_theta), 0, np.sin(best_theta)], [0, 1, 0], [- np.sin(best_theta), 0,  np.cos(best_theta)]])
    rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
    points2 = torch.matmul(rot_matrix, points)
    mesh_tmp = trimesh.Trimesh(vertices=points2[0].transpose(1,0).data.cpu().numpy(), faces=global_variables.network.mesh.faces, process=False)
    bbox = np.array([[np.max(mesh_tmp.vertices[:,0]), np.max(mesh_tmp.vertices[:,1]), np.max(mesh_tmp.vertices[:,2])], [np.min(mesh_tmp.vertices[:,0]), np.min(mesh_tmp.vertices[:,1]), np.min(mesh_tmp.vertices[:,2])]])
    norma = torch.from_numpy((bbox[0] + bbox[1]) / 2).float().cuda()
    norma2 = norma.unsqueeze(1).expand(3,points2.size(2)).contiguous()
    points2[0] = points2[0] - norma2
    pointsReconstructed1 = regress(points2)
    # unrotate with optimal angle
    norma3 = norma.unsqueeze(0).expand(pointsReconstructed1.size(1), 3).contiguous()
    rot_matrix = np.array([[np.cos(-best_theta), 0, np.sin(-best_theta)], [0, 1, 0], [- np.sin(-best_theta), 0,  np.cos(-best_theta)]])
    rot_matrix = torch.from_numpy(rot_matrix).float().cuda()
    pointsReconstructed1[0] = pointsReconstructed1[0] + norma3
    pointsReconstructed1 = torch.matmul(pointsReconstructed1, rot_matrix.transpose(1,0))
    
    # create optimal reconstruction
    meshReg = trimesh.Trimesh(vertices=(pointsReconstructed1[0].data.cpu().numpy()  + translation)/scalefactor, faces=faces_tosave, process=False)

    print("... Done!")
    return mesh, meshReg

def save(mesh, mesh_color, path, red, green, blue):
    """
    Home-made function to save a ply file with colors. A bit hacky
    """
    to_write = mesh.vertices
    b = np.zeros((len(mesh.faces),4)) + 3
    b[:,1:] = np.array(mesh.faces)
    points2write = pd.DataFrame({
        'lst0Tite': to_write[:,0],
        'lst1Tite': to_write[:,1],
        'lst2Tite': to_write[:,2],
        'lst3Tite': red,
        'lst4Tite': green,
        'lst5Tite': blue,
        })
    write_ply(filename=path, points=points2write, as_text=True, text=False, faces = pd.DataFrame(b.astype(int)), color = True)    
def reconstruct(input_p):
    """
    Recontruct a 3D shape by deforming a template
    :param input_p: input path
    :return: None (but save reconstruction)
    """
    input = trimesh.load(input_p, process=False)
    scalefactor = 1.0
    if global_variables.opt.scale:
        input, scalefactor = scale(input, global_variables.mesh_ref_LR) #scale input to have the same volume as mesh_ref_LR
    if global_variables.opt.clean:
        input = clean(input) #remove points that doesn't belong to any edges
    test_orientation(input)
    mesh, meshReg = run(input, scalefactor)

    if not global_variables.opt.HR:
        red = global_variables.red_LR
        green = global_variables.green_LR
        blue = global_variables.blue_LR
        mesh_ref = global_variables.mesh_ref_LR
    else:
        blue = global_variables.blue_HR
        red = global_variables.red_HR
        green = global_variables.green_HR
        mesh_ref = global_variables.mesh_ref

    save(mesh, global_variables.mesh_ref_LR, input_p[:-4] + "InitialGuess.ply", global_variables.red_LR, global_variables.green_LR, global_variables.blue_LR )
    save(meshReg, mesh_ref, input_p[:-4] + "FinalReconstruction.ply",  red, green, blue)
    # Save optimal reconstruction
   
