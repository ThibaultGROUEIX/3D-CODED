import sys
sys.path.append('/home/thibault/lib/smpl')

import pymesh
import numpy as np
from smpl_webuser.serialization import load_model
mesh_ref = pymesh.load_mesh("./template/template_color.ply")
import cPickle as pickle
import os


def generate_surreal(pose, beta, outmesh_path):
    """
    This function generation 1 human using a random pose and shape estimation from surreal
    """
    ## Assign gaussian pose
    m.pose[:] = pose
    m.betas[:] = beta
    m.pose[0:3]=0
    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid


    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def generate_gaussian(pose, beta, outmesh_path):
    """
    This function generation 1 human using a random gaussian pose and shape
    """
    ## Assign gaussian pose
    m.betas[:] = beta
    m.pose[0:3]=0
    m.pose[3:]=0.3 * np.random.randn(69)
    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid


    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def generate_benthuman(pose, beta, outmesh_path):
    """
    This function generation 1 human using a random gaussian pose and shape, with random gaussian parameters for specific pose parameters
    """
    ## Assign random pose parameters except for certain ones to have random bent humans
    m.pose[:] = pose
    m.betas[:] = beta

    a = np.random.randn(12)
    m.pose[1] = 0
    m.pose[2] = 0
    m.pose[3] = -1.0 + 0.1*a[0]
    m.pose[4] = 0 + 0.1*a[1]
    m.pose[5] = 0 + 0.1*a[2]
    m.pose[6] = -1.0 + 0.1*a[0]
    m.pose[7] = 0 + 0.1*a[3]
    m.pose[8] = 0 + 0.1*a[4]
    m.pose[9] = 0.9 + 0.1*a[6]
    m.pose[0] = - (-0.8 + 0.1*a[0] )
    m.pose[18] = 0.2 + 0.1*a[7]
    m.pose[43] = 1.5 + 0.1*a[8]
    m.pose[40] = -1.5 + 0.1*a[9]
    m.pose[44] = -0.15 
    m.pose[41] = 0.15
    m.pose[48:54] = 0

    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid

    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def find_joint_influence(pose, beta, outmesh_path,i):
    m.pose[:] = 0
    m.betas[:] = beta
    m.pose[i] = 1
    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) 
    point_set[:,0:3] = point_set[:,0:3] - centroid

    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def generate_potential_templates(pose, beta, outmesh_path):
    # template 0
    m.pose[:] = 0
    m.betas[:] = beta
    m.pose[5] = 0.5
    m.pose[8] = -0.5
    m.pose[53] = -0.5
    m.pose[50] = 0.5

    point_set = m.r.astype(np.float32)
    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh('search/template0.ply', mesh, "red", "green", "blue", ascii=True)

    # template 1
    m.pose[:] = 0
    point_set = m.r.astype(np.float32)

    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))


    pymesh.meshio.save_mesh('search/template1.ply', mesh, "red", "green", "blue", ascii=True)
    return


def get_random(poses, betas):
    beta_id = np.random.randint(np.shape(betas)[0]-1)
    beta = betas[beta_id]
    pose_id = np.random.randint(len(poses)-1)
    pose_ = database[poses[pose_id]]
    pose_id = np.random.randint(np.shape(pose_)[0])
    pose = pose_[pose_id]
    return pose, beta


def generate_database_surreal(male):
    #TRAIN DATA
    nb_generated_humans = 100000
    nb_generated_humans_val = 100
    if male:
        betas = database['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database['femaleshapes']
        offset = nb_generated_humans
        offset_val = nb_generated_humans_val

    poses = [i for i in database.keys() if "pose" in i]
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))
    params = []
    for i in range(nb_generated_humans):
        pose, beta = get_random(poses, betas)
        generate_surreal(pose, beta, 'dataset-surreal/' + str(offset + i) + '.ply')
    

    #VAL DATA
    for i in range(nb_generated_humans_val):
        pose, beta = get_random(poses, betas)
        generate_surreal(pose, beta, 'dataset-surreal-val/' + str(offset_val + i) + '.ply')

    return 0


def generate_database_benthumans(male):
    #TRAIN DATA
    nb_generated_humans = 15000
    nb_generated_humans_val = 100
    if male:
        betas = database['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database['femaleshapes']
        offset = nb_generated_humans
        offset_val = nb_generated_humans_val

    poses = [i for i in database.keys() if "pose" in i]
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))
    params = []
    for i in range(nb_generated_humans):
        pose, beta = get_random(poses, betas)
        generate_benthuman(pose, beta, 'dataset-bent/' + str(offset + i) + '.ply')
    
    #VAL DATA
    for i in range(nb_generated_humans_val):
        pose, beta = get_random(poses, betas)
        generate_benthuman(pose, beta, 'dataset-bent-val/' + str(offset_val + i) + '.ply')

    return 0


if __name__ == '__main__':
    os.mkdir("dataset-surreal")
    os.mkdir("dataset-surreal-val")
    os.mkdir("dataset-bent")
    os.mkdir("dataset-bent-val")
    ### GENERATE MALE EXAMPLES
    m = load_model("./smpl_data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    database = np.load("/home/thibault/tmp/SURREAL/smpl_data/smpl_data.npz")
    generate_database_surreal(male=True)
    generate_database_benthumans(male=True)
   
    ### GENERATE FEMALE EXAMPLES
    m = load_model('./smpl_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    database = np.load("./smpl_data/smpl_data.npz")
    generate_database_surreal(male=False)
    generate_database_benthumans(male=False)
   
