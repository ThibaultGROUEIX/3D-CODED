'''
    This is a short demo to see how to load and use the SMAL model.
    Please read the README.txt file for requirements.

'''

from smpl_webuser.serialization import load_model
from my_mesh.mesh import myMesh
import pickle as pkl
import numpy as np
# Load the smal model 
model_path = 'smal_CVPR2017.pkl'
model = load_model(model_path)

# Save the mean model
m = myMesh(v=model.r, f=model.f)
m.save_ply('smal_mean_shape.ply')
print 'saved mean shape'

# Load the family clusters data (see paper for details)
# and save the mean per-family shape
# 0-felidae(cats); 1-canidae(dogs); 2-equidae(horses);
# 3-bovidae(cows); 4-hippopotamidae(hippos);
# The clusters are over the shape coefficients (betas);
# setting different betas changes the shape of the model
model_data_path = 'smal_CVPR2017_data.pkl'
data = pkl.load(open(model_data_path))
print(data['cluster_cov'])


for i, betas in enumerate(data['cluster_means']):
    if not(i==4):
        continue
    model.betas[:] = betas
    model.pose[:] = 0.
    model.trans[:] = 0.
    m = myMesh(v=model.r, f=model.f)
    m.save_ply('family_'+str(i)+'.ply')
    print(np.shape(model.pose))
    # VAL DATA
    for j in range(200):
        model.pose[0:3]=0
        model.pose[3:]=0.2 * np.random.randn(96)
        m = myMesh(v=model.r, f=model.f)
        m.save_ply('dataset_hyppo_val/hyppo'+str(0+j)+'.ply') 
    
    # TRAINING DATA
    for j in range(200000):
            model.pose[0:3]=0
            model.pose[3:]=0.2 * np.random.randn(96)
            m = myMesh(v=model.r, f=model.f)
            m.save_ply('dataset_hyppo/hyppo'+str(0+j)+'.ply') 
