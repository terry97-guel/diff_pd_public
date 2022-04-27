import sys
sys.path.append('./python')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
from tqdm import tqdm

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from experiment.env.SorosimRountingEnv import CRoutingTendonEnv3d 
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
from py_diff_pd.common.project_path import root_path

seed = 42
np.random.seed(seed)
folder = Path('custom_routing_tendon')
youngs_modulus = 5e5
poissons_ratio = 0.45
refinement = 11
muscle_cnt = 8
act_max = 2
env = CRoutingTendonEnv3d(seed, folder, {
    'muscle_cnt': muscle_cnt,
    'refinement': refinement,
    'youngs_modulus': youngs_modulus,
    'poissons_ratio': poissons_ratio })
deformable = env.deformable()
mesh = env.mesh
# Camera options
env._camera_pos = (0.3, -1, .25)
env._scale = 2

# Optimization parameters.
import multiprocessing
cpu_cnt = multiprocessing.cpu_count()
thread_ct = 96
print_info('Detected {:d} CPUs. Using {} of them in this run'.format(cpu_cnt, thread_ct)) 
pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
    'use_bfgs': 1, 'bfgs_history_size': 10 }
methods = ('pd_eigen',)
opts = (pd_opt,)

dt = 1e-2
frame_num = 10

# Initial state.
dofs = deformable.dofs()
act_dofs = deformable.act_dofs()
q0 = env.default_init_position()
v0 = np.zeros(dofs)
f0 = [np.zeros(dofs) for _ in range(frame_num)]
act_maps = env.act_maps()
u_dofs = len(act_maps)
cell_nums = env._cell_nums
node_nums = env._node_nums

assert cell_nums[0]*cell_nums[1]*cell_nums[2] == mesh.NumOfElements()
assert node_nums[0]*node_nums[1]*node_nums[2] == mesh.NumOfVertices()

def variable_to_act(x):
    act = np.ones(act_dofs)
    for i, a in enumerate(act_maps):
        act[a] = 1 - x[i]
    return act

# Get Marker
segNumber = 10

zcorrdinate = np.linspace(0,0.12,segNumber) / env._dx
xcoordinate = np.ones_like(zcorrdinate) * 0.01 /env._dx
ycoordinate = np.ones_like(zcorrdinate) * 0.02 /env._dx

py_verticeIdxs = []
for m in range(segNumber):
    if m==0:
        continue
    idx = node_nums[2]*node_nums[1]*xcoordinate[m] + node_nums[2]*ycoordinate[m] + zcorrdinate[m]
    idx = int(idx)
    py_verticeIdxs.append(idx)
py_verticeIdxs = np.array(py_verticeIdxs)

env.set_marker(py_verticeIdxs,verbose=True)


import json

# Opening JSON file
trainInter,valInter = \
    open('python/experiment/preprocess/trainInterpolate.json'),open('python/experiment/preprocess/valInterpolate.json')

trainInterpolate, valInterpolate = json.load(trainInter), json.load(valInter)
trainInterpolate = trainInterpolate['data'], valInterpolate['data']
trainInter.close(), valInter.close()


# Set method
method,opt = methods[0], opts[0]

soropos = [
    1.243147e-02,0,9.813514e-03,
    2.485175e-02,0,9.254224e-03,
    3.724967e-02,0,8.322633e-03,
    4.961406e-02,0,7.019579e-03,
    6.193380e-02,0,5.346236e-03,
    7.419781e-02,0,3.304108e-03,
    8.639505e-02,0,8.950347e-04,
    9.851453e-02,0,-1.878817e-03, 
    1.105454e-01,0,-5.014951e-03]
soropos = np.array(soropos).reshape(-1,3)

position = np.zeros_like(soropos)
position[:,0] = soropos[:,1] + 0.01
position[:,1] = soropos[:,2] + 0.01
position[:,2] = soropos[:,0]

position.tolist()

data = {'actuation':[10,0,0,0], 'position': position}
act = data['actuation']
act = [ele for ele in act]
act = variable_to_act(act)
markers = data['position']
env._construct_loss_and_grad(markers)

acts = [[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]]
for idx,singleact in enumerate(acts):
    if True:
        act = [ele for ele in singleact]
        print("Acutation:", act)
        act = variable_to_act(act)
        vis_folder = 'test' + str(idx)
        loss, info = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [act for _ in range(frame_num)], f0, require_grad=False, vis_folder=vis_folder)


        vis_folder = 'test' +str(idx)
        file_name = vis_folder +'.png'

        i = 10
        mesh_file = str(env._folder / vis_folder / '{:04d}.bin'.format(i))

        if idx%2==0:
            ## Back view
            options = {
                'file_name': file_name,
                'light_map': 'uffizi-large.exr',
                'max_depth': 2,
                'camera_pos': (0.2, 1, .55),
                'camera_lookat': (0, -.55, -.15),
            }
        else:
            ## Side view
            options = {
                'file_name': file_name,
                'light_map': 'uffizi-large.exr',
                'max_depth': 2,
                'camera_pos': (1, 0.3, .25),
                'camera_lookat': (-0.1, 0, 0.15),
            }
            
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=(.3, .7, .5), transforms=[
            ('s', env._scale),
        ])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        for element in env.act_maps()[idx]:
            vertices = mesh.py_element(element)
            out = np.zeros(8*3).reshape(8,3)
            assert len(vertices) == 8
            for id,vertice in enumerate(vertices):
                out[id] = mesh.py_vertex(vertice)

            marker = out.mean(0)
            
            renderer.add_shape_mesh({ 'name': 'sphere', 'center': marker, 'radius': env._dx*env._scale },
                transforms=[('s', env._scale)], color=(0.1, 0.1, 0.9))


        renderer.render()

