import sys

from pathlib import Path

try: 
    get_ipython().__class__.__name__
    BASEDIR = Path().absolute()
except: BASEDIR = Path(__file__).parent

sys.path.append(str(BASEDIR/".."))

import time
import numpy as np
import scipy.optimize
import pickle
from tqdm import tqdm

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from EXPERIMENT_FINGER.env.FINGER_RountingEnv import CRoutingTendonEnv3d 
from EXPERIMENT_FINGER.dataloader import get_FINGER_Dataset, find_most_extrem
from functools import partial
import wandb
import torch
import random


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

if __name__ == '__main__':
    wandb.init(project = 'DiffPD with FINGER_DATASET')
    wandb.run.name = 'Baseline_Test_L1'
    
    
    seed = 42
    np.random.seed(seed)
    folder = BASEDIR / 'FINGER_ROUTING_TENDON'
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    refinement = 4
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
    node_nums = env._node_nums
    # assert u_dofs * (refinement ** 3) * muscle_ext == act_dofs

    def variable_to_act(x):
        act = np.ones(act_dofs)
        for i, a in enumerate(act_maps):
            act[a] = 1 - x[i]
        return act
    
    # Get Marker
    segNumber = 2
    
    zcorrdinate = (np.linspace(0,0.12,segNumber)    /env._dx).astype(int)
    xcoordinate = (np.ones_like(zcorrdinate) * 0.01 /env._dx).astype(int)
    ycoordinate = (np.ones_like(zcorrdinate) * 0.01 /env._dx).astype(int)
    
    py_verticeIdxs = []
    for m in range(segNumber):
        if m==0:
            continue
        idx = node_nums[2]*node_nums[1]*xcoordinate[m] + node_nums[2]*ycoordinate[m] + zcorrdinate[m]
        idx = int(idx)
        py_verticeIdxs.append(idx)
    py_verticeIdxs = np.array(py_verticeIdxs)
    
    env.set_marker(py_verticeIdxs,verbose=True)
    
    
    # Get dataset
    trainDataset,valDataset,testDataset,extTestDataset = get_FINGER_Dataset(BASEDIR)
    
# Set method
    method,opt = methods[0], opts[0]
    
    def simulate_batch(x, dataset, batchsize=16, vis_folderprefix=None, requires_grad=False, batchidxs = None):
        
        E = np.exp(x[0])
        nu = np.exp(x[1])
        
        if vis_folderprefix==None: savefolder=folder
        else: savefolder = folder / vis_folderprefix
        env = CRoutingTendonEnv3d(seed, savefolder, {
        'muscle_cnt': muscle_cnt,
        'refinement': refinement,
        'youngs_modulus': E,
        'poissons_ratio': nu})
        env.set_marker(py_verticeIdxs, verbose=False)
        
        # Camera options
        env._camera_pos = (0.3, -1, .25)
        env._scale = 2
        
        losses = []
        grads = []
        
        if batchidxs is None: 
            batchidxs = np.random.choice(np.arange(len(dataset)), size=batchsize, replace=False)
        else:
            batchsize = len(batchidxs)
            print("Batchidxs Explicitly Defined")
        
        for batchidx,idx in enumerate(batchidxs):
            print("Optimizing... batchidx:{}/{}".format(batchidx+1,batchsize))
            
            act, markers = dataset.__getitem__(idx)
            print('actuation: ', act)
            print('position:', markers)
            
            act = variable_to_act(act)
            env._construct_loss_and_grad(markers)
            
            # set output folder path
            if not vis_folderprefix==None: vis_folder = vis_folderprefix + str(batchidx+1)
            else: vis_folder = None
            
            if requires_grad:
                loss, _, info = env.simulate(dt, frame_num, method, opt, q0, v0, [act for _ in range(frame_num)], f0, require_grad=requires_grad, vis_folder=vis_folder)
                grad = info['material_parameter_gradients']
                print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                    loss, np.linalg.norm(grad), E, nu, info['forward_time'], info['backward_time']))
            else:
                loss, info = env.simulate(dt, frame_num, method, opt, q0, v0, [act for _ in range(frame_num)], f0, require_grad=requires_grad, vis_folder=vis_folder)
                grad = 0
                print('loss: {:8.3f}, forward time: {:6.3f}s,'.format(loss, info['forward_time']))

            
            grad = grad * np.exp(x)

            
            losses.append(loss)
            grads.append(grad)
        
        losses,grads = np.array(losses), np.array(grads)
        
        avgLoss, avgGrads = losses.mean(0), grads.mean(0)

        
        if requires_grad:
            wandb.log({'train_Loss':avgLoss, 'logEGrads':float(avgGrads[0]),'NuGrads':float(avgGrads[1]), 'E':E, 'nu':nu})

        return avgLoss, avgGrads
        
        
    ## Set batchsize
    batchsize = 64
    
    # ## Initial guess for material parameter.
    x_lb = ndarray([np.log(5e1), np.log(0.2)])
    x_ub = ndarray([np.log(1e7), np.log(0.45)])
    x_init = np.random.uniform(x_lb, x_ub)
    print_info('Simulating initial solution. Please check out the {}/init folder'.format(folder))
    
    ##### Simulate INIT  #####
    batchidxs = find_most_extrem(trainDataset)
    loss, info = simulate_batch(x=x_init,dataset = trainDataset, vis_folderprefix='init', requires_grad=False, batchidxs = batchidxs)
    wandb.log({'Loss_Extrem_test':(2*loss)**(1/2)})    
    print("Inital L1 Loss: ", (2*loss)**(1/2))
    print_info('Initial guess is ready.')
    
    ##### Train #####
    ## Optimization for material parameter
    Train = True
    
    t0 = time.time()
    if Train:
        bounds = scipy.optimize.Bounds(x_lb, x_ub)
        simulate_trainbatch = partial(simulate_batch,dataset = trainDataset, batchsize = batchsize, vis_folderprefix = None, requires_grad=True)
        result = scipy.optimize.minimize(simulate_trainbatch, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-2, 'maxiter': 100 })
        
        print(result.success)
        x_final = result.x
    else:
        x_final = x_init
    t1 = time.time()
    
    E  = np.exp(x_final[0])
    nu = np.exp(x_final[1])
    wandb.log({'E':E, 'nu':nu})
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))

    ##### Test #####
    batchidxs = find_most_extrem(trainDataset)
    loss, info = simulate_batch(x=x_init,dataset = trainDataset, vis_folderprefix='final', requires_grad=False, batchidxs = batchidxs)
    wandb.log({'Loss_Extrem_test':(2*loss)**(1/2)})
    
    loss, info = simulate_batch(x=x_final, dataset = valDataset,batchsize=len(valDataset), vis_folderprefix='val', requires_grad=False)
    wandb.log({'val_Loss_L1':(2*loss)**(1/2)})
    
    loss, info = simulate_batch(x=x_final, dataset = testDataset,batchsize=len(testDataset), requires_grad=False)
    wandb.log({'test_Loss_L1':(2*loss)**(1/2)})
    
    loss, info = simulate_batch(x=x_final, dataset = extTestDataset,batchsize=len(extTestDataset), requires_grad=False)
    wandb.log({'Ext_test_Loss_L1':(2*loss)**(1/2)})
    

