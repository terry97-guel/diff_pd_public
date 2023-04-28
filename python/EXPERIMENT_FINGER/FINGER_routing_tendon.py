import sys

from pathlib import Path

try: 
    get_ipython().__class__.__name__
    BASEDIR = Path().absolute()
except: 
    BASEDIR = Path(__file__).parent


gettrace = getattr(sys, 'gettrace', None)
if gettrace() is None:
    print('Run Mode')
    DEBUG_MODE = False
    
elif gettrace():
    print('Debugging Mode... Skip Logging...')
    DEBUG_MODE = True
    
else:
    print("Let's do something interesting")
    sys.exit(0)

runname = "Full_Batch_Scale_50"

folder = BASEDIR / 'FINGER_ROUTING_TENDON' / runname
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
from EXPERIMENT_FINGER.dataloader import get_dataset, Sampler
from functools import partial
import wandb
import torch
import random


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

# lr = 2

PROPORTION_MATRIX = np.array(
    [
        [1,0.1,0.4,0.4, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        [0.1,1,0.4,0.4, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        [0.4,0.4,1,0.1, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        [0.4,0.4,0.1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0],
        
        [0,0,0,0,  1,0.1,0.4,0.4, 0,0,0,0, 0,0,0,0],
        [0,0,0,0,  0.1,1,0.4,0.4,  0,0,0,0, 0,0,0,0],
        [0,0,0,0,  0.4,0.4,1,0.1, 0,0,0,0, 0,0,0,0],
        [0,0,0,0,  0.4,0.4,0.1,1, 0,0,0,0, 0,0,0,0],
        
        [0,0,0,0, 0,0,0,0, 1,0.1,0.4,0.4, 0,0,0,0],
        [0,0,0,0, 0,0,0,0, 0.1,1,0.4,0.4, 0,0,0,0],
        [0,0,0,0, 0,0,0,0, 0.4,0.4,1,0.1, 0,0,0,0],
        [0,0,0,0, 0,0,0,0, 0.4,0.4,0.1,1, 0,0,0,0],
        
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0.1,0.4,0.4],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0.1,1,0.4,0.4],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0.4,0.4,1,0.1],
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 0.4,0.4,0.1,1],
        ])
PROPORTION_MATRIX = PROPORTION_MATRIX[:4,:4]


if __name__ == '__main__':
    if not DEBUG_MODE:
        wandb.init(project = 'DiffPD with FINGER_DATASET')
        wandb.run.name = str.replace(runname,"_"," ")
    
    
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    refinement = 4
    muscle_cnt = 8
    env = CRoutingTendonEnv3d(SEED, folder, {
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
    frame_num = 15

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
    trainDataset,valDataset,testDataset,extTestDataset = get_dataset(f"python/EXPERIMENT_FINGER/FINGER.json")
    
    # Set method
    method,opt = methods[0], opts[0]
    
    # p_offset
    idx = torch.argmin(torch.norm(trainDataset.motor_control - torch.tensor([0,0]), dim=1))
    init_pose = trainDataset.position[idx].flatten()
    p_offset = np.array([init_pose[0], init_pose[1], 0])
    
    def simulate_batch(x, dataset, batchsize=16, vis_folderprefix=None, requires_grad=False, batchidxs = None, prefix = None):
        if prefix is None:
            prefix = vis_folderprefix
            
        
        E = np.exp(x[0])
        nu = np.exp(x[1])
        
        if vis_folderprefix==None: savefolder=folder
        else: savefolder = folder / vis_folderprefix
        env = CRoutingTendonEnv3d(SEED, savefolder, {
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
        
        FAILED_NUMBER = 0
        for batchidx,idx in enumerate(batchidxs):
            print("Optimizing... batchidx:{}/{}".format(batchidx+1,batchsize))
            
            datapoint = dataset.__getitem__(idx)
            act_raw = datapoint['motor_control']
            
            temp = np.zeros(4)
            if act_raw[0]>0:
                temp[0] = act_raw[0]
            else:
                temp[1] = -act_raw[0]
                
            if act_raw[1]>0:
                temp[2] = act_raw[1]
            else:
                temp[3] = -act_raw[1]
            
            # act_raw = temp/16_000
            act_raw = temp/50
            
            act_raw = PROPORTION_MATRIX @ act_raw
            
            act_raw = act_raw[[2,0,3,1]]
            
            markers = datapoint['position'].flatten() - p_offset + torch.tensor([10,10,.0])/1000
            print('actuation: ', act_raw)
            print('position:', markers)

            act = variable_to_act(act_raw)
            env._construct_loss_and_grad(markers)
            
            # set output folder path
            if not vis_folderprefix==None: vis_folder = vis_folderprefix + str(batchidx+1)
            else: vis_folder = None
            
            import json
            with open(f"{folder}/temp/{vis_folder}_act", 'w') as outfile:
                json.dump(dict(act=act_raw.tolist()), outfile)
            
            act_frames = []
            for _ in range(frame_num):
                act_frames.append(act)
            
            if requires_grad:
                try:
                    loss, _, info = env.simulate(dt, frame_num, method, opt, q0, v0, act_frames, f0, require_grad=requires_grad, vis_folder=vis_folder)
                except RuntimeError:
                    print("[WARNING!!] SIMULATION HAS FAILED TO CONVERGE!!")
                    print("Skipping Current MOTOR COMMAND!!")
                    FAILED_NUMBER = FAILED_NUMBER + 1
                    continue
                grad = info['material_parameter_gradients']
                print('loss: {:8.3f}, |grad|: {:8.3f}, gradE{:12.3e}, gradNu{:12.3e}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                    loss, np.linalg.norm(grad), grad[0], grad[1], E, nu, info['forward_time'], info['backward_time']))
            else:
                try:
                    loss, info = env.simulate(dt, frame_num, method, opt, q0, v0, act_frames, f0, require_grad=requires_grad, vis_folder=vis_folder)
                except RuntimeError:
                    print("[WARNING!!] SIMULATION HAS FAILED TO CONVERGE!!")
                    print("Skipping Current MOTOR COMMAND!!")
                    FAILED_NUMBER = FAILED_NUMBER + 1
                    continue
                grad = 0
                print('loss: {:8.3f}, forward time: {:6.3f}s,'.format(loss, info['forward_time']))

            grad = grad * np.exp(x)
            if grad[0]<0 and DEBUG_MODE:
                print("Found!!!")
                pass
            losses.append(loss)
            grads.append(grad)
        
        losses,grads = np.array(losses), np.array(grads)
        avgLoss, avgGrads = losses.mean(0), grads.mean(0)
        
        # avgGrads = avgGrads * lr
        FAILED_RATIO = FAILED_NUMBER / len(batchidxs)
        if not DEBUG_MODE:
            if requires_grad:
                wandb.log(
                    {prefix+'_L2Loss':avgLoss, prefix+'_L1_Loss':(2*avgLoss)**(1/2), \
                        prefix+'_logEGrads':float(avgGrads[0]),prefix+'_NuGrads':float(avgGrads[1]), \
                            prefix+'_E':E, prefix+'_nu':nu, prefix+'_FAILED_RATIO':FAILED_RATIO})
            else:
                wandb.log({prefix+'_L1_Loss':(2*avgLoss)**(1/2), prefix+'_FAILED_RATIO':FAILED_RATIO})
        
        return avgLoss, avgGrads
    
    # ## Initial guess for material parameter.
    x_lb = ndarray([np.log(5e5), np.log(0.2)])
    x_ub = ndarray([np.log(1e7), np.log(0.45)])
    x_init = np.random.uniform(x_lb, x_ub)
    print_info('Simulating initial solution. Please check out the {}/init folder'.format(folder))
    
    ##### Simulate INIT  #####
    batch_idxs_saved = []
    
    idx = torch.argmin(torch.norm(trainDataset.motor_control - torch.tensor([1000,0]), dim=1))
    batch_idxs_saved.append(idx)

    idx = torch.argmin(torch.norm(trainDataset.motor_control - torch.tensor([-1000,0]), dim=1))
    batch_idxs_saved.append(idx)

    idx = torch.argmin(torch.norm(trainDataset.motor_control - torch.tensor([0,1000]), dim=1))
    batch_idxs_saved.append(idx)

    idx = torch.argmin(torch.norm(trainDataset.motor_control - torch.tensor([0,-1000]), dim=1))
    batch_idxs_saved.append(idx)
    batch_idxs_saved = np.array(batch_idxs_saved)
    
    # loss, info = simulate_batch(x=x_init,dataset = trainDataset, vis_folderprefix='init', requires_grad=False, batchidxs = batchidxs)
    
    # if not DEBUG_MODE:
    loss, info = simulate_batch(x=x_init,dataset = trainDataset, vis_folderprefix='init', requires_grad=True, batchidxs = batch_idxs_saved, prefix = "Init")
    print("Inital L1 Loss: ", (2*loss)**(1/2))
    print_info('Initial guess is ready.')
    # sys.exit(0)
    
    
    ##### Train #####
    ## Optimization for material parameter
    Train = True
     
    t0 = time.time()
    if Train:
        vis_folderprefix = 'train' if DEBUG_MODE else None
        bounds = scipy.optimize.Bounds(x_lb, x_ub)
        batchsize = 64
        simulate_trainbatch = partial(simulate_batch,dataset = trainDataset, batchsize = batchsize, vis_folderprefix = vis_folderprefix, requires_grad=True, prefix = "Train")
        # simulate_trainbatch = partial(simulate_batch,dataset = trainDataset, batchsize = len(trainDataset), vis_folderprefix = vis_folderprefix, requires_grad=True, prefix="Train")
        result = scipy.optimize.minimize(simulate_trainbatch, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-2, 'maxiter': 100 })
        
        print(result.success)
        x_final = result.x
    else:
        x_final = x_init
    t1 = time.time()
    
    E  = np.exp(x_final[0])
    nu = np.exp(x_final[1])
    if not DEBUG_MODE: wandb.log({'E':E, 'nu':nu})
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))

    ##### Test #####
    loss, info = simulate_batch(x=x_init, dataset = trainDataset, vis_folderprefix='final', requires_grad=False, batchidxs = batch_idxs_saved, prefix = "Final")
    
    loss, info = simulate_batch(x=x_final, dataset = valDataset,batchsize=len(valDataset), requires_grad=False, prefix = "Val")
    
    loss, info = simulate_batch(x=x_final, dataset = testDataset,batchsize=len(testDataset), requires_grad=False, prefix = "Test")
    
    loss, info = simulate_batch(x=x_final, dataset = extTestDataset,batchsize=len(extTestDataset), requires_grad=False, prefix = "extTest")
    
    

