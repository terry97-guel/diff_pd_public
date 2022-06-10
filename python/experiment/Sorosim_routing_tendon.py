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
from functools import partial
import wandb

if __name__ == '__main__':
    wandb.init(project = 'DiffPD with Sorosim Data')
    wandb.run.name = 'Baseline'
    
    
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
    node_nums = env._node_nums
    # assert u_dofs * (refinement ** 3) * muscle_ext == act_dofs

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
    
    # env = None
    # print('delete temporary environments')
    
    # Get dataset
    import json
    trainInter,valInter = \
        open('python/experiment/preprocess/trainInterpolate.json'),open('python/experiment/preprocess/valInterpolate.json')

    trainInterpolate, valInterpolate = json.load(trainInter), json.load(valInter)

    trainInterpolate, valInterpolate = trainInterpolate['data'], valInterpolate['data']
    trainInter.close(), valInter.close()
    
    # Set method
    method,opt = methods[0], opts[0]
    
    def simulate_batch(x, datajson, batchsize=16, vis_folderprefix=None, requires_grad=False):
        
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
        
        for batchidx in range(batchsize):
            print("Optimizing... batchidx:{}/{}".format(batchidx+1,batchsize))
            data = np.random.choice(datajson)
            act = data['actuation']
            # print('Actuation:', act)
            act = [ele/1 for ele in act]
            act = variable_to_act(act)
            markers = data['position']
            # print('position:', markers)
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
            wandb.log({'avgLoss':avgLoss, 'avgGrads':avgGrads, 'E':E, 'nu':nu})

        return avgLoss, avgGrads        
        
        
    ## Set batchsize
    batchsize = 32
    
    # ## Initial guess for material parameter.
    x_lb = ndarray([np.log(1e4), np.log(0.2)])
    x_ub = ndarray([np.log(1e7), np.log(0.45)])
    x_init = np.random.uniform(x_lb, x_ub)
    print_info('Simulating initial solution. Please check out the {}/init folder'.format(folder))
    
    loss, info = simulate_batch(x=x_init,datajson = trainInterpolate,batchsize=batchsize, vis_folderprefix='init', requires_grad=False)
    wandb.log({'Loss_init':loss, 'E_init':np.exp(x_init[0]), 'nu_init': np.exp(x_init[1])})
    
    print(loss)
    print_info('Initial guess is ready.')
    
    ## Optimization for material parameter
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    
    simulate_trainbatch = partial(simulate_batch,datajson = trainInterpolate, batchsize = batchsize, vis_folderprefix = None, requires_grad=True)
    
    t0 = time.time()
    result = scipy.optimize.minimize(simulate_trainbatch, np.copy(x_init),
        method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-2, 'maxiter': 100 })
    t1 = time.time()
    print(result.success)
    x_final = result.x
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))

    # # Visualize results.
    loss, info = simulate_batch(x=x_final, datajson = valInterpolate,batchsize=batchsize, vis_folderprefix='val', requires_grad=False)
    wandb.log({'Loss_final':loss, 'E_final':np.exp(x_final[0]), 'nu_final': np.exp(x_final[1])})

##################################################################################################################################
    # # Normalize the loss.
    # rand_state = np.random.get_state()
    # random_guess_num = 16
    # random_loss = []
    # for _ in range(random_guess_num):
    #     x_rand = np.random.uniform(low=x_lb, high=x_ub)
    #     E = np.exp(x_rand[0])
    #     nu = np.exp(x_rand[1])
    #     env_opt = CRoutingTendonEnv3d(seed, folder, {
    #         'muscle_cnt': muscle_cnt,
    #         'muscle_ext': muscle_ext,
    #         'refinement': refinement,
    #         'youngs_modulus': E,
    #         'poissons_ratio': nu })
    #     env_opt.set_marker(py_verticeIdxs)
    #     loss, _ = env_opt.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=False, vis_folder=None)
    #     print('E: {:3e}, nu: {:3f}, loss: {:3f}'.format(E, nu, loss))
    #     random_loss.append(loss)
    # loss_range = ndarray([0, np.mean(random_loss)])
    # print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    # np.random.set_state(rand_state)

    # # loss_range = np.array([0, 128.36210175])
    # print(loss_range)
    # data = { 'loss_range': loss_range }
    # method_display_names = { 'pd_eigen': 'pd_eigen'}
    
    # for method, opt in zip(methods, opts):
    #     data[method] = []
    #     print_info('Optimizing with {}...'.format(method_display_names[method]))
    #     def loss_and_grad(x):
    #         E = np.exp(x[0])
    #         nu = np.exp(x[1])
    #         env_opt = CRoutingTendonEnv3d(seed, folder, {
    #             'muscle_cnt': muscle_cnt,
    #             'muscle_ext': muscle_ext,
    #             'refinement': refinement,
    #             'youngs_modulus': E,
    #             'poissons_ratio': nu })
    #         env_opt.set_marker(py_verticeIdxs)
    #         loss, _, info = env_opt.simulate(dt, frame_num, method, opt, q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=True, vis_folder=None)
    #         grad = info['material_parameter_gradients']
    #         grad = grad * np.exp(x)
    #         print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
    #             loss, np.linalg.norm(grad), E, nu, info['forward_time'], info['backward_time']))
    #         single_data = {}
    #         single_data['loss'] = loss
    #         single_data['grad'] = np.copy(grad)
    #         single_data['E'] = E
    #         single_data['nu'] = nu
    #         single_data['forward_time'] = info['forward_time']
    #         single_data['backward_time'] = info['backward_time']
    #         data[method].append(single_data)
    #         return loss, grad


    #     t0 = time.time()
    #     result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
    #         method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-2, 'maxiter': 10 })
    #     t1 = time.time()
    #     print(result.success)
    #     x_final = result.x
    #     print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
    #     pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

    #     # Visualize results.
    #     E = np.exp(x_final[0])
    #     nu = np.exp(x_final[1])
    #     env_opt = CRoutingTendonEnv3d(seed, folder, {
    #             'muscle_cnt': muscle_cnt,
    #             'muscle_ext': muscle_ext,
    #             'refinement': refinement,
    #             'youngs_modulus': youngs_modulus,
    #             'poissons_ratio': poissons_ratio })
    #     env_opt.set_marker(py_verticeIdxs)
    #     env_opt.simulate(dt, frame_num, method, opt, q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=False, vis_folder=method)
