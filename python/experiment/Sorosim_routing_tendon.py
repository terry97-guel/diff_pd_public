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

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('custom_routing_tendon')
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    refinement = 8
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
    # assert u_dofs * (refinement ** 3) * muscle_ext == act_dofs

    def variable_to_act(x):
        act = np.ones(act_dofs)
        for i, a in enumerate(act_maps):
            act[a] = 1 - x[i]
        return act
    def variable_to_act_gradient(x, grad_act):
        grad_u = np.zeros(u_dofs)
        for i, a in enumerate(act_maps):
            grad_u[i] = np.sum(grad_act[a])
        return grad_u

    a0 = [10,0,0,0]
    a0 = variable_to_act(a0)
    
    # Set marker in env
    Markernumber = 9
    
    # for i in range(mesh.NumOfVertices()):
    #     x,y,z = mesh.py_vertex(int(i))
    #     assert(x == mesh.py_vertices()[i*3])
    #     assert(y == mesh.py_vertices()[i*3+1])
    #     assert(z == mesh.py_vertices()[i*3+2])
    
    # py_verticeIdxs = np.random.randint(mesh.NumOfVertices()-1, size = Markernumber)
    # env.set_marker(py_verticeIdxs)
    
    # Generate groundtruth motion
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=False, vis_folder='pull2')
    print_info("Groundtruth motion generated")
    
    # # # Visualize initial guess.
    # x_lb = ndarray([np.log(1e4), np.log(0.2)])
    # x_ub = ndarray([np.log(5e6), np.log(0.45)])
    # x_init = np.random.uniform(x_lb, x_ub)
    
    # print_info('Simulating and rendering initial solution. Please check out the {}/init folder'.format(folder))
    # env = CRoutingTendonEnv3d(seed, folder, {
    #     'muscle_cnt': muscle_cnt,
    #     'muscle_ext': muscle_ext,
    #     'refinement': refinement,
    #     'youngs_modulus': np.exp(x_init[0]),
    #     'poissons_ratio': np.exp(x_init[1]) })
    # env.set_marker(py_verticeIdxs)
    # loss, info = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [a0 for _ in range(frame_num)], f0, require_grad=False, vis_folder='init')
    # print(loss)
    # print_info('Initial guess is ready. You can play it by opening {}/init.gif'.format(folder))
    
    # bounds = scipy.optimize.Bounds(x_lb, x_ub)

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
