import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import export_gif
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector

class CRoutingTendonEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        muscle_cnt = options['muscle_cnt']
        muscle_ext = options['muscle_ext']
        refinement = options['refinement']
        youngs_modulus = options['youngs_modulus']
        poissons_ratio = options['poissons_ratio']
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([5.,])

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        # Mesh size: 2 x 2 x (muscle_cnt x muscle_ext).
        cell_nums = (2 * refinement, 2 * refinement, muscle_ext * muscle_cnt * refinement)
        origin = ndarray([0, 0, 0])
        node_nums = tuple(n + 1 for n in cell_nums)
        dx = 0.05 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Boundary conditions.
        for i in range(node_nums[0]):
            for j in range(node_nums[1]):
                node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
                vx, vy, vz = mesh.py_vertex(node_idx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # Actuation.
        element_num = mesh.NumOfElements()
        act_indices = list(range(element_num))
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], act_indices)
        act_maps = []
        for i in range(2):
            for j in range(2):
                for k in range(muscle_cnt):
                    act = []
                    for ii in range(refinement):
                        for jj in range(refinement):
                            for kk in range(muscle_ext * refinement):
                                idx_i = i * refinement + ii
                                idx_j = j * refinement + jj
                                idx_k = k * muscle_ext * refinement + kk
                                idx = idx_i * (2 * refinement * muscle_ext * muscle_cnt * refinement) + \
                                    idx_j * muscle_cnt * muscle_ext * refinement + idx_k
                                act.append(idx)
                    act_maps.append(act)
        self.__act_maps = act_maps

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self.mesh = mesh
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._actuator_parameters = actuator_parameters
        self._f_ext = f_ext
        self._stepwise_loss = True
        self._dx = dx

        self.__spp = options['spp'] if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        k = dof % self.__node_nums[2]
        return k == 0

    def act_maps(self):
        return self.__act_maps

    def set_marker(self, py_verticeIdxs):
        # Idxs of vertice -> (x,y,z) combined
        self.py_verticeIdxs = py_verticeIdxs
        Markernumber = py_verticeIdxs.shape[0]
        # Idxs of vertice -> (x),(y),(z)
        self.py_verticesIdxs = np.zeros(Markernumber*3,dtype=np.int)
        self.markerVertices = []
        
        for idx,elementidx in enumerate(self.py_verticeIdxs):
            
            self.py_verticesIdxs[idx*3] = elementidx*3
            self.py_verticesIdxs[idx*3+1] = elementidx*3 +1
            self.py_verticesIdxs[idx*3+2] = elementidx*3 + 2
            
            # markerpos = mesh.py_vertex(int(elementidx))
            # self.markerVertices.append(list(markerpos))
        
        NumOfq0 = self._q0.shape[0]
        self.onehot_py_verticesIdxs = np.eye(NumOfq0)[self.py_verticesIdxs]
        self.onehot_py_verticesIdxs = np.sum(self.onehot_py_verticesIdxs,axis=0,dtype=np.bool)
        
    def _display_mesh(self, mesh_file, file_name):
        # Render.
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.4, -1., .25),
            'camera_lookat': (0, .15, .15),
        }
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=(.3, .7, .5), transforms=[
            ('s', 0.4),
        ])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        # display marker
        q = ndarray(mesh.py_vertices())
        markers = q[self.py_verticesIdxs].reshape(-1,3)
        for marker in markers:
            renderer.add_shape_mesh({ 'name': 'sphere', 'center': marker, 'radius': 0.025 },
                transforms=[('s', 0.4)], color=(0.9, 0.1, 0.1))
        renderer.render()

    def _stepwise_loss_and_grad(self, q, v, i):    
        mesh_file = self._folder / 'groundtruth' / '{:04d}.bin'.format(i)
        if not mesh_file.exists(): return 0, np.zeros(q.size), np.zeros(q.size)

        mesh = HexMesh3d()
        mesh.Initialize(str(mesh_file))
        q_ref = ndarray(mesh.py_vertices())
        
        q_ref = np.where(self.onehot_py_verticesIdxs,q_ref,0)
        q = np.where(self.onehot_py_verticesIdxs,q,0)
        grad = q - q_ref
        loss = 0.5 * grad.dot(grad)
        return loss, grad, np.zeros(q.size)