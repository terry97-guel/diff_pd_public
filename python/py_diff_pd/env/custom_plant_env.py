import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer

class CPlantEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 5e3

        bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'plant.bin'
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))
        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Obtain dx.
        fi = ndarray(mesh.py_element(0))
        dx = np.linalg.norm(ndarray(mesh.py_vertex(int(fi[0]))) - ndarray(mesh.py_vertex(int(fi[1]))))
        # Boundary conditions.
        vertex_num = mesh.NumOfVertices()
        dirichlet_dof = []
        for vi in range(vertex_num):
            vx, vy, vz = mesh.py_vertex(vi)
            if vz < dx / 2:
                deformable.SetDirichletBoundaryCondition(3 * vi, vx)
                deformable.SetDirichletBoundaryCondition(3 * vi + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * vi + 2, vz)
                dirichlet_dof.append(vi)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        dofs = deformable.dofs()
        print('Plant element: {:d}, DoFs: {:d}.'.format(mesh.NumOfElements(), dofs))
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._mesh = mesh
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._stepwise_loss = True
        self.__dirichlet_dof = dirichlet_dof

        # Optional data members for rendering.
        scale = 0.5
        self._spp = options['spp'] if 'spp' in options else 4
        self._camera_pos = (0.4, -1, .25)
        self._camera_lookat = scale/2 * ndarray(np.ones(3))
        self._color = (0.3, 0.9, 0.3)
        self._scale = scale

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
        
        self.onehot_py_verticesIdxs = np.eye(29763)[self.py_verticesIdxs]
        self.onehot_py_verticesIdxs = np.sum(self.onehot_py_verticesIdxs,axis=0,dtype=np.int)
    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return dof in self.__dirichlet_dof

    def _stepwise_loss_and_grad(self, q, v, i):
        mesh_file = self._folder / 'groundtruth' / '{:04d}.bin'.format(i)
        if not mesh_file.exists(): return 0, np.zeros(q.size), np.zeros(q.size)

        mesh = HexMesh3d()
        mesh.Initialize(str(mesh_file))
        
        q_ref = np.where(self.onehot_py_verticesIdxs,q_ref,0)
        q = np.where(self.onehot_py_verticesIdxs,q,0)
        q = q[self.onehot_py_verticesIdxs]
        grad = q - q_ref
        loss = 0.5 * grad.dot(grad)
        return loss, grad, np.zeros(q.size)
    
    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': self._camera_pos,
            'camera_lookat': self._camera_lookat,
            'resolution': self._resolution
        }
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=self._color, transforms=[
            ('s', self._scale),
        ])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 3)])
        
        q = ndarray(mesh.py_vertices())
        markers = q[self.py_verticesIdxs].reshape(-1,3)
        for marker in markers:
            renderer.add_shape_mesh({ 'name': 'sphere', 'center': marker, 'radius': 0.025 },
                transforms=[('s', 0.4)], color=(0.9, 0.1, 0.1))
        
        renderer.render()