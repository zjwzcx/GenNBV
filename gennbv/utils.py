import json
import logging
import os
import time
import gym
import numpy as np
import torch
import open3d as o3d
from typing import Callable
from PIL import Image
from torchvision.transforms import ToPILImage
from stable_baselines3.common.utils import set_random_seed
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import set_seed
from legged_gym import OPEN_ROBOT_ROOT_DIR
import xml.etree.ElementTree as etxml
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

transform_normdepth2PIL = ToPILImage()


def bresenham3D_pycuda(pts_source, pts_target, map_size):
    if isinstance(map_size, list):
        assert len(map_size) == 3 and map_size[0] == map_size[1] == map_size[2], "map_size must be cubic"
        map_size = map_size[0]

    # Keep data on GPU if already there
    device = pts_source.device
    source_pts = pts_source.int().contiguous()
    target_pts = pts_target.int().contiguous()
    num_rays = target_pts.shape[0]
    
    # Optimize max_pts_per_ray calculation based on manhattan distance
    # max_pts_per_ray = min(map_size * 3, int(1.2 * torch.max(torch.abs(target_pts - source_pts.expand_as(target_pts)).sum(dim=1))))
    max_pts_per_ray = map_size * 3
    
    # Allocate output memory directly on GPU
    trajectory_pts = torch.zeros((num_rays, max_pts_per_ray, 3), dtype=torch.int32, device=device)
    trajectory_lengths = torch.zeros(num_rays, dtype=torch.int32, device=device)

    kernel_code = """
    __device__ __forceinline__ int max3(int a, int b, int c) {
        return max(max(a, b), c);
    }

    __device__ void bresenham_line_3d(
        const int x0, const int y0, const int z0,
        const int x1, const int y1, const int z1,
        int *__restrict__ trajectory,
        int *__restrict__ length,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int dx = abs(x1 - x0);
        const int dy = abs(y1 - y0);
        const int dz = abs(z1 - z0);
        
        const int sx = (x0 < x1) ? 1 : -1;
        const int sy = (y0 < y1) ? 1 : -1;
        const int sz = (z0 < z1) ? 1 : -1;
        
        const int dm = max3(dx, dy, dz);
        int x = x0, y = y0, z = z0;
        int idx = 0;
        
        #pragma unroll 1
        if (dm == dx) {
            int p1 = 2 * dy - dx;
            int p2 = 2 * dz - dx;
            
            // Pre-compute bounds check
            const bool x_valid = (x >= 0 && x < map_size);
            const bool y_valid = (y >= 0 && y < map_size);
            const bool z_valid = (z >= 0 && z < map_size);
            
            if (x_valid && y_valid && z_valid) {
                trajectory[idx * 3] = x;
                trajectory[idx * 3 + 1] = y;
                trajectory[idx * 3 + 2] = z;
                idx++;
            }
            
            #pragma unroll 4
            for (int i = 0; i < dx && idx < max_pts_per_ray; i++) {
                if (p1 >= 0) { y += sy; p1 -= 2 * dx; }
                if (p2 >= 0) { z += sz; p2 -= 2 * dx; }
                x += sx;
                p1 += 2 * dy;
                p2 += 2 * dz;
                
                if (x >= 0 && x < map_size && 
                    y >= 0 && y < map_size && 
                    z >= 0 && z < map_size) {
                    trajectory[idx * 3] = x;
                    trajectory[idx * 3 + 1] = y;
                    trajectory[idx * 3 + 2] = z;
                    idx++;
                }
            }
        } else if (dm == dy) {
            // Similar optimizations for dy dominant case
            int p1 = 2 * dx - dy;
            int p2 = 2 * dz - dy;
            
            if (x >= 0 && x < map_size && 
                y >= 0 && y < map_size && 
                z >= 0 && z < map_size) {
                trajectory[idx * 3] = x;
                trajectory[idx * 3 + 1] = y;
                trajectory[idx * 3 + 2] = z;
                idx++;
            }
            
            #pragma unroll 4
            for (int i = 0; i < dy && idx < max_pts_per_ray; i++) {
                if (p1 >= 0) { x += sx; p1 -= 2 * dy; }
                if (p2 >= 0) { z += sz; p2 -= 2 * dy; }
                y += sy;
                p1 += 2 * dx;
                p2 += 2 * dz;
                
                if (x >= 0 && x < map_size && 
                    y >= 0 && y < map_size && 
                    z >= 0 && z < map_size) {
                    trajectory[idx * 3] = x;
                    trajectory[idx * 3 + 1] = y;
                    trajectory[idx * 3 + 2] = z;
                    idx++;
                }
            }
        } else {
            // Similar optimizations for dz dominant case
            int p1 = 2 * dx - dz;
            int p2 = 2 * dy - dz;
            
            if (x >= 0 && x < map_size && 
                y >= 0 && y < map_size && 
                z >= 0 && z < map_size) {
                trajectory[idx * 3] = x;
                trajectory[idx * 3 + 1] = y;
                trajectory[idx * 3 + 2] = z;
                idx++;
            }
            
            #pragma unroll 4
            for (int i = 0; i < dz && idx < max_pts_per_ray; i++) {
                if (p1 >= 0) { x += sx; p1 -= 2 * dz; }
                if (p2 >= 0) { y += sy; p2 -= 2 * dz; }
                z += sz;
                p1 += 2 * dx;
                p2 += 2 * dy;
                
                if (x >= 0 && x < map_size && 
                    y >= 0 && y < map_size && 
                    z >= 0 && z < map_size) {
                    trajectory[idx * 3] = x;
                    trajectory[idx * 3 + 1] = y;
                    trajectory[idx * 3 + 2] = z;
                    idx++;
                }
            }
        }
        
        *length = idx;
    }

    __global__ void ray_casting_kernel_3d(
        const int *__restrict__ source_pts,
        const int *__restrict__ target_pts,
        int *__restrict__ trajectory_pts,
        int *__restrict__ trajectory_lengths,
        const int num_rays,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= num_rays) return;
        
        const int src_x = source_pts[0];
        const int src_y = source_pts[1];
        const int src_z = source_pts[2];
        const int tgt_x = target_pts[ray_idx * 3];
        const int tgt_y = target_pts[ray_idx * 3 + 1];
        const int tgt_z = target_pts[ray_idx * 3 + 2];
        
        bresenham_line_3d(
            src_x, src_y, src_z,
            tgt_x, tgt_y, tgt_z,
            &trajectory_pts[ray_idx * max_pts_per_ray * 3],
            &trajectory_lengths[ray_idx],
            map_size,
            max_pts_per_ray
        );
    }
    """
    
    # Compile kernel with optimization flags
    mod = SourceModule(kernel_code, options=['-O3'])
    ray_casting_kernel = mod.get_function("ray_casting_kernel_3d")
    
    # Configure kernel launch parameters
    block_size = 256
    grid_size = (num_rays + block_size - 1) // block_size
    
    # Launch kernel with streamed execution
    stream = cuda.Stream()
    ray_casting_kernel(
        source_pts,
        target_pts,
        trajectory_pts,
        trajectory_lengths,
        np.int32(num_rays),
        np.int32(map_size),
        np.int32(max_pts_per_ray),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        stream=stream
    )
    
    # Process results efficiently using GPU operations
    mask = torch.arange(max_pts_per_ray, device=device)[None, :] < trajectory_lengths[:, None]
    mask = mask.unsqueeze(-1).expand(-1, -1, 3)
    results = trajectory_pts[mask].view(-1, 3)

    return results.to(torch.long)


def scanned_pts_to_idx_3D(pts_target, range_gt, voxel_size_gt, map_size=256):
    """
    Params:
        pts_target: [num_env, num_pts, 3], list of torch.tensor, target points by back-projection
        range_gt_scenes: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        voxel_size_scenes: [num_env, 3]

    Return:
        pts_target_idxs: list of (num_valid_pts_idx, 3)
    """
    num_env = len(pts_target)

    xyz_max_voxel = range_gt[:, [0,2,4]] + 0.5 * voxel_size_gt
    xyz_min_voxel = range_gt[:, [1,3,5]] - 0.5 * voxel_size_gt

    pts_target_idxs = []
    for env_idx in range(num_env):
        # Convert current environment points to torch tensor
        pts_env = pts_target[env_idx]
        
        # Convert to indices
        pts_target_idx = torch.floor(
            (pts_env - xyz_min_voxel[env_idx]) / voxel_size_gt[env_idx]
        ).long()

        # Bounds checking masks
        bound_mask = (xyz_max_voxel[env_idx] > pts_env) & (pts_env > xyz_min_voxel[env_idx])
        bound_mask = torch.all(bound_mask, dim=-1)  # [num_pts]

        valid_pts = pts_target_idx[bound_mask]

        if len(valid_pts) == 0:
            pts_target_idxs.append([])
            continue

        # Unique and clip
        valid_pts = torch.unique(valid_pts, dim=0)
        valid_pts = torch.clamp(valid_pts, min=0, max=map_size-1)
        pts_target_idxs.append(valid_pts)

    return pts_target_idxs


def pose_coord_to_idx_3D(poses, range_gt, voxel_size_gt, map_size=256, if_col=False):
    """
    Accelerated 3D version of pose coordinate to index conversion
    
    Params:
        poses: [num_step, 3], x-y-z pose
        range_gt: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        map_size: int, size of the voxel grid (default: 256)
    
    Return:
        poses_idx: [num_env, 3]
    """
    # Extract minimum bounds for each dimension
    x_min = range_gt[:, 1]  # [num_env]
    y_min = range_gt[:, 3]  # [num_env]
    z_min = range_gt[:, 5]  # [num_env]
    
    # Stack minimum bounds
    xyz_min = torch.stack([x_min, y_min, z_min], dim=-1)  # [num_env, 3]
    
    # Calculate voxel boundaries with offset
    xyz_min_voxel = xyz_min - 0.5 * voxel_size_gt  # [num_env, 3]
    
    assert poses.shape[1] == 3, f"Invalid poses shape: {poses.shape}"
    poses_idx = ((poses - xyz_min_voxel) / voxel_size_gt).floor().long()

    # if not if_col:
    #     # for computing ray casting, clip values to valid range
    #     poses_idx = torch.clip(poses_idx, min=0, max=map_size-1)
    if if_col:
        # for collision check
        poses_idx[(poses_idx < 0).any(dim=-1)] = -1
        poses_idx[(poses_idx > map_size-1).any(dim=-1)] = -1
    return poses_idx


def grid_occupancy_tri_cls(grid_prob, threshold_occu=0.5, threshold_free=0.0, return_tri_cls_only=False):
    """
    Params:
        grid_prob: [num_env, X, Y, Z], from self.grid_backproj[..., 3]

    Return:
        grid_occupancy: [num_env, X, Y, Z], voxel value among {0/1}. 0: free/unknown, 1: occupied
        grid_tri_cls: [num_env, X, Y, Z], voxel value among {-1/0/1}. -1: free, 0: unknown, 1: occupied
    """
    grid_occupancy = (grid_prob > threshold_occu).to(torch.float32)
    grid_free = (grid_prob < threshold_free).to(torch.float32)

    grid_tri_cls = grid_occupancy - grid_free   # element value: {-1, 0, 1}
    if return_tri_cls_only:
        return grid_tri_cls
    else:
        return grid_occupancy, grid_tri_cls


def getURDFParameter(urdf_path, parameter_name: str):
    """Reads a parameter from a drone's URDF file.

    This method is nothing more than a custom XML parser for the .urdf
    files in folder `assets/`.

    Parameters
    ----------
    parameter_name : str
        The name of the parameter to read.

    Returns
    -------
    float
        The value of the parameter.

    """
    #### Get the XML tree of the drone model to control ########
    path = urdf_path
    URDF_TREE = etxml.parse(path).getroot()
    #### Find and return the desired parameter #################
    if parameter_name == 'm':
        return float(URDF_TREE[1][0][1].attrib['value'])
    elif parameter_name in ['ixx', 'iyy', 'izz']:
        return float(URDF_TREE[1][0][2].attrib[parameter_name])
    elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                            'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
        return float(URDF_TREE[0].attrib[parameter_name])
    elif parameter_name in ['length', 'radius']:
        return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
    elif parameter_name == 'collision_z_offset':
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        return COLLISION_SHAPE_OFFSETS[2]


def save_pcd(points, save_path):
    points = points.to("cpu")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_env_register(env_id: str, rank: int, seed: int = 1, args=None, env_cfg=None) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env_cfg.seed = seed
        env, env_config = task_registry.make_env(env_id, args, env_cfg)
        set_seed(seed)
        # return env, env_config
        return env
    set_random_seed(seed)
    return _init