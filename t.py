# # import torch

# # # 假设有一个 3x4x5 的张量
# # tensor = torch.rand(2, 3, 2)
# # print(tensor)

# # # 通过 topk 找到扁平化后最大的 5 个元素及其索引
# # topk_values, topk_flat_indices = torch.topk(tensor.view(-1), 3)

# # # 使用 torch.unravel_index 将扁平化索引转为原始张量的三维索引
# # unraveled_indices = torch.unravel_index(topk_flat_indices, tensor.shape)

# # # 将分开的多维索引堆叠在一起，使每一行代表一个多维索引
# # stacked_indices = torch.stack(unraveled_indices, dim=1)

# # print("扁平化索引:", topk_flat_indices)
# # print("多维索引:", unraveled_indices)
# # print("直接返回的多维索引:\n", stacked_indices)

# # print(tensor[unraveled_indices])

# import torch

# def matrix_to_viewpoint_params(batch_matrix):
#     """
#     Convert rotation matrices to direction vectors (axis_x) and in-plane rotation angles.
    
#     Input:
#         batch_matrix: [torch.FloatTensor, (N,3,3)]
#             Rotation matrices in batch
            
#     Output:
#         batch_towards: [torch.FloatTensor, (N, 3)]
#             Direction vectors in batch
#         batch_angle: [torch.FloatTensor, (N,)]
#             In-plane rotation angles in batch
#     """
#     # Step 1: Extract direction vectors (axis_x) - the first column of the rotation matrix
#     batch_towards = batch_matrix[:, :, 0]
    
#     # Step 2: Extract in-plane rotation angles
#     # The angle can be extracted from the 2x2 rotation matrix in the YZ-plane (bottom-right corner of R1)
#     cos_angle = batch_matrix[:, 1, 1]  # cos(angle) is located at (1, 1) of the matrix
#     sin_angle = batch_matrix[:, 1, 2]  # sin(angle) is located at (1, 2) of the matrix
    
#     # Compute the angle using arctan2 to account for the quadrant of the angle
#     batch_angle = torch.atan2(sin_angle, cos_angle)
    
#     return batch_towards, batch_angle


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """ Transform approach vectors and in-plane rotation angles to rotation matrices.

        Input:
            batch_towards: [torch.FloatTensor, (N,3)]
                approach vectors in batch
            batch_angle: [torch.floatTensor, (N,)]
                in-plane rotation angles in batch
                
        Output:
            batch_matrix: [torch.floatTensor, (N,3,3)]
                rotation matrices in batch
    """
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:,1], axis_x[:,0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y,1] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    batch_matrix = torch.matmul(R2, R1)
    return batch_matrix


def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (ZYX).
    
    Input:
        R: [torch.FloatTensor, (N, 3, 3)]
            Rotation matrices in batch
    
    Output:
        euler_angles: [torch.FloatTensor, (N, 3)]
            Euler angles in radians: [yaw, pitch, roll]
    """
    # Ensure input is a batch of rotation matrices
    assert R.shape[1:] == (3, 3)

    # Extract the elements of the rotation matrix
    m00 = R[:, 0, 0]
    m01 = R[:, 0, 1]
    m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]
    m11 = R[:, 1, 1]
    m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]
    m21 = R[:, 2, 1]
    m22 = R[:, 2, 2]

    # Calculate yaw (around Z-axis)
    yaw = torch.atan2(m10, m00)

    # Calculate pitch (around Y-axis)
    # Using the condition to prevent gimbal lock
    sy = -m20
    if torch.abs(sy) < 1.0:  # Normal case
        pitch = torch.asin(sy)
        roll = torch.atan2(m21, m22)
    else:  # Gimbal lock case
        pitch = torch.tensor(np.sign(sy) * (torch.pi / 2))
        roll = torch.atan2(-m12, m11)

    # Stack angles to form a tensor
    euler_angles = torch.stack([yaw, pitch, roll], dim=-1)
    
    return euler_angles

def viewpoint_params_to_euler_angles(batch_towards, batch_angle):
    """
    Convert approach vectors and in-plane rotation angles to Euler angles (ZYX).
    
    Input:
        batch_towards: [torch.FloatTensor, (N, 3)]
            Approach vectors in batch
        batch_angle: [torch.floatTensor, (N,)]
            In-plane rotation angles in batch
            
    Output:
        euler_angles: [torch.FloatTensor, (N, 3)]
            Euler angles in radians: [yaw, pitch, roll]
    """
    # 计算偏航 (Yaw)
    yaw = torch.atan2(batch_towards[:, 1], batch_towards[:, 0])

    # 计算俯仰 (Pitch)
    pitch = torch.asin(-batch_towards[:, 2])

    # 滚转 (Roll) 使用给定的平面旋转角度
    roll = batch_angle

    # 组合成一个张量
    euler_angles = torch.stack([yaw, pitch, roll], dim=-1)
    
    return euler_angles

def euler_angles_to_viewpoint_params(euler_angles):
    """
    Convert Euler angles (ZYX) to approach vectors and in-plane rotation angles.
    
    Input:
        euler_angles: [torch.FloatTensor, (N, 3)]
            Euler angles in radians: [yaw, pitch, roll]
            
    Output:
        batch_towards: [torch.FloatTensor, (N, 3)]
            Approach vectors in batch
        batch_angle: [torch.FloatTensor, (N,)]
            In-plane rotation angles in batch
    """
    # 解耦欧拉角
    yaw = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    roll = euler_angles[:, 2]
    
    # 计算朝向向量
    # X分量 = cos(yaw) * cos(pitch)
    # Y分量 = sin(yaw) * cos(pitch)
    # Z分量 = -sin(pitch)
    batch_towards = torch.stack([
        torch.cos(yaw) * torch.cos(pitch),
        torch.sin(yaw) * torch.cos(pitch),
        -torch.sin(pitch)
    ], dim=-1)
    
    # 旋转角度直接取roll
    batch_angle = roll
    
    return batch_towards, batch_angle


# # Example usage:
# batch_matrix = torch.tensor([
#     [[1, 0, 0],
#      [0, 0.7071, -0.7071],
#      [0, 0.7071, 0.7071]],
    
#     # [[0, 0, 1],
#     #  [0, 1, 0],
#     #  [-1, 0, 0]],

#     # [[0, -1, 0],
#     #  [1, 0, 0],
#     #  [0, 0, 1]],
# ], dtype=torch.float32)

# batch_towards, batch_angle = matrix_to_viewpoint_params(batch_matrix)
# batch_matrix_ = batch_viewpoint_params_to_matrix(batch_towards, batch_angle)
# print("Direction vectors (batch_towards):\n", batch_towards)
# print("In-plane rotation angles (batch_angle):\n", batch_angle)
# print(batch_matrix_)

# import torch

# # 假设有一个 300x3 的整数点云张量
# cloud = torch.tensor([[4, 3, 2],
#                     [4, 3, 2],
#                     [1, 1, 8],
#                     [4, 9, 1],
#                     [8, 5, 6],
#                     [4, 8, 8],
#                     [4, 6, 2],
#                     [9, 3, 2],
#                     [1, 3, 2],
#                     [8, 6, 3]])

# # 假设有一个单独的点
# point = cloud[0]  # 示例单点

# # 使用 torch.isclose 进行比较并找到索引
# # 由于值为整数，所以可以直接进行比较
# indices = torch.argwhere((cloud == point).all(dim=1))

# print("点云:")
# print(cloud)
# print("查找的点:")
# print(point)
# print("索引:")
# print(indices.squeeze(), indices.squeeze().shape)

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_vector_and_roll(yaw, pitch, roll):
    # 假设输入的角度是弧度
    # 计算方向向量
    direction_vector = torch.zeros(3)
    direction_vector[0] = torch.cos(yaw) * torch.cos(pitch)  # x 分量
    direction_vector[1] = torch.sin(yaw) * torch.cos(pitch)  # y 分量
    direction_vector[2] = torch.sin(pitch)                   # z 分量
    
    # 平面内的旋转角度就是 roll
    in_plane_rotation = roll
    
    return direction_vector, in_plane_rotation

yaw = torch.tensor(0)   # 假设 yaw, pitch, roll 都是弧度
pitch = torch.tensor(0)
roll = torch.tensor(0)

direction_vector = torch.tensor([0, 1, 0], dtype=torch.float32)
in_plane_rotation = torch.tensor(0.5, dtype=torch.float32)

martix = batch_viewpoint_params_to_matrix(direction_vector.unsqueeze(0), in_plane_rotation.unsqueeze(0))
euler = rotation_matrix_to_euler_angles(martix)
euler_ = viewpoint_params_to_euler_angles(direction_vector.unsqueeze(0), in_plane_rotation.unsqueeze(0))
print(f"旋转矩阵: {martix}")
print(f"欧拉角: {euler}")
print(f"欧拉角_: {euler_}")
direction_vector, in_plane_rotation = euler_angles_to_viewpoint_params(euler_)
print(f"方向向量: {direction_vector}", f"平面内旋转角度: {in_plane_rotation}")

direction_vector = np.array([0, 0, 1])
in_plane_rotation = np.array(0)
martix = R.from_rotvec(direction_vector * in_plane_rotation).as_matrix()
print(f"旋转矩阵: {martix}")

martix = np.array([[  0., 0., -1],
                    [ 0., 1., 0.],
                    [ 1., 0., 0.]])
rotvec = R.from_matrix(martix).as_rotvec()
euler = R.from_matrix(martix).as_euler('zyx', degrees=False)
print(f"旋转向量: {rotvec}")
print(f"欧拉角: {euler}")




# direction_vector, in_plane_rotation = euler_to_vector_and_roll(yaw, pitch, roll)
# martix = batch_viewpoint_params_to_matrix(direction_vector.unsqueeze(0), in_plane_rotation.unsqueeze(0))

# print(f"方向向量: {direction_vector}")
# print(f"平面内旋转角度: {in_plane_rotation}")

# martix = R.from_euler('zyx', [0.5, 0.8, 0.3]).as_matrix()
# print(f"旋转矩阵: {martix}")

# martix = np.array([[[ 0.6114, -0.6441, -0.4597],
#                     [ 0.3340,  0.7368, -0.5879],
#                     [ 0.7174,  0.2059,  0.6656]]])
# martix = np.array([[[1, 0, 0],
#                     [0, 0.7071, -0.7071],
#                     [0, 0.7071, 0.7071]]])
# martix = np.array([[[1, 0, 0],
#                     [0, 1, 0],
# #                     [0, 0, 1]]])
# euler = R.from_matrix(martix).as_euler('xyz', degrees=False)
# print(f"欧拉角: {euler}")
