import numpy as np
import open3d as o3d

# 创建一个随机的 occupancy voxel grid
occupancy = np.load('voxel_pc.npy')
points = np.argwhere(occupancy==1)
print(points.shape)
# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化
o3d.visualization.draw_geometries([pcd], window_name='Voxel Grid Visualization')