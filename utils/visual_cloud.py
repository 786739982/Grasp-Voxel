import numpy as np
import open3d as o3d

points = np.load("cloud_sampled.npy")
points = np.load('devoxel_pc.npy')
# 创建点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 保存为 PLY 文件
o3d.io.write_point_cloud("output.ply", point_cloud)

print("点云数据已保存为 PLY 文件！")

o3d.visualization.draw_geometries([point_cloud], window_name='Voxel Grid Visualization')