"""
三维重建头盖骨模型，点云数据是 Skull.txt
"""

import open3d as o3d
import numpy as np

# === 1. 数据加载与预处理 ===
# 从文本文件读取点云数据（假设每行数据格式为 XYZ）
points = np.loadtxt("./01_pointCloudFoundation/data/Skull.txt", dtype=np.float32)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 由于下载的 Skull.txt 点云数据已经过去噪，所以此处省略，也不必下采样。
# 去噪滤波（移除统计离群点）
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
# pcd = pcd.select_by_index(ind)

# 降采样（提升计算效率）
# pcd = pcd.voxel_down_sample(voxel_size=0.05)  # 体素尺寸根据点云密度调整

# === 2. 法线估计 ===
# 计算点云法向量（泊松重建必需）
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)  # 搜索半径影响曲面光滑度
pcd.orient_normals_consistent_tangent_plane(k=15)  # 统一法线方向

# === 3. 表面重建 ===
# 泊松重建（生成封闭网格，适合封闭环境）
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9  # 深度值控制细节层级（建议 8-10）
)

# === 4. 网格优化 ===
# 移除低密度区域（过滤重建噪声）
vertices_to_remove = densities < np.quantile(densities, 0.03)
mesh.remove_vertices_by_mask(vertices_to_remove)

# 平滑与简化
mesh = mesh.filter_smooth_taubin(number_of_iterations=10)  # 平滑曲面
mesh = mesh.simplify_vertex_clustering(  # 下采样简化网格
    voxel_size=0.01, contraction=o3d.geometry.SimplificationContraction.Average
)

# === 5. 结果输出与可视化 ===
# 保存重建模型
# o3d.io.write_triangle_mesh("skull_reconstructed.obj", mesh)  # OBJ 格式
# o3d.io.write_triangle_mesh("skull_reconstructed.ply", mesh)  # PLY 格式（可选）

# 可视化对比
o3d.visualization.draw_geometries(
    [pcd], window_name="原始点云",width=800,height=800
)
o3d.visualization.draw_geometries(
    [mesh], window_name="重建网格", width=800,height=800,mesh_show_back_face=True
)