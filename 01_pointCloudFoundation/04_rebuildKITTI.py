"""
重建 KITTI 数据集中的 bin 格式点云数据
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Union

# ==================== 1. 数据加载与预处理 ====================
def load_kitti_bin(bin_path: str, remove_reflectance: bool = True) -> np.ndarray:
    """
    加载 KITTI 点云数据（.bin格式）
    参数:
        bin_path: .bin 文件路径
        remove_reflectance: 是否移除反射强度（默认移除）
    返回:
        N×3 的 NumPy 数组（XYZ 坐标）
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3] if remove_reflectance else points # 默认不保留反射强度

def preprocess_pointcloud(points: np.ndarray, voxel_size: float = 0.1) -> o3d.geometry.PointCloud:
    """
    点云预处理：降采样 + 去噪
    参数:
        points: 原始点云坐标（N×3）
        voxel_size: 体素下采样尺寸（默认 0.1 米），0.1 米的范围内找一个代表点
    返回:
        预处理后的 Open3D 点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 体素下采样（加速处理）
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 统计滤波去除离群点
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd

# ==================== 2. 三维重建算法 ====================
def poisson_reconstruction(pcd: o3d.geometry.PointCloud, depth: int = 10) -> o3d.geometry.TriangleMesh:
    """
    泊松表面重建（适合封闭物体）
    参数:
        pcd: 预处理后的点云
        depth: 重建深度（越高细节越多，默认 10）
    返回:
        三角网格模型
    """
    # 估计法线（泊松重建必需）
    pcd.estimate_normals()
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh

def alpha_shapes_reconstruction(pcd: o3d.geometry.PointCloud, alpha: float = 2.0) -> o3d.geometry.TriangleMesh:
    """
    Alpha Shapes 重建（适合开放场景）
    参数:
        pcd: 预处理后的点云
        alpha: 控制表面光滑度（越大越平滑）
    返回:
        三角网格模型
    """
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh

# ==================== 3. 可视化与导出 ====================
def visualize_reconstruction(pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh) -> None:
    """
    交互式可视化点云与重建网格
    参数:
        pcd: 点云对象
        mesh: 网格对象
    """
    # 设置点云颜色（高程渲染）
    z_min, z_max = np.min(np.asarray(pcd.points)[:, 2]), np.max(np.asarray(pcd.points)[:, 2])
    colors = plt.cm.viridis((np.asarray(pcd.points)[:, 2] - z_min) / (z_max - z_min))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 设置网格颜色和法线
    mesh.paint_uniform_color([0.5, 0.7, 1.0])
    mesh.compute_vertex_normals()
    
    # 可视化
    o3d.visualization.draw_geometries(
        [pcd, mesh],
        window_name="KITTI 点云重建",
        width=1024,
        height=768,
        zoom=0.6,
        front=[0.3, -0.2, -0.9],
        lookat=[1, 1, 1],
        up=[0, -1, 0]
    )
    

# def export_to_web(mesh: o3d.geometry.TriangleMesh, output_path: str = "reconstruction.html") -> None:
#     """
#     导出重建结果为 Web 可交互模型
#     参数:
#         mesh: 网格对象
#         output_path: 输出 HTML 路径
#     """
#     import pyvista as pv
#     pv_mesh = pv.wrap(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
#     plotter = pv.Plotter()
#     plotter.add_mesh(pv_mesh, color='lightblue')
#     plotter.export_html(output_path)

# ==================== 4. 主流程 ====================
if __name__ == "__main__":
    # 配置参数
    bin_path = "./01_pointCloudFoundation/data/000000.bin"  # 数据的实际路径
    voxel_size = 0.2  # 下采样体素尺寸（米）
    reconstruction_method = "alpha"  # 可选："poisson" 或 "alpha"
    
    # Step 1: 加载并预处理点云
    points = load_kitti_bin(bin_path)
    pcd = preprocess_pointcloud(points, voxel_size=voxel_size)
    
    # Step 2: 三维重建
    if reconstruction_method == "poisson":
        mesh = poisson_reconstruction(pcd, depth=9)
    else:
        mesh = alpha_shapes_reconstruction(pcd, alpha=1.0)
    
    # Step 3: 可视化与导出
    visualize_reconstruction(pcd, mesh)
    # export_to_web(mesh, "kitti_reconstruction.html")