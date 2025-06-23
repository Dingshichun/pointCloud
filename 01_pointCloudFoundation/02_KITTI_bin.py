"""
读取 KITTI 数据集中 bin 格式的点云数据，并实现可视化
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt  # 作图的模块
from mpl_toolkits.mplot3d import Axes3D


def load_kitti_bin(bin_path: str, remove_reflectance: bool = False) -> np.ndarray:
    """
    加载KITTI点云数据（.bin格式）

    参数:
        bin_path: .bin 文件路径
        remove_reflectance: 是否移除反射强度信息（默认保留）

    返回:
        N×3 或 N×4 的 NumPy 数组（XYZ 或 XYZ + 反射强度）
    """
    # 读取 .bin 格式要使用 np.fromfile()
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # 点云数据的第四列是反射强度，要移除的话就只返回前三列数据 XYZ
    return point_cloud[:, :3] if remove_reflectance else point_cloud


def visualize_with_open3d(points: np.ndarray, color_mode: str = "height") -> None:
    """
    使用Open3D可视化点云（支持颜色渲染）

    参数:
        points: 点云数组（N×3或N×4）
        color_mode:
            "height": 基于 Z 轴高程着色（默认）
            "intensity": 基于反射强度着色（需 N×4 输入）
            "uniform": 单色显示
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 仅需 XYZ 坐标

    # 颜色渲染策略
    if color_mode == "height":
        # 每行序号从 0 开始，2 是 Z 的值
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        colors = plt.cm.viridis((points[:, 2] - z_min) / (z_max - z_min))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif color_mode == "intensity" and points.shape[1] > 3:
        intensity = points[:, 3] / np.max(points[:, 3])  # 归一化反射强度
        colors = plt.cm.plasma(intensity)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:  # 单色模式
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

    # 可视化设置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="KITTI点云可视化", width=800, height=600)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.5  # 点大小调整
    vis.get_render_option().background_color = np.array([0, 0, 0])  # 黑色背景
    vis.run()
    vis.destroy_window()  # 销毁窗口


# 示例调用
if __name__ == "__main__":
    bin_path = "./01_pointCloudFoundation/data/000000.bin"  # 数据实际路径

    # 加载点云（保留反射强度）
    points = load_kitti_bin(bin_path, remove_reflectance=False)

    # 可视化（高程着色）
    visualize_with_open3d(points, color_mode="height")

    # 可选：反射强度着色（需数据含强度值）
    # visualize_with_open3d(points, color_mode="intensity")
