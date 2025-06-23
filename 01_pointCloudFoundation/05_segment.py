"""
实现 KITTI 数据中 bin 格式点云数据的分割
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# === 1. 数据加载与预处理 ===
def load_txt_data(file_path:str) -> np.ndarray:
    """加载 txt 格式点云数据"""
    points = np.loadtxt(file_path, dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # 添加颜色信息（如果原始数据包含）
    if points.shape[1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    return pcd

def load_bin_data(file_path:str) -> np.ndarray:
    """加载 bin 格式点云数据"""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

# 加载数据，这里加载 bin 格式数据
# skull_txt = load_txt_data("01_pointCloudFoundation/data/Skull.txt")
KITTI_bin = load_bin_data("./pointData/000002.bin")

# 预处理：去噪与降采样
cl, ind = KITTI_bin.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
skull_pcd = KITTI_bin.select_by_index(ind)
skull_pcd = skull_pcd.voxel_down_sample(voxel_size=0.5)  # 根据点云密度调整

# === 2. 法线估计（聚类必需）===
# KDTreeSearchParamHybrid 是混合搜索策略，radius 搜索半径，max_nn 最大邻域点数
KITTI_bin.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
)

# 法线方向调整。PCA 法线方向存在二义性（可能指向表面内外两侧），导致相邻法线方向不一致
# 步骤：随机选择一个种子点，传播其法线方向，对邻域内的点，
# 若当前法线与种子点法线夹角大于 90 度，则翻转方向，迭代直至覆盖全部点云。
# 参数 k=15 就是传播过程中的邻域点数
KITTI_bin.orient_normals_consistent_tangent_plane(k=15)

# === 3. 点云分割方法 ===
def dbscan_clustering(pcd, eps=1.5, min_points=15):
    """DBSCAN 密度聚类分割"""
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_
    
    # 可视化分割结果
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 噪声点设为黑色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, labels

# 执行 DBSCAN 聚类分割
segmented_pcd, cluster_labels = dbscan_clustering(KITTI_bin, eps=1.2, min_points=10)

# === 4. 深度学习辅助分割（可选）===
# try:
#     # 若安装 SAM 模型支持包
#     from segmentAnything import segmentObjectsFromEmbeddings
#     # 提取 SAM 特征嵌入
#     embeddings = extractEmbeddings(sam_model, segmented_pcd)
#     # 基于嵌入的分割
#     masks = segmentObjectsFromEmbeddings(sam_model, embeddings, segmented_pcd)
#     # 此处可添加后处理逻辑...
# except ImportError:
#     print("未安装SAM支持包，跳过深度学习分割")

# === 5. 结果可视化与输出 ===
o3d.visualization.draw_geometries([segmented_pcd], 
                                  window_name="点云分割结果",
                                  width=800,height=800)

# 保存分割结果
# o3d.io.write_point_cloud("skull_segmented.ply", segmented_pcd)
# np.savetxt("cluster_labels.txt", cluster_labels, fmt="%d")