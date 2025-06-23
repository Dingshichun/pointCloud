"""
实现点云配准全流程
"""

import numpy as np
import open3d as o3d
import os

# 1. KITTI点云加载（适配 .bin 格式）
def load_kitti_bin(bin_path):
    """
    加载 KITTI 的 .bin 点云文件
    :param bin_path: .bin文件路径
    :return: (N, 3)的XYZ坐标数组
    """
    data = np.fromfile(bin_path, dtype=np.float32)
    points = data.reshape(-1, 4)[:, :3]  # 提取 XYZ 坐标，忽略反射强度
    return points

# 2. 点云预处理（下采样+法线估计+方向统一）
def preprocess_point_cloud(pcd, voxel_size):
    """
    处理点云：下采样 -> 法线估计 -> 法线方向统一
    :param pcd: 输入点云
    :param voxel_size: 下采样体素尺寸
    :return: 带法线的下采样点云
    """
    # 体素下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 动态法线搜索半径（适应 KITTI 近密远疏特性）
    radius_normal = max(voxel_size * 3, 0.5)  # 确保最小半径 0.5 米
    
    # 法线估计（修复目标点云无法线问题）
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, 
            max_nn=50  # 增加邻域点数提升稳定性
        )
    )
    
    # 强制统一法线方向（避免内外翻转）
    pcd_down.orient_normals_consistent_tangent_plane(k=25)
    return pcd_down

# 3. 特征提取（FPFH 描述子）
def extract_fpfh(pcd, voxel_size):
    """
    计算 FPFH 特征描述子
    :param pcd: 带法线的点云
    :param voxel_size: 特征搜索半径基准
    :return: FPFH 特征
    """
    radius_feature = voxel_size * 5  # 特征搜索范围
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

# 4. 粗配准（RANSAC 基于特征匹配）
def coarse_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    """
    使用 RANSAC 进行初始变换估计
    :return: 初始变换矩阵
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result.transformation

# 5. 精配准（Point-to-Plane ICP）
def fine_registration(source, target, initial_trans, voxel_size):
    """
    执行 ICP 精配准
    :return: 最终变换矩阵, 配准精度
    """
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, 
        distance_threshold, 
        initial_trans,
        # 使用点到平面 ICP（依赖法线）
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return result.transformation, result.fitness

# 6. 主流程
def register_kitti_point_clouds(source_path, target_path):
    # 加载原始点云
    source_points = load_kitti_bin(source_path)
    target_points = load_kitti_bin(target_path)
    
    # 转为 Open3D 对象
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    
    # 预处理参数
    voxel_size = 0.3  # KITTI 城市场景建议值
    
    # 预处理点云（含法线估计）
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)  # 关键修复：目标点云法线计算
    
    # ---- 法线存在性验证（避免后续失败）----
    assert source_down.has_normals(), "源点云法线未生成！"
    assert target_down.has_normals(), "目标点云法线未生成！"  # 检查点
    
    # ---- 法线可视化检查（调试用）----
    source_down.paint_uniform_color([1, 0, 0])  # 红色：源点云
    target_down.paint_uniform_color([0, 0, 1])  # 蓝色：目标点云
    o3d.visualization.draw_geometries([source_down, target_down], 
                                      point_show_normal=True, 
                                      window_name="法线方向检查",
                                      width=1024, height=768)
    
    # 提取 FPFH 特征
    source_fpfh = extract_fpfh(source_down, voxel_size)
    target_fpfh = extract_fpfh(target_down, voxel_size)
    
    # 粗配准（RANSAC）
    initial_trans = coarse_registration(
        source_down, target_down, 
        source_fpfh, target_fpfh, 
        voxel_size
    )
    
    # 精配准（ICP）
    final_trans, fitness = fine_registration(
        source_down,  # 使用下采样点云加速
        target_down, 
        initial_trans,
        voxel_size
    )
    
    # 应用变换到原始点云
    source.transform(final_trans)
    
    # 结果可视化
    source.paint_uniform_color([1, 0, 0])  # 红色：配准后点云
    target.paint_uniform_color([0, 1, 0])  # 绿色：目标点云
    o3d.visualization.draw_geometries([source, target], 
                                      window_name="配准结果",
                                      width=1024, height=768)
    
    print(f"配准精度(fitness): {fitness:.4f} (1为最佳)")
    return final_trans

# 7. 示例调用
if __name__ == "__main__":
    # 实际路径。由于没有在不同视角采集的同一个场景的点云数据
    # 所以使用相同的点云文件进行配准，配准结果自然为 1。
    source_path = "./pointData/000000.bin"  
    target_path = "./pointData/000000.bin"  
    
    transformation_matrix = register_kitti_point_clouds(source_path, target_path)
    print("变换矩阵:\n", transformation_matrix)