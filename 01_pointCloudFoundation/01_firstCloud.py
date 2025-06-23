"""
简单读取点云数据并可视化
"""

"""
# 1.读取 pcd 格式的点云数据
import open3d as o3d

pcd=o3d.io.read_point_cloud("./01_pointCloudFoundation/data/bunny.pcd")
print(pcd)
o3d.visualization.draw_geometries([pcd],width=800,height=800)
"""

# 2.读取 txt 格式。
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud(
    "./01_pointCloudFoundation/data/Skull.txt", format="xyz"
)  # 读取 txt 数据
print(pcd)  # 输出点云的点个数
# print(np.asarray(pcd.points)) # 打印点的三维坐标
print("给所有的点上一个统一的颜色，颜色是在 RGB 空间的 [0，1] 范围内的值")
# pcd.paint_uniform_color([0, 1, 0]) # 绿色
# o3d.io.write_point_cloud("Skull.pcd", pcd) # 将读取到的数据保存为 pcd 格式
o3d.visualization.draw_geometries([pcd], width=800, height=800)  # 可视化并设置窗口大小
