# -*- coding: UTF-8 -*-
'''
=================================================
@path   ：3D_Edge_Detection -> test
@IDE    ：PyCharm
@Author ：dell
@Date   ：2021/8/18 9:58
==================================================
'''
__author__ = 'dell'

import os
import numpy as np
import open3d as o3d
import pandas as pd
from pyntcloud import PyntCloud
# from vedo import *
from vedo import load


def extract_vertices(input_str):
    """
    将提取的点保存到一个点云文件
    :param input_str: 输入文件名或路径
    :return:
    """
    mesh = o3d.io.read_triangle_mesh(input_str)  # 读取文件中的 `TriangleMesh`
    # 判断读取的文件是 `stl`文件，则移除重复的点
    if os.path.splitext(input_str)[-1] == ".stl":
        mesh.remove_duplicated_vertices()
    # 点云由点坐标、点颜色和点法线组成,转成三维向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    return pcd


def ransac_seg(input_pcd, distance_threshold, ramsac_n, num_iterations, mode):
    """
    用RANSAC算法在点云上分割出一个平面，选择少于或者多于平面的点
    :param input_pcd:输入点云
    :param distance_threshold:点与平面的最大距离仍然被认为是一个内嵌体（内点）
    :param ramsac_n:在每次迭代中被认为是内嵌的点的数目
    :param num_iterations:迭代次数
    :param mode:0表示少于，1表示多于
    :return:
    """

    # segment_plane():利用RANSAC算法在点云中分割平面,返回Tuple[numpy.ndarray[float64[4, 1]], List[int]]，函数返回（a, b, c,
    # d）作为一个平面，和内点索引的列表
    plane_model, inliers = input_pcd.segment_plane(distance_threshold, ramsac_n, num_iterations)
    [a, b, c, d] = plane_model
    print(f"plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # select_by_index()从输入网格选出输出网格
    # 选择平面内的点
    inlier_cloud = input_pcd.select_by_index(inliers)
    # 选择平面外的点
    outlier_cloud = input_pcd.select_by_index(inliers, invert=True)

    dists = input_pcd.compute_point_cloud_distance(inlier_cloud)
    dists = np.asarray(dists)
    index = np.where(dists > 0.01)[0]
    pcd_without_inlier_cloud = input_pcd.select_by_index(index)

    # get_axis_aligned_bounding_box()集合图形的轴对齐边界
    axis_aligned_bounding_box = inlier_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)  # 设置边框颜色

    tmp_pts = np.asarray(pcd_without_inlier_cloud.points)
    tmp_cls = np.asarray(pcd_without_inlier_cloud.colors)

    final_pts = []
    final_cls = []
    # 0表示少于，1表示多于
    if mode == 0:
        for i in range(tmp_pts.shape[0]):
            if tmp_pts[i][0] * a + tmp_pts[i][1] * b + tmp_pts[i][2] * c + d < 0:
                final_pts.append(tmp_pts[i])
                final_cls.append(tmp_cls[i])
            else:
                continue
        output = o3d.geometry.PointCloud()
        output.points = o3d.utility.Vector3dVector(np.array(final_pts))
        output.colors = o3d.utility.Vector3dVector(np.array(final_cls))
        return output, axis_aligned_bounding_box, inlier_cloud, outlier_cloud
    else:
        for i in range(tmp_pts.shape[0]):
            if tmp_pts[i][0] * a + tmp_pts[i][1] * b + tmp_pts[i][2] * c + d > 0:
                final_pts.append(tmp_pts[i])
                final_cls.append(tmp_cls[i])
            else:
                continue
        output = o3d.geometry.PointCloud()
        output.points = o3d.utility.Vector3dVector(np.array(final_pts))
        output.colors = o3d.utility.Vector3dVector(np.array(final_cls))
        return output, axis_aligned_bounding_box, inlier_cloud, outlier_cloud


if __name__ == '__main__':
    # 加载和合并点
    # 提取.stl文件中的点
    # Mandibular.stl
    vertices_pts = extract_vertices('Example.stl')
    # 将提取到的点保存到文件“vertices.pcd”中
    o3d.io.write_point_cloud('vertices.pcd', vertices_pts)

    # 读取 “vertices.pcd”文件
    input_pts = PyntCloud.from_file("vertices.pcd")

    # 加载和计算网格曲率
    mesh = load('Example.stl')

    # 加载参照网格
    ref_mesh = o3d.io.read_triangle_mesh('Example.stl')

    k_n = 20
    threshold = 0.003

    # ————————————————————————Edge Detection —————————————————————————— #
    pcd_np = np.zeros((len(input_pts.points), 6))

    # 找邻点
    kdtree_id = input_pts.add_structure("kdtree")
    k_neighbors = input_pts.get_neighbors(k=k_n, kdtree=kdtree_id)

    # 计算特征值
    ev = input_pts.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = input_pts.points['x'].values
    y = input_pts.points['y'].values
    z = input_pts.points['z'].values

    e1 = input_pts.points['e3(' + str(k_n + 1) + ')'].values
    e2 = input_pts.points['e2(' + str(k_n + 1) + ')'].values
    e3 = input_pts.points['e1(' + str(k_n + 1) + ')'].values

    sum_eg = np.add(np.add(e1, e2), e3)  # 求和
    sigma = np.divide(e1, sum_eg)  # 计算比例
    sigma_value = sigma

    # 保存边缘信息和点云
    thresh_min = sigma_value < threshold
    sigma_value[thresh_min] = 0
    thresh_max = sigma_value > threshold
    sigma_value[thresh_max] = 255

    pcd_np[:, 0] = x
    pcd_np[:, 1] = y
    pcd_np[:, 2] = z
    pcd_np[:, 3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:, 3] == 0), axis=0)  # 删除 sigma_value = 0 的那行值（axis = 0 表示删掉一行）
    # print(edge_np)

    clmns = ['x', 'y', 'z', 'red', 'green', 'blue']
    # DataFrame是二维数据结构，即数据以行和列的表格方式排列
    pcd_pd = pd.DataFrame(data=pcd_np, columns=clmns)
    pcd_pd['red'] = sigma_value.astype(np.uint8)

    edge_points = PyntCloud(pd.DataFrame(data=pcd_np, columns=clmns))  # 加载点坐标

    PyntCloud.to_file(edge_points, "edges.ply")

    input_pts = o3d.io.read_point_cloud("edges.ply")
    o3d.visualization.draw_geometries([input_pts])

    # 修改下面的代码,打印出平面等式
    # ——————————————————————————RANSAC1———————————————————————— #
    # ransac_seg(input_pcd, distance_threshold, ramsac_n, num_iterations, mode)
    ransac_result, _, _, _ = ransac_seg(input_pts, 0.8, 10, 1000, 0)

    # ——————————————————————————RANSAC2———————————————————————— #
    ransac_result, _, _, _ = ransac_seg(ransac_result, 0.5, 10, 1000, 1)
    # print(ransac_result)

    # ——————————————————————————RANSAC3———————————————————————— #
    ransac_result, _, tmp_inlier, _ = ransac_seg(ransac_result, 4, 10, 1000, 1)

    o3d.io.write_point_cloud("result.pcd", tmp_inlier)

    # 获得曲率数组
    mesh.addCurvatureScalars(1)
    curvature = np.asarray(mesh.getPointArray())

    # 加载并合并点坐标
    feature_pts = o3d.io.read_point_cloud('result.pcd')
    vertices_pts.paint_uniform_color([0.5, 0.5, 0.5])

    # 画图
    # 合并下面两个点集
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.append(np.asarray(vertices_pts.points), np.asarray(feature_pts.points), axis=0))
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.append(np.asarray(vertices_pts.colors), np.asarray(feature_pts.colors), axis=0))
    print(merged_pcd)

    # 从合并变量中创造KDTree
    pcd_tree = o3d.geometry.KDTreeFlann(merged_pcd)

    result = []

    # 通过检测1-NN曲率枚举所有特征点
    for idx, val in enumerate(merged_pcd.points):
        while idx >= np.shape(vertices_pts.points)[0]:
            # return Tuple[int, open3d.utility.IntVector, open3d.utility.DoubleVector]
            [_, index, _] = pcd_tree.search_hybrid_vector_3d(merged_pcd.points[idx], 0.3, 2)
            if idx == index[1] and index[0] < np.shape(vertices_pts.points)[0] and curvature[index[0]] < -1:
                result.append(merged_pcd.points[index[0]])
            else:
                break
            break
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(result))
    # voxel_down_sample函数将输入点云向下采样到具有体素的输出点云。如果存在的话，法线和颜色是平均的
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=1)
    # compute_vertex_normals（）计算顶点法线，通常在渲染之前调用
    ref_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([ref_mesh, pcd])
    # remove_statistical_outlier函数删除平均距离其邻居较远的点
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=5,
                                                        std_ratio=1.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)
    outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)
    o3d.visualization.draw_geometries([ref_mesh, inlier_cloud])
    o3d.io.write_point_cloud('inlier_cloud.pcd', inlier_cloud)

