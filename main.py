import os

import numpy as np
import open3d as o3d
import pandas as pd
from pyntcloud import PyntCloud
from vedo import *


def extract_vertices(input_str):
    """
    Extract vertices to a unique point cloud file.
    :param input_str: Input file name/path.
    :return:
    """
    mesh = o3d.io.read_triangle_mesh(input_str)
    if os.path.splitext(input_str)[-1] == ".stl":
        mesh.remove_duplicated_vertices()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    return pcd


def ransac_seg(input_pcd, distance_threshold, ransac_n, num_iterations, mode):
    """
    Segments a plane in the point cloud using the RANSAC algorithm. Then choose points are less or larger than plane.
    :param mode: 0 represents less, 1 represents larger
    :param input_pcd: Input point cloud.
    :param distance_threshold: Max distance a point can be from the plane model, and still be considered an inlier.
    :param ransac_n: Number of initial points to be considered inliers in each iteration.
    :param num_iterations: Number of iterations.
    :return: 
    """
    plane_model, inliers = input_pcd.segment_plane(distance_threshold,
                                                   ransac_n,
                                                   num_iterations)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = input_pcd.select_by_index(inliers)
    outlier_cloud = input_pcd.select_by_index(inliers, invert=True)

    dists = input_pcd.compute_point_cloud_distance(inlier_cloud)
    dists = np.asarray(dists)
    index = np.where(dists > 0.01)[0]
    pcd_without_inlier_cloud = input_pcd.select_by_index(index)

    axis_aligned_bounding_box = inlier_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)

    tmp_pts = np.asarray(pcd_without_inlier_cloud.points)
    tmp_cls = np.asarray(pcd_without_inlier_cloud.colors)

    final_pts = []
    final_cls = []
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
    # Load and merge Points
    vertices_pts = extract_vertices('data/Example.stl')
    o3d.io.write_point_cloud('data/vertices.pcd', vertices_pts)

    # Read vertices pcd file
    input_pts = PyntCloud.from_file("data/vertices.pcd")

    # Load and calc curvature Meshes
    mesh = load("data/Example.stl")
    
    # Load reference mesh
    ref_mesh = o3d.io.read_triangle_mesh("data/Example.stl")
    
    # define hyper parameters
    k_n = 20
    threshold = 0.003
    
    # ——————————————————————Edge Detection—————————————————————— #
    pcd_np = np.zeros((len(input_pts.points), 6))

    # find neighbors
    kdtree_id = input_pts.add_structure("kdtree")
    k_neighbors = input_pts.get_neighbors(k=k_n, kdtree=kdtree_id)

    # calculate eigenvalues
    ev = input_pts.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = input_pts.points['x'].values
    y = input_pts.points['y'].values
    z = input_pts.points['z'].values

    e1 = input_pts.points['e3(' + str(k_n + 1) + ')'].values
    e2 = input_pts.points['e2(' + str(k_n + 1) + ')'].values
    e3 = input_pts.points['e1(' + str(k_n + 1) + ')'].values

    sum_eg = np.add(np.add(e1, e2), e3)
    sigma = np.divide(e1, sum_eg)
    sigma_value = sigma

    # Save the edges and point cloud
    thresh_min = sigma_value < threshold
    sigma_value[thresh_min] = 0
    thresh_max = sigma_value > threshold
    sigma_value[thresh_max] = 255

    pcd_np[:, 0] = x
    pcd_np[:, 1] = y
    pcd_np[:, 2] = z
    pcd_np[:, 3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:, 3] == 0), axis=0)


    clmns = ['x', 'y', 'z', 'red', 'green', 'blue']
    pcd_pd = pd.DataFrame(data=pcd_np, columns=clmns)
    pcd_pd['red'] = sigma_value.astype(np.uint8)

    edge_points = PyntCloud(pd.DataFrame(data=edge_np, columns=clmns))

    PyntCloud.to_file(edge_points, "data/edges.ply")

    input_pts = o3d.io.read_point_cloud("data/edges.ply")

    # Modify codes down below
    # ——————————————————————RANSAC#1—————————————————————— #
    ransac_result, _, _, _ = ransac_seg(input_pts, 0.8, 10, 1000, 0)

    # ——————————————————————RANSAC#2—————————————————————— #
    ransac_result, _, _, _ = ransac_seg(ransac_result, 0.5, 10, 1000, 1)

    # ——————————————————————RANSAC#3—————————————————————— #
    ransac_result, _, tmp_inlier, _ = ransac_seg(ransac_result, 4, 10, 1000, 1)

    o3d.io.write_point_cloud("data/result.pcd", tmp_inlier)

    # Get curvatures array
    mesh.addCurvatureScalars(1)
    curvatures = np.asarray(mesh.getPointArray())

    # Load and merge Points
    feature_pts = o3d.io.read_point_cloud('data/result.pcd')
    vertices_pts.paint_uniform_color([0.5, 0.5, 0.5])

    # Merge two point sets down below
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.append(np.asarray(vertices_pts.points), np.asarray(feature_pts.points), axis=0))
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.append(np.asarray(vertices_pts.colors), np.asarray(feature_pts.colors), axis=0))
    print(merged_pcd)

    # Create KDTree from merged variable
    pcd_tree = o3d.geometry.KDTreeFlann(merged_pcd)

    # Initial result list
    result = []

    # Enumerate all feature_pts with detecting 1-NN curvatures
    for idx, val in enumerate(merged_pcd.points):
        while idx >= np.shape(vertices_pts.points)[0]:
            [_, index, _] = pcd_tree.search_hybrid_vector_3d(merged_pcd.points[idx], 0.3, 2)
            if idx == index[1] and index[0] < np.shape(vertices_pts.points)[0] and curvatures[index[0]] < -1:
                result.append(merged_pcd.points[index[0]])
            else:
                break
            break

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(result))

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=1)
    ref_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([ref_mesh, pcd])
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=5,
                                                        std_ratio=1.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)
    outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)
    o3d.visualization.draw_geometries([ref_mesh, inlier_cloud])
    o3d.io.write_point_cloud('data/inlier_cloud.pcd', inlier_cloud)
