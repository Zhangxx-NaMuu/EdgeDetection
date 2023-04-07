# -*- coding: UTF-8 -*-
'''
=================================================
@path   ：3D_Edge_Detection -> vedo_test
@IDE    ：PyCharm
@Author ：dell
@Date   ：2021/8/18 17:53
==================================================
'''
__author__ = 'dell'
"""
Example to show how to use recoSurface()
to reconstruct a surface from points.
 1. An object is loaded and
    noise is added to its vertices.
 2. the point cloud is smoothened with MLS
    (see moving_least_squares.py)
 3. mesh.clean() imposes a minimum distance
    among mesh points where 'tol' is the
    fraction of the mesh size.
 4. a triangular mesh is extracted from
    this set of sparse Points, 'bins' is the
    number of voxels of the subdivision
"""
print(__doc__)

from vedo import *
import numpy as np
import os
import open3d as o3d


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
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = mesh.vertices
    return pcd1



plt = Plotter(N=4, axes=0)

mesh = plt.load(dataurl + "apple.ply").subdivide()
plt.show(mesh, at=0)

noise = np.random.randn(mesh.N(), 3) * 0.03

pts0 = Points(mesh.points() + noise, r=3).legend("noisy cloud")
plt.show(pts0, at=1)

pts1 = pts0.clone().smoothMLS2D(f=0.8)  # smooth cloud

print("Nr of points before cleaning nr. points:", pts1.N())

# impose a min distance among mesh points
pts1.clean(tol=0.005).legend("smooth cloud")
print("after  cleaning nr. points:", pts1.N())

plt.show(pts1, at=2)

# reconstructed surface from point cloud
reco = recoSurface(pts1, dims=100, radius=0.2).legend("surf. reco")
plt.show(reco, at=3, axes=7, zoom=1.2, interactive=1).close()

"""
from vedo import *

line1 = Line(Circle().points()).lw(5).c('black')

line2 = Line((-2, -1.5), (2, 2)).lw(5).c('green')

m1 = line1.extrude(1).shift(0, 0, -0.5)
p = Points(m1.intersectWithLine(*line2.points()), r=20).c('red')

show(line1, line2, p, axes=1)
"""