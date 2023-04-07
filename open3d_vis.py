# -*- coding: UTF-8 -*-
"""
=================================================
@path   ：3D_Edge_Detection -> open3d_vis
@IDE    ：PyCharm
@Author ：dell
@Date   ：2021/8/18 17:00
==================================================
"""
__author__ = 'dell'

import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("edges.ply")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=True,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


