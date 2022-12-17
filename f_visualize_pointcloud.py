def visualize_pointcloud(coords, show_normals=False, colors=None):

    '''Takes point coordinates, an array of RBG codes (n_points, 3)
    and True or False fro show_normals. Visualizes the pointcloud 
    with open3D in the given colors'''

    import open3d as o3d

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(coords)
    
    if colors is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

    if show_normals:
        pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 5))
        pointcloud.orient_normals_consistent_tangent_plane(k=5)
        
        o3d.visualization.draw_geometries([pointcloud], point_show_normal = True)

    else: 
        o3d.visualization.draw_geometries([pointcloud])