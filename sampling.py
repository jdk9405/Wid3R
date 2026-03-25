import open3d as o3d
import ipdb

# read pcd
pcd = o3d.io.read_point_cloud("./assets/points.ply")

# flying point removal
pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# # voxel down sampling
pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=0.02)
# more sampling
# pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=0.05)


# save pcd
o3d.io.write_point_cloud("./assets/points_sampled.ply", pcd_filtered)
