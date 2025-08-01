import open3d as o3d
import numpy as np
import cv2
import sys

if len(sys.argv) != 3:
    print("Usage: python3 3d_model.py <rgb_image_path> <depth_image_path>")
    sys.exit(1)

rgb_path = sys.argv[1]
depth_path = sys.argv[2]

rgb = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if rgb is None or depth is None:
    print("Failed to load image(s). Check the file paths.")
    sys.exit(1)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(rgb),
    o3d.geometry.Image(depth),
    convert_rgb_to_intensity=False,
    depth_scale=1000.0,  # Adjust based on RealSense settings
    depth_trunc=3.0      # Max depth in meters
)

intrinsics = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, intrinsics)

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("output.ply", pcd)
print("Point cloud saved as output.ply")

