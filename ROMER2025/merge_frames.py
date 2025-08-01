import open3d as o3d
import numpy as np
import cv2

# Frame siyahısı: RGB və depth şəkil cütləri
frames = [
    ("rgb/frame_1.png", "depth/frame_1.png"),
    ("rgb/frame_2.png", "depth/frame_2.png"),
    ("rgb/frame_3.png", "depth/frame_3.png"),
    ("rgb/frame_4.png", "depth/frame_4.png"),
    ("rgb/frame_5.png", "depth/frame_5.png"),
    # əlavə şəkilləri buraya daxil et
]

# Bütün nöqtələri birləşdirmək üçün boş PointCloud obyekt
all_points = o3d.geometry.PointCloud()

for rgb_path, depth_path in frames:
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if rgb is None or depth is None:
        print(f"❌ Couldn't load {rgb_path} or {depth_path}")
        continue

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=3.0
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    )

    all_points += pcd

# Yekun point cloud .ply faylına yazılır
o3d.io.write_point_cloud("merged_model.ply", all_points)
print("✅ Combined model saved as merged_model.ply")

