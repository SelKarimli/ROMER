import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# Create folders for saving
os.makedirs("rgb", exist_ok=True)
os.makedirs("depth", exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting RealSense pipeline...")
pipeline.start(config)

print("Press 's' to save RGB and Depth frames. Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Visualize depth (for debugging)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RGB + Depth', images)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            rgb_path = f"rgb/frame_{timestamp}.png"
            depth_path = f"depth/frame_{timestamp}.png"

            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_path, depth_image)

            print(f"Saved RGB: {rgb_path}")
            print(f"Saved Depth: {depth_path}")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

