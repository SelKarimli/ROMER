import cv2
import numpy as np
import open3d as o3d

# === Step 1: Load images and depth maps ===
img1 = cv2.imread("cam1_rgb.png")
img2 = cv2.imread("cam2_rgb.png")
depth1 = cv2.imread("cam1_depth.png", cv2.IMREAD_UNCHANGED)
depth2 = cv2.imread("cam2_depth.png", cv2.IMREAD_UNCHANGED)

# === Step 2: Camera Intrinsics ===
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5

def pixel_to_3d(u, v, depth, scale=1000.0):
    z = depth[v, u]
    
    # If z is an array (like [z, z, z]), convert to scalar
    if isinstance(z, np.ndarray):
        z = z[0]  # or use z = z.mean() if unsure

    z = float(z) / scale

    if z == 0.0:
        return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

# === Step 3: Feature Detection + Matching ===
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# === Step 4: Convert matched 2D points to 3D using depth ===
pts1 = []
pts2 = []

for m in matches[:100]:  # limit to good matches
    u1, v1 = map(int, kp1[m.queryIdx].pt)
    u2, v2 = map(int, kp2[m.trainIdx].pt)

    p1 = pixel_to_3d(u1, v1, depth1)
    p2 = pixel_to_3d(u2, v2, depth2)

    if p1 is not None and p2 is not None:
        pts1.append(p1)
        pts2.append(p2)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

# === Step 5: Estimate transformation (Umeyama/SVD) ===
def compute_transformation(src, dst):
    assert src.shape == dst.shape
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)

    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

T_12 = compute_transformation(pts1, pts2)
print("Transformation matrix from camera 1 to camera 2:\n", T_12)

