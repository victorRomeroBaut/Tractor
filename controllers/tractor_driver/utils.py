import numpy as np
import open3d as o3d


#? get point cloud 
def get_pcd(rgb_cam, depth_cam, height:int, width:int, focal_length, cx, cy):
    depth_data = depth_cam.getRangeImage()
    depth_image = np.array(depth_data).reshape((height,width))
    img_buffer = rgb_cam.getImage()
    
    if not img_buffer: return None 
    
    # rgb
    rgb_img = np.frombuffer(img_buffer, np.uint8).reshape((height, width, 4))
    rgb_img = rgb_img[:,:,:3][:,:,::-1]/255.0
    
    # invalid too close of far points
    mask = (depth_image > depth_cam.getMinRange()) & (depth_image < depth_cam.getMaxRange())
    # point cloud
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # apply mask
    z = np.where(mask, depth_image, 0)
    x = (u - cx) * z / focal_length
    y = (v - cy) * z / focal_length
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb_img.reshape(-1, 3)
    
    valid_index = mask.flatten()
    points = points[valid_index]
    colors = colors[valid_index]
    
    # open3D object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
    
def get_voxel(rgb_cam, depth_cam, height:int, width:int, focal_length, cx, cy):
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, width/2, height/2)
    depth_data = depth_cam.getRangeImage() # list of floats
    img_buffer = rgb_cam.getImage()        # BGRA 

    depth_array = np.array(depth_data, dtype=np.float32).reshape((height, width))
    depth_array = np.ascontiguousarray(depth_array)
    depth_array[np.isinf(depth_array)] = 0

    rgb_img = np.frombuffer(img_buffer, np.uint8).reshape((height, width, 4))
    rgb_img = rgb_img[:,:,:3][:,:,::-1].copy()

    o3d_rgb = o3d.geometry.Image(rgb_img)
    o3d_depth = o3d.geometry.Image(depth_array)

    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsics)

    # Voxelize
    voxel_size = 0.05
    voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxels

def create_prior_visualization(row_width, length=2.0, height=0.15):
    points = [ [-row_width/2, height, 0], [-row_width/2, height, length], # Left row line
                [row_width/2, height, 0], [row_width/2, height, length],] # Right row line
    lines = [[0, 1], [2, 3]]
    colors = [[1, 0.5, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_prior_row_map(length=10.0, width=1.0, step=0.1):
    """
    Creates an 'Ideal Row' centered at X=0.
    Returns two lines of points representing the left and right crop rows.
    """
    z_coords = np.arange(0, length, step)
    left_row = np.stack([np.full_like(z_coords, -width/2), np.zeros_like(z_coords), z_coords], axis=1)
    right_row = np.stack([np.full_like(z_coords, width/2), np.zeros_like(z_coords), z_coords], axis=1)
    return np.vstack([left_row, right_row])

def create_3d_prior_map(row_width=1.0, crop_thickness=0.3, crop_height=0.6, length=3.0):
    """ """
    # Create a grid of points within the expected volume of the bushes/crops
    x_range = np.linspace(-crop_thickness/2, crop_thickness/2, 5)
    y_range = np.linspace(0.1, crop_height, 5) # From 10cm to 60cm high
    z_range = np.linspace(0, length, 15)
    
    # Generate the grid
    xv, yv, zv = np.meshgrid(x_range, y_range, z_range)
    
    # Offset to create two rows (Left and Right)
    left_row = np.stack([xv.flatten() - row_width/2, yv.flatten(), zv.flatten()], axis=1)
    right_row = np.stack([xv.flatten() + row_width/2, yv.flatten(), zv.flatten()], axis=1)
    
    prior_points = np.vstack([left_row, right_row])
    
    pcd_prior = o3d.geometry.PointCloud()
    pcd_prior.points = o3d.utility.Vector3dVector(prior_points)
    pcd_prior.paint_uniform_color([1, 0.7, 0]) # Orange "Belief" boxes
    return pcd_prior

def get_aligned_prior(live_pcd, row_width=0.8):
    """
    Finds the ground plane and aligns the Prior boxes to it.
    """
    # 1. Detect the floor using RANSAC
    # distance_threshold: how thick the 'floor' layer is (5cm)
    plane_model, inliers = live_pcd.segment_plane(distance_threshold=0.05,
                                                ransac_n=3,
                                                num_iterations=1000)
    [a, b, c, d] = plane_model # Plane equation: ax + by + cz + d = 0
    terrain_normal = np.array([a, b, c]) 

    # 2. Create the standard 3D Prior (as defined in the previous step)
    prior_pcd = create_3d_prior_map(row_width=row_width)

    # 3. Calculate Rotation to match Terrain Normal
    # We want to rotate the 'Up' axis [0, 1, 0] to match terrain_normal
    up_vector = np.array([0, 1, 0])
    terrain_normal = terrain_normal / np.linalg.norm(terrain_normal)
    
    # Ensure the normal is pointing 'Up' (sometimes RANSAC flips it)
    if terrain_normal[1] < 0:
        terrain_normal = -terrain_normal

    # Cross product and math to find rotation matrix
    v = np.cross(up_vector, terrain_normal)
    ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + ssc + np.matmul(ssc, ssc) * (1 / (1 + np.dot(up_vector, terrain_normal)))

    # 4. Apply transformation to the Prior
    prior_pcd.rotate(rotation_matrix, center=(0, 0, 0))
    
    # 5. Correct the Height (d/b is the vertical offset from origin to plane)
    height_offset = -d / b
    prior_pcd.translate((0, height_offset, 0))

    return prior_pcd, inliers