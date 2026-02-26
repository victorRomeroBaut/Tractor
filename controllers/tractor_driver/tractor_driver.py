from controller import Robot
import numpy as np
import cv2 as cv
import torch
from active_inference import ActiveInferenceNav
import open3d as o3d 
from visualizer import Visualizer
from track import Track


def get_pcd(rgb_cam, depth_cam, height: int, width: int, focal_length, cx, cy):
    """Get point cloud from RGB and depth camera."""
    depth_data = depth_cam.getRangeImage()
    depth_image = np.array(depth_data).reshape((height, width))
    img_buffer = rgb_cam.getImage()
    
    if not img_buffer:
        return None 
    
    # rgb
    rgb_img = np.frombuffer(img_buffer, np.uint8).reshape((height, width, 4))
    rgb_img = rgb_img[:, :, :3][:, :, ::-1] / 255.0
    
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
    # Rotate environment to match Webots coordinate system (Y up, Z forward)
    R = pcd.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
    pcd.rotate(R, center=(0,0,0))
    return pcd


def main():
    """Main tractor driver controller loop."""
    
    # 1. Initialize the Robot
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 2. Define the device names
    # These are the standard names for the Husarion ROSbot 2.0 PROTO fl_wheel_joint
    motor_names = [
        'fl_wheel_joint',
        'fr_wheel_joint',
        'rl_wheel_joint',
        'rr_wheel_joint']

    motors = []
    for name in motor_names:
        motor = robot.getDevice(name)
        if motor:
            # Set to velocity control mode (required for continuous movement)
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            motors.append(motor)
        else:
            print(f"FAILED TO FIND: {name}")
        
    # Activate Lidar
    lidar = robot.getDevice("laser")
    if lidar:
        lidar.enable(timestep)
    # Setup camera
    cam = robot.getDevice("camera rgb")
    if cam:
        cam.enable(timestep)
    depth_cam = robot.getDevice("camera depth")
    depth_cam.enable(timestep)

    width = depth_cam.getWidth()
    height = depth_cam.getHeight()
    fov = depth_cam.getFov()
    # Compute focal length for projection
    focal_length = width / (2 * np.tan(fov / 2))
    print(f"focal: {focal_length}")
    cx, cy = width // 2, height // 2
    # check motors
    if len(motors) == 4:
        print("All motors found")
    else:
        print("Check motor names in the Scene Tree. Movement aborted.")

    SAFE_DIST = 0.25 # 
    BASE_SPEED = 2.0 # 4.0
    KP = 2.5 # 3.5
    STEER_SENSITIVITY = 40.0 # 
    #? Active inference object
    voxel_size = 0.05 # 0.03
    #nav = ActiveInferenceNav(row_width=1, voxel_size=voxel_size)
    track = Track(length=10.0, width=2.0, curvature=0.01, num_x_points=100, num_z_points=200,
                position=(0.1, -0.35, -0.5), rotation=(0, np.pi/2, 0))  # Slightly above the ground and rotated to align with the track
    visualizer = Visualizer()
    
    # Create and add track geometry to visualizer
    track_mesh = Visualizer.create_track_geometry(track)
    visualizer.add_geometry(track_mesh, is_track=True)
    
    # Load and add robot model (.obj file) positioned on the tracker
    robot_model = Visualizer.load_robot_model(obj_file_path="/home/victor/Tractor/protos/shadow.obj",
                                            scale=0.8,  # Adjust scale as needed for your model
                                            position=(0, 0.1, 0.1),  # Position (x, y, z) on the tracker - y is height above track
                                            rotation=(-np.pi/2, np.pi/2, 0))  # Rotation in radians (roll, pitch, yaw)
    if robot_model:
        visualizer.add_geometry(robot_model, is_track=False)
    
    # Robot pose in track frame (position and orientation)
    # These should be updated if the robot moves/rotates in your simulation
    ROBOT_POSITION = np.array([0.0, 0.0, 0.0])  # (x, y, z) in track frame
    ROBOT_ROTATION = np.array([0.0, 0.0, 0.0])  # (roll, pitch, yaw) in radians
    
    # Set initial camera view for better visualization
    #visualizer.reset_view()
    #ctr = visualizer.get_view_control()
    #ctr.set_front([0, -1, -1])
    #ctr.set_lookat([0, 0, 0])
    #ctr.set_up([0, -1, 0])
    #ctr.set_zoom(-0.05)
    #? 4. Main Simulation Loop
    while robot.step(timestep) != -1:
        # Logic can be added here (e.g., stop after 10 seconds)
        ranges = lidar.getRangeImage()
        # check a 30-degree slice in the front center
        front_arc = ranges[int(len(ranges)*0.45) : int(len(ranges) *0.55)]
        min_dist_front = min(front_arc)
        #? point cloud generation
        pcd = get_pcd(rgb_cam=cam,
                    depth_cam=depth_cam,
                    height=height,
                    width=width,
                    focal_length=focal_length,
                    cx=cx, cy=cy)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
        # point cloud processing
        ground_cloud = pcd.select_by_index(inliers, invert=True)
        crop_cloud = pcd.select_by_index(inliers, invert=True)
        ground_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        #? point cloud visualization
        voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(crop_cloud, voxel_size)
        # steering, belief_dist = nav.select_action(voxels, prior)
        speed_l = 0.0 # BASE_SPEED + (steering * 2.0)
        speed_r = 0.0 # BASE_SPEED - (steering * 2.0)
        
        if min_dist_front < SAFE_DIST:
            print(f"LiDAR obstacle at: {min_dist_front:.2f}")
            speed_l = 0.0
            speed_r = 0.0
        
        #o3d.visualization.draw_geometries([crop_cloud, ground_cloud], window_name="Agriculture point cloud")
        # Transform point clouds to world frame (track coordinates) BEFORE creating voxels
        ground_cloud = Visualizer.align_geometries_in_world_frame(
            ground_cloud, 
            robot_position=tuple(ROBOT_POSITION), 
            robot_rotation=tuple(ROBOT_ROTATION)
        )
        crop_cloud = Visualizer.align_geometries_in_world_frame(
            crop_cloud,
            robot_position=tuple(ROBOT_POSITION),
            robot_rotation=tuple(ROBOT_ROTATION)
        )
        
        # Create voxels from the transformed point cloud (now in world frame)
        voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(crop_cloud, voxel_size)
        
        # Update visualization - events are processed at the start of update_visualization
        visualizer.update_visualization(geometry=ground_cloud, voxels=voxels)
        
        # apply control to motors
        # Motor indices: 0=FL, 1=FR, 2=RL, 3=RR
        motors[0].setVelocity(speed_l) # Front Left max(min(speed_l, 10), -10)
        motors[2].setVelocity(speed_l) # Rear Left max(min(speed_l, 10), -10)
        motors[1].setVelocity(speed_r) # Front Right max(min(speed_r, 10), -10)
        motors[3].setVelocity(speed_r) # Rear Right max(min(speed_r, 10), -10)
        #time.sleep(0.1)  # Sleep to simulate control loop timing (adjust as needed)

if __name__ == '__main__':
    main()
    