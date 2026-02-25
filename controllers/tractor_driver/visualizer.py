import numpy as np
import open3d as o3d


class Visualizer:
    """Handles visualization of point clouds, voxels, and priors using Open3D."""
    
    def __init__(self):
        """Initialize the Open3D visualizer."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Tractor Driver Visualization", width=1200, height=800)
        # Enable proper rendering
        self.vis.get_render_option().light_on = True
        self.vis.get_render_option().mesh_show_wireframe = False
        
        # Store references to dynamic geometries
        self._track_mesh = None
        self._current_geometry = None
        self._current_voxels = None
    
    @staticmethod
    def get_voxel(rgb_cam, depth_cam, height: int, width: int, focal_length, cx, cy):
        """Get voxel grid from RGB and depth camera."""
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, width/2, height/2)
        depth_data = depth_cam.getRangeImage()  # list of floats
        img_buffer = rgb_cam.getImage()  # BGRA 

        depth_array = np.array(depth_data, dtype=np.float32).reshape((height, width))
        depth_array = np.ascontiguousarray(depth_array)
        depth_array[np.isinf(depth_array)] = 0

        rgb_img = np.frombuffer(img_buffer, np.uint8).reshape((height, width, 4))
        rgb_img = rgb_img[:, :, :3][:, :, ::-1].copy()

        o3d_rgb = o3d.geometry.Image(rgb_img)
        o3d_depth = o3d.geometry.Image(depth_array)

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsics)

        # Voxelize
        voxel_size = 0.05
        voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
        return voxels

    @staticmethod
    def create_prior_visualization(row_width, length=2.0, height=0.15):
        """Create line visualization for row prior."""
        points = [[-row_width/2, height, 0], [-row_width/2, height, length],  # Left row line
                    [row_width/2, height, 0], [row_width/2, height, length]]  # Right row line
        lines = [[0, 1], [2, 3]]
        colors = [[1, 0.5, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    @staticmethod
    def create_prior_row_map(length=10.0, width=1.0, step=0.1):
        """
        Creates an 'Ideal Row' centered at X=0.
        Returns two lines of points representing the left and right crop rows.
        """
        z_coords = np.arange(0, length, step)
        left_row = np.stack([np.full_like(z_coords, -width/2), np.zeros_like(z_coords), z_coords], axis=1)
        right_row = np.stack([np.full_like(z_coords, width/2), np.zeros_like(z_coords), z_coords], axis=1)
        return np.vstack([left_row, right_row])

    @staticmethod
    def create_3d_prior_map(row_width=1.0, crop_thickness=0.3, crop_height=0.6, length=3.0):
        """Create 3D point cloud representing expected crop volume."""
        # Create a grid of points within the expected volume of the bushes/crops
        x_range = np.linspace(-crop_thickness/2, crop_thickness/2, 5)
        y_range = np.linspace(0.1, crop_height, 5)  # From 10cm to 60cm high
        z_range = np.linspace(0, length, 15)
        
        # Generate the grid
        xv, yv, zv = np.meshgrid(x_range, y_range, z_range)
        
        # Offset to create two rows (Left and Right)
        left_row = np.stack([xv.flatten() - row_width/2, yv.flatten(), zv.flatten()], axis=1)
        right_row = np.stack([xv.flatten() + row_width/2, yv.flatten(), zv.flatten()], axis=1)
        
        prior_points = np.vstack([left_row, right_row])
        
        pcd_prior = o3d.geometry.PointCloud()
        pcd_prior.points = o3d.utility.Vector3dVector(prior_points)
        pcd_prior.paint_uniform_color([1, 0.7, 0])  # Orange "Belief" boxes
        return pcd_prior

    @staticmethod
    def get_aligned_prior(live_pcd, row_width=0.8):
        """
        Finds the ground plane and aligns the Prior boxes to it.
        """
        # 1. Detect the floor using RANSAC
        # distance_threshold: how thick the 'floor' layer is (5cm)
        plane_model, inliers = live_pcd.segment_plane(distance_threshold=0.05,
                                                    ransac_n=3,
                                                    num_iterations=1000)
        [a, b, c, d] = plane_model  # Plane equation: ax + by + cz + d = 0
        terrain_normal = np.array([a, b, c])

        # 2. Create the standard 3D Prior (as defined in the previous step)
        prior_pcd = Visualizer.create_3d_prior_map(row_width=row_width)

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

    def update_visualization(self, geometry, voxels):
        """
        Update the visualizer with new dynamic geometries while preserving track and camera state.
        
        Processes events FIRST to allow mouse interaction, then updates only dynamic geometries.
        
        Args:
            geometry: Point cloud or other geometry to visualize
            voxels: Voxel grid to visualize
        """
        # CRITICAL: Poll events FIRST to process mouse and keyboard input
        # This must happen before any geometry updates to preserve camera state
        self.vis.poll_events()
        
        # Remove only the previous dynamic geometries (not the track mesh)
        if self._current_geometry is not None:
            try:
                self.vis.remove_geometry(self._current_geometry, reset_bounding_box=False)
            except:
                pass  # Geometry might not exist in visualizer yet
        
        if self._current_voxels is not None:
            try:
                self.vis.remove_geometry(self._current_voxels, reset_bounding_box=False)
            except:
                pass
        
        # Add new dynamic geometries
        #self.vis.add_geometry(geometry, reset_bounding_box=False)
        #self.vis.add_geometry(voxels, reset_bounding_box=False)
        
        # Store references for next frame's removal
        self._current_geometry = geometry
        self._current_voxels = voxels
        
        # Update renderer
        self.vis.update_renderer()

    def add_geometry(self, geometry, is_track=False):
        """
        Add a geometry to the visualizer.
        
        Args:
            geometry: The geometry to add
            is_track: If True, store as track mesh for persistence
        """
        if is_track:
            self._track_mesh = geometry
        self.vis.add_geometry(geometry)

    def render(self):
        """Render the current visualization."""
        # Process events first to allow mouse interaction
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def process_events(self):
        """Explicitly process pending events (mouse, keyboard, window events)."""
        try:
            self.vis.poll_events()
        except:
            pass  # In case visualizer window is closed
    
    def get_view_control(self):
        """Get the view control object to manipulate camera."""
        return self.vis.get_view_control()
    
    def reset_view(self):
        """Reset the camera to default viewpoint."""
        self.vis.reset_view_point(True)

    def clear(self):
        """Clear all geometries from the visualizer."""
        self.vis.clear_geometries()

    def close(self):
        """Close the visualizer window."""
        self.vis.destroy_window()

    @staticmethod
    def create_track_geometry(track):
        """
        Create a white solid mesh geometry from a Track object.
        
        Args:
            track: Track object with x_grid, y_grid, z_grid attributes
            
        Returns:
            o3d.geometry.TriangleMesh: White mesh representing the track
        """
        # Get track points from grids
        num_z, num_x = track.x_grid.shape
        
        # Create vertices: flatten the grid points
        vertices = np.stack([
            track.x_grid.detach().cpu().numpy().flatten(),
            track.y_grid.detach().cpu().numpy().flatten(),
            track.z_grid.detach().cpu().numpy().flatten()
        ], axis=1)
        
        # Create triangles between adjacent grid points
        triangles = []
        for i in range(num_z - 1):
            for j in range(num_x - 1):
                # Current vertices indices
                v0 = i * num_x + j
                v1 = i * num_x + (j + 1)
                v2 = (i + 1) * num_x + j
                v3 = (i + 1) * num_x + (j + 1)
                
                # Two triangles per grid cell
                triangles.append([v0, v1, v2])
                triangles.append([v1, v3, v2])
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Paint white and compute normals for better visualization
        mesh.paint_uniform_color([1.0, 1.0, 1.0])  # White
        mesh.compute_vertex_normals()
        ## adds
        
        return mesh