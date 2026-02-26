"""
Track model class for representing a parameterized geometric track shape.
Uses PyTorch tensors for compatibility with loss functions and optimization.
"""

import torch
import numpy as np


class Track:
    """
    Represents a flat track on the ground with parameterized geometry.
    
    The track is defined by:
    - length: longitudinal extent along z-axis
    - width: lateral extent along x-axis
    - curvature: radius of curvature (positive = curves to the right)
    - position: (x, y, z) offset for track location in 3D space
    - rotation: (roll, pitch, yaw) rotation angles in radians
    
    By default, track origin is at (0, 0, 0) with no rotation.
    """
    
    def __init__(self, length: float, width: float, curvature: float, 
                num_z_points: int = 100, num_x_points: int = 50,
                position: tuple = (0, 0, 0), rotation: tuple = (0, 0, 0),
                device: str = 'cpu', dtype: torch.dtype = torch.float32):
        """
        Initialize the Track model.
        
        Args:
            length: Longitudinal extent (along z-axis) in meters
            width: Lateral extent (along x-axis) in meters
            curvature: Radius of curvature in meters (inf or very large = straight)
            num_z_points: Number of points along the track length
            num_x_points: Number of points across the track width
            position: Tuple (x, y, z) offset for track position in meters
            rotation: Tuple (roll, pitch, yaw) rotation in radians
            device: Device to place tensors on ('cpu' or 'cuda')
            dtype: Data type for tensors (torch.float32 or torch.float64)
        """
        self.length = length
        self.width = width
        self.curvature = curvature if curvature != float('inf') else 1e6
        self.num_z_points = num_z_points
        self.num_x_points = num_x_points
        self.position = np.array(position)  # Store as numpy array
        self.rotation = np.array(rotation)  # Store as numpy array (roll, pitch, yaw)
        self.device = device
        self.dtype = dtype
        
        # Generate track geometry
        self._generate_track()
    
    def _generate_track(self):
        """ Generate the track surface points as PyTorch tensors."""
        # Create parametric z coordinates (along the track length)
        z_coords = torch.linspace(0, self.length, self.num_z_points, 
                                device=self.device, dtype=self.dtype)
        
        # Create parametric x coordinates (across the track width)
        # Centered at x=0, ranging from -width/2 to +width/2
        x_coords = torch.linspace(-self.width / 2, self.width / 2, self.num_x_points,
                                device=self.device, dtype=self.dtype)
        
        # Apply curvature: x' = x + offset based on z position
        # For a circular arc with radius R, the x offset is: R - sqrt(R^2 - z^2)
        # For small angles, we can approximate: x_offset(z) ≈ z^2 / (2*R)
        if abs(self.curvature) > 1e5:  # Nearly straight
            x_offset = torch.zeros_like(z_coords)
        else:
            x_offset = (z_coords ** 2) / (2 * self.curvature)
        
        # Create meshgrid for all track surface points
        # Shape: (num_x_points, num_z_points)
        z_grid, x_grid = torch.meshgrid(z_coords, x_coords, indexing='ij')
        
        # Add curvature offset to x coordinates
        x_curved = x_grid + x_offset.unsqueeze(1)
        
        # Z coordinate is always 0 (ground plane)
        z_ground = torch.zeros_like(x_curved)
        
        # Stack to create 3D point cloud: (num_points, 3)
        # Reshape to (num_z_points * num_x_points, 3)
        points = torch.stack([
            x_curved.reshape(-1),
            z_ground.reshape(-1),
            z_grid.reshape(-1)
            ], dim=1)  # Shape: (num_z_points * num_x_points, 3)
        
        # Apply rotation to points
        if np.any(self.rotation != 0):
            # Convert rotation to rotation matrix (Roll-Pitch-Yaw)
            roll, pitch, yaw = self.rotation
            
            # Rotation around X axis (roll)
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            
            # Rotation around Y axis (pitch)
            R_y = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            # Rotation around Z axis (yaw)
            R_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            # Combined rotation matrix (Roll-Pitch-Yaw order)
            R = R_z @ R_y @ R_x
            
            # Apply rotation to points
            points_np = points.detach().cpu().numpy()
            points_rotated = points_np @ R.T
            points = torch.from_numpy(points_rotated).to(device=self.device, dtype=self.dtype)
        
        # Apply position offset
        if np.any(self.position != 0):
            position_tensor = torch.from_numpy(self.position).to(device=self.device, dtype=self.dtype)
            points = points + position_tensor
        
        self.points = points
        
        # Store grid versions for visualization purposes
        self.x_grid = x_curved  # (num_z_points, num_x_points)
        self.y_grid = z_ground   # (num_z_points, num_x_points)
        self.z_grid = z_grid     # (num_z_points, num_x_points)
        
        # Apply transformations to grids as well for visualization
        if np.any(self.rotation != 0) or np.any(self.position != 0):
            # Flatten grids and apply transformations
            x_flat = self.x_grid.reshape(-1)
            y_flat = self.y_grid.reshape(-1)
            z_flat = self.z_grid.reshape(-1)
            
            grid_points = torch.stack([x_flat, y_flat, z_flat], dim=1)
            
            # Apply rotation
            if np.any(self.rotation != 0):
                points_np = grid_points.detach().cpu().numpy()
                roll, pitch, yaw = self.rotation
                R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
                R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
                R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                R = R_z @ R_y @ R_x
                points_rotated = points_np @ R.T
                grid_points = torch.from_numpy(points_rotated).to(device=self.device, dtype=self.dtype)
            
            # Apply position
            if np.any(self.position != 0):
                position_tensor = torch.from_numpy(self.position).to(device=self.device, dtype=self.dtype)
                grid_points = grid_points + position_tensor
            
            # Reshape back to grids
            self.x_grid = grid_points[:, 0].reshape(self.num_z_points, self.num_x_points)
            self.y_grid = grid_points[:, 1].reshape(self.num_z_points, self.num_x_points)
            self.z_grid = grid_points[:, 2].reshape(self.num_z_points, self.num_x_points)
    
    def get_track_edges(self):
        """
        Get the left and right edges of the track.
        Returns:
            left_edge: Tensor of shape (num_z_points, 3) - left edge points
            right_edge: Tensor of shape (num_z_points, 3) - right edge points
        """
        left_edge = torch.stack([
            self.x_grid[:, 0],
            self.y_grid[:, 0],
            self.z_grid[:, 0]
        ], dim=1)
        
        right_edge = torch.stack([
            self.x_grid[:, -1],
            self.y_grid[:, -1],
            self.z_grid[:, -1]
        ], dim=1)
        
        return left_edge, right_edge
    
    def get_center_line(self):
        """
        Get the center line of the track.
        
        Returns:
            Tensor of shape (num_z_points, 3) - center line points
        """
        center_idx = self.num_x_points // 2
        center_line = torch.stack([
            self.x_grid[:, center_idx],
            self.y_grid[:, center_idx],
            self.z_grid[:, center_idx]
        ], dim=1)
        
        return center_line
    
    def update_parameters(self, length: float = None, width: float = None, 
                        curvature: float = None, position: tuple = None, rotation: tuple = None):
        """
        Update track parameters and regenerate geometry.
        Args:
            length: New track length (if None, keeps current value)
            width: New track width (if None, keeps current value)
            curvature: New curvature (if None, keeps current value)
            position: New position tuple (x, y, z) (if None, keeps current value)
            rotation: New rotation tuple (roll, pitch, yaw) (if None, keeps current value)
        """
        if length is not None:
            self.length = length
        if width is not None:
            self.width = width
        if curvature is not None:
            self.curvature = curvature if curvature != float('inf') else 1e6
        if position is not None:
            self.position = np.array(position)
        if rotation is not None:
            self.rotation = np.array(rotation)
        
        self._generate_track()
    
    def get_points_numpy(self):
        """
        Get track points as NumPy array (for compatibility with Open3D).
        
        Returns:
            NumPy array of shape (num_points, 3)
        """
        return self.points.detach().cpu().numpy()
    
    def __repr__(self):
        return (f"Track(length={self.length}, width={self.width}, "
                f"curvature={self.curvature}, points={self.points.shape[0]})")
