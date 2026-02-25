"""
Track model class for representing a parameterized geometric track shape.
Uses PyTorch tensors for compatibility with loss functions and optimization.
"""

import torch
import numpy as np


class Track:
    """
    Represents a flat track on the ground (z=0) with parameterized geometry.
    
    The track is defined by:
    - length: longitudinal extent along z-axis
    - width: lateral extent along x-axis
    - curvature: radius of curvature (positive = curves to the right)
    
    Origin is at (0, 0) in the XY plane.
    """
    
    def __init__(self, length: float, width: float, curvature: float, 
                num_z_points: int = 100, num_x_points: int = 50,
                device: str = 'cpu', dtype: torch.dtype = torch.float32):
        """
        Initialize the Track model.
        
        Args:
            length: Longitudinal extent (along z-axis) in meters
            width: Lateral extent (along x-axis) in meters
            curvature: Radius of curvature in meters (inf or very large = straight)
            num_z_points: Number of points along the track length
            num_x_points: Number of points across the track width
            device: Device to place tensors on ('cpu' or 'cuda')
            dtype: Data type for tensors (torch.float32 or torch.float64)
        """
        self.length = length
        self.width = width
        self.curvature = curvature if curvature != float('inf') else 1e6
        self.num_z_points = num_z_points
        self.num_x_points = num_x_points
        self.device = device
        self.dtype = dtype
        
        # Generate track geometry
        self._generate_track()
    
    def _generate_track(self):
        """Generate the track surface points as PyTorch tensors."""
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
        self.points = torch.stack([
            x_curved.reshape(-1),
            z_ground.reshape(-1),
            z_grid.reshape(-1)
        ], dim=1)  # Shape: (num_z_points * num_x_points, 3)
        
        # Store grid versions for visualization purposes
        self.x_grid = x_curved  # (num_z_points, num_x_points)
        self.y_grid = z_ground   # (num_z_points, num_x_points)
        self.z_grid = z_grid     # (num_z_points, num_x_points)
    
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
                        curvature: float = None):
        """
        Update track parameters and regenerate geometry.
        Args:
            length: New track length (if None, keeps current value)
            width: New track width (if None, keeps current value)
            curvature: New curvature (if None, keeps current value)
        """
        if length is not None:
            self.length = length
        if width is not None:
            self.width = width
        if curvature is not None:
            self.curvature = curvature if curvature != float('inf') else 1e6
        
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
