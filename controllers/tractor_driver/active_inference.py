"""
Core concepts:
Components: RGBD camera 
1. Initially, Agent must create a representation (belief) of its environment 
    Using a World Model (millions of slam 3D cubes of space (voxels) with side length of 15cm).
2. Ate the beginning of a task, Agent knows nothing about what each voxel contains.
3. Agent must explore independently, without any pre-specified route or destination
"""

import numpy as np
import open3d as o3d

class ActiveInference:
    def __init__(self, target_state, internal_belief, learning_rate, action_precision):
        self.target_state = target_state
        self.internal_belief = internal_belief
        self.memory = {"left":1.0, "right":1.0} # initial belief
        self.learning_rate = learning_rate
        self.action_precision = action_precision
        self.trust_factor = 0.1 # How much robot trusts new sensor data vs old memory
        self.decay_rate = 0.95  # How much memory persist (%)
    
    def update_memory(self, l_dist:float, r_dist:float):
        # If sensor sees nothing (inf), rely 100% on memory
        # If sensor sees something, we blend it with memory
        if l_dist < 10.0: # valid reading
            self.memory["left"] = (self.memory["left"] * (1 - self.trust_factor)) + (l_dist * self.trust_factor)
        else: # sensor blocked
            self.memory["left"] *= self.decay_rate # slowly fade the memory
        if r_dist < 10.0:
            self.memory["right"] = (self.memory["right"] * (1 - self.trust_factor)) + (r_dist * self.trust_factor)
        else:
            self.memory["right"] *= self.decay_rate
    

    def compute(self, l_dist:float, r_dist:float, var):
        #? Mind representation
        self.update_memory(l_dist, r_dist)
        #? Perception -> update internal belief (minimize prediction error)
        observation = self.memory["left"] - self.memory["right"] #l_dist - r_dist
        prediction_error = observation - self.internal_belief
        sensor_precision = 1.0 / (var + 0.01) # lower precision if ranges are messy (desalineado)
        print(f"error: {prediction_error:.2f} sensor: {sensor_precision:.2f} var: {var:.3f}")
        self.internal_belief += (self.learning_rate * sensor_precision) * prediction_error
        #? Action -> Minimize Expected Surprise (Minimize Free Energy)
        # Robot moves to change the environment so that internal belief -> target
        # In inference, action is the gradient descent on the expected sensory error
        drive_error = self.internal_belief - self.target_state
        steering = -self.action_precision * drive_error # Negative because counteract the error
        return steering

class VoxelActiveInference:
    def __init__(self, voxel_size, num_actions:int=3):
        self.voxel_size = voxel_size
        # Actions: 0: Left, 1: Straight, 2: Right
        self.actions = [-0.5, 0.0, 0.5] # Velocities

    def get_observation(self, voxel_grid):
        voxels = voxel_grid.get_voxels()
        if not voxels: return 0
        # local coordinates (X: left/right, Z: forward)
        indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
        local_coords = indices * self.voxel_size
        local_x = indices[:, 0] * self.voxel_size
        local_z = indices[:, 2] * self.voxel_size
        # filter for the crop row height and a look-ahead distance
        mask = (local_z > 0.2) & (local_z < 1.5)
        height_mask = (local_coords[:, 1] > 0.35) & (local_coords[:, 1] < 0.5)
        disk_mask = (local_coords[:, 2] > 0.3) & (local_coords[:, 2] < 2.0)
        final_mask = height_mask & disk_mask
        #print(f"lz>: {local_z < 0.2}, lz<: {local_z <1.2}")
        row_voxels_x = local_x[mask]
        valid_voxels_x = local_coords[final_mask, 0]
        # compute imbalance
        left_count = np.sum(valid_voxels_x < -0.05) # row_voxels_x
        right_count = np.sum(valid_voxels_x > 0.05) # row_voxels_x
        print(f"remaining voxels: {len(valid_voxels_x)}")
        if (left_count + right_count) == 0:
            return 0
        # Normalized observation (-1.0 to 1.0)
        norm_oservation = (right_count - left_count) / (left_count + right_count + 1e-6)
        return norm_oservation
    
    def compute_efe(self, observation, action_idx):
        #? Expected Free Energy (Simplified)
        # Predict next observation based on action
        # if robot turn left (-0.5), the right side count will increase
        predicted_change = self.actions[action_idx] * 0.2
        predicted_obs = observation + predicted_change

        # 1. Divergence from preference (cost)
        # Wanted predicted_obs be 0 (centered)
        cost = (predicted_obs - 0)**2
        # 2. Epistemic value
        uncertainty = 1.0 / (abs(observation) + 0.1) # High voxel counts = lower uncertainty
        return cost + 0.1 * uncertainty
    
    def select_action(self, voxel_grid):
        obs = self.get_observation(voxel_grid)
        # Evaluate EFE for each possible action
        efes = [self.compute_efe(obs, i) for i in range(len(self.actions))]
        # Use softmax to pick the action with minimum free energy
        probs = np.exp(-np.array(efes))
        probs /= np.sum(probs)
        action = self.actions[np.argmax(probs)]
        print(f"probs: {probs}, action: {action}")
        return action

class ActiveInferenceNav:
    def __init__(self, row_width=1.0, voxel_size=0.05):
        self.voxels_size = voxel_size
        self.row_width = row_width

        #? 1. Prior: Mental model of where crops should be (left and right)
        self.prior_left_x = -row_width / 2
        self.prior_right_x = row_width / 2
        # Possible steering actions (turning states)
        self.actions = np.array([-0.6, -0.3, 0.0, 0.3, 0.6])

    def get_live_voxels_coords(self, voxel_grid):
        """ """
        voxels = voxel_grid.get_voxels()
        if not voxels:
            return np.array([])
        # Convert indices to meters
        indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
        coords = indices * self.voxels_size
        #? Height filter: ignore floor (y < 0.1) and high noise (y > 1.0)
        # assuming Y is up, adjust index to [2] if Z is up
        mask = (coords[:,1] > 0.15) & (coords[:, 1] < 0.8) & (coords[:, 2] < 2.0)
        return coords[mask]
    
    def calculate_efe(self, live_coords, action):
        """ how an action will shift the voxels and estimates the surprise (distance to prior)"""
        #? 2. Prediction (generative model)
        shift = action * 0.25 # Expected lateral shift
        projected_x = live_coords[:, 0] + shift
        #? 3. Likelihood (matching prior to perception)
        dist_to_left = (projected_x - self.prior_left_x)**2
        dist_to_right = (projected_x - self.prior_right_x)**2
        # Robot prefers voxels to be exactly on the prior lines
        min_distances = np.minimum(dist_to_left, dist_to_right)
        # Expected Free Energy = Mean Squared Error of the alignment
        efe = np.mean(min_distances)
        return efe
    
    def calculate_efe_3d(self, live_coords, action, prior_pcd):
        # 1. Predict movement (Shift the live voxels based on action)
        predicted_shift = action * 0.2
        projected_coords = live_coords.copy()
        projected_coords[:, 0] += predicted_shift
        # 2. Use a KDTree for fast distance calculation
        # (Open3D's built-in way to find how far points are from the prior)
        prior_tree = o3d.geometry.KDTreeFlann(prior_pcd)
        total_error = 0
        for pt in projected_coords:
            # Find distance to the nearest point in our 3D prior
            [_, _, dist_sq] = prior_tree.search_knn_vector_3d(pt, 1)
            total_error += dist_sq[0]
        
        # EFE is the average squared distance of all live voxels to the 3D Prior
        return total_error / len(live_coords)
    
    def predict_movement(self, coords, action):
        """
        Simulates the transformation of the voxel cloud based on steering
        """
        theta = action * 0.3
        c, s = np.cos(theta), np.sin(theta)
        # Rotation around Y-axis Up
        R = np.array([[c, 0, s], 
                    [0, 1, 0], 
                    [-s, 0, c]])
        return (coords @ R.T) + np.array([0, 0, 0.1])
    
    def select_action(self, voxel_grid, dynamic_prior):
        #live_coords = self.get_live_voxels_coords(voxel_grid)
        voxels = voxel_grid.get_voxels()
        #if len(live_coords) == 0:
        #    return 0.0, [0.2] * len(self.actions)
        if not voxels:
            return 0.0, [0.2] * len(self.actions)
        # Compute EFE  for every potential action
        indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
        live_coords = indices * self.voxels_size
        prior_tree = o3d.geometry.KDTreeFlann(dynamic_prior)
        efes = []
        for action in self.actions:
            # PREDICTION: Project where voxels will be if we take this action
            # We simulate the lateral shift and a small forward step
            projected_coords = self.predict_movement(live_coords, action)
            # CALCULATE SURPRISE (Likelihood)
            # We sum the distances of all voxels to the nearest point in the Prior Volume
            total_dist = 0
            for pt in projected_coords:
                # search_knn_vector_3d returns: [count, indices, dist_squared]
                _, _, dist_sq = prior_tree.search_knn_vector_3d(pt, 1)
                total_dist += dist_sq[0]
            # EFE = avg distance (lower is better)
            efe = total_dist / len(projected_coords)
            efes.append(efe)
        #efes = np.array([self.calculate_efe_3d(live_coords, a) for a in self.actions])
        # 3. BELIEF GENERATION
        # Higher precision (10.0) makes the robot more decisive
        precision = 10.0
        exp_efe = np.exp(-precision * np.array(efes))
        beliefs = exp_efe / (np.sum(exp_efe) + 0.0001)
        best_action = self.actions[np.argmax(beliefs)]
        print(f"best: {best_action}")
        return best_action, beliefs
