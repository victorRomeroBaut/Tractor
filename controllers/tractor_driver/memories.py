"""
PyBullet Robot Motion Simulator Example

This script demonstrates how to use PyBullet to simulate robot motion.
It includes examples of:
- Loading a robot model
- Setting up the physics engine
- Applying forces and joint control
- Running a simulation loop
- Collecting motion data
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


class RobotSimulator:
    """A simple robot motion simulator using PyBullet."""
    
    def __init__(self, use_gui=True, gravity=(0, 0, -9.81)):
        """
        Initialize the PyBullet simulator.
        
        Args:
            use_gui (bool): If True, uses GUI mode; otherwise uses DIRECT mode
            gravity (tuple): Gravity vector (x, y, z)
        """
        # Connect to PyBullet engine
        if use_gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # Set up the environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Store simulation parameters
        self.step_count = 0
        self.time_step = 1.0 / 240.0  # 240 Hz simulation
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSubSteps=1)
        
        self.robot_id = None
    
    def load_robot(self, urdf_path, position=(0, 0, 1), orientation=(0, 0, 0, 1)):
        """
        Load a robot model from a URDF file.
        
        Args:
            urdf_path (str): Path to the URDF file
            position (tuple): Initial position (x, y, z)
            orientation (tuple): Initial orientation as quaternion (x, y, z, w)
        
        Returns:
            int: Robot body ID
        """
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False
        )
        return self.robot_id
    
    def load_simple_box(self, mass=1.0, position=(0, 0, 1), size=(0.1, 0.1, 0.1)):
        """
        Create a simple box object for testing.
        
        Args:
            mass (float): Mass of the box
            position (tuple): Initial position
            size (tuple): Dimensions (length, width, height)
        
        Returns:
            int: Body ID of the box
        """
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size],
                                          rgbaColor=[0.5, 0.5, 0.8, 1])
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        return body_id
    
    def apply_force(self, body_id, force, position=None):
        """
        Apply a force to a body.
        
        Args:
            body_id (int): ID of the body
            force (tuple): Force vector (fx, fy, fz)
            position (tuple): Position where force is applied (world coordinates)
        """
        if position is None:
            position = [0, 0, 0]
        p.applyExternalForce(
            body_id,
            -1,  # Link index (-1 for base)
            force,
            position,
            p.WORLD_FRAME
        )
    
    def set_joint_motor_control(self, body_id, joint_id, target_position, 
                               max_force=500, kp=1.0, kd=0.1):
        """
        Control a joint with position control (PD controller).
        
        Args:
            body_id (int): ID of the robot body
            joint_id (int): Index of the joint
            target_position (float): Target joint angle
            max_force (float): Maximum force the motor can apply
            kp (float): Proportional gain
            kd (float): Derivative gain
        """
        p.setJointMotorControl2(
            body_id,
            joint_id,
            p.POSITION_CONTROL,
            targetPosition=target_position,
            maxForce=max_force,
            positionGain=kp,
            velocityGain=kd
        )
    
    def get_body_state(self, body_id):
        """
        Get the current state of a body.
        
        Returns:
            dict: Position, orientation, linear velocity, angular velocity
        """
        pos, orn = p.getBasePositionAndOrientation(body_id)
        lin_vel, ang_vel = p.getBaseVelocity(body_id)
        
        return {
            'position': np.array(pos),
            'orientation': np.array(orn),
            'linear_velocity': np.array(lin_vel),
            'angular_velocity': np.array(ang_vel)
        }
    
    def step_simulation(self):
        """Execute one simulation step."""
        p.stepSimulation()
        self.step_count += 1
    
    def run_simulation(self, duration=5.0, callback=None):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration (float): Duration of simulation in seconds
            callback (callable): Optional callback function called at each step
        """
        num_steps = int(duration / self.time_step)
        
        for i in range(num_steps):
            if callback:
                callback(i)
            
            self.step_simulation()
            
            # In GUI mode, add small sleep to see motion smoothly
            time.sleep(0.001)
    
    def cleanup(self):
        """Disconnect from PyBullet."""
        p.disconnect()


def example_free_fall():
    """Example 1: Simple free fall of a box."""
    print("Example 1: Free Fall Simulation")
    print("=" * 50)
    
    sim = RobotSimulator(use_gui=True)
    
    # Create a box
    box_id = sim.load_simple_box(mass=1.0, position=(0, 0, 2), size=(0.2, 0.2, 0.2))
    
    # Run simulation
    def callback(step):
        state = sim.get_body_state(box_id)
        if step % 24 == 0:  # Print every 10 frames
            print(f"Step {step}: Z position = {state['position'][2]:.3f}, "
                  f"Z velocity = {state['linear_velocity'][2]:.3f}")
    
    sim.run_simulation(duration=3.0, callback=callback)
    sim.cleanup()
    print("Simulation complete!\n")


def example_pushed_box():
    """Example 2: Box being pushed by applied forces."""
    print("Example 2: Pushed Box Simulation")
    print("=" * 50)
    
    sim = RobotSimulator(use_gui=True)
    
    # Create a box in the middle
    box_id = sim.load_simple_box(mass=1.0, position=(0, 0, 1), size=(0.2, 0.2, 0.2))
    
    # Run simulation with periodic forces
    def callback(step):
        # Apply a push force every 60 steps
        if step % 60 == 0:
            sim.apply_force(box_id, force=(5, 0, 0), position=(0, 0, 0))
        
        if step % 24 == 0:
            state = sim.get_body_state(box_id)
            print(f"Step {step}: Position = ({state['position'][0]:.2f}, "
                  f"{state['position'][1]:.2f}, {state['position'][2]:.2f})")
    
    sim.run_simulation(duration=5.0, callback=callback)
    sim.cleanup()
    print("Simulation complete!\n")


def example_rolling_motion():
    """Example 3: Multiple objects simulating rolling motion."""
    print("Example 3: Rolling Motion Simulation")
    print("=" * 50)
    
    sim = RobotSimulator(use_gui=True)
    
    # Create a cylinder (rolling object)
    cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.05)
    cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, height=0.05,
                                          rgbaColor=[1.0, 0.0, 0.0, 1])
    cylinder_id = p.createMultiBody(
        baseMass=2.0,
        baseCollisionShapeIndex=cylinder_shape,
        baseVisualShapeIndex=cylinder_visual,
        basePosition=(0, 0, 0.1)
    )
    
    # Give it initial angular velocity
    p.resetBaseVelocity(cylinder_id, linearVelocity=(0, 0, 0),
                       angularVelocity=(0, 20, 0))
    
    # Create an inclined plane
    incline_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.5, 0.1, 0.05))
    incline_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(0.5, 0.1, 0.05),
                                         rgbaColor=[0.7, 0.7, 0.7, 1])
    incline_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=incline_shape,
        baseVisualShapeIndex=incline_visual,
        basePosition=(0, 0, 0)
    )
    
    def callback(step):
        if step % 24 == 0:
            state = sim.get_body_state(cylinder_id)
            print(f"Step {step}: Position = {state['position']}, "
                  f"AngVel = {state['angular_velocity']}")
    
    sim.run_simulation(duration=3.0, callback=callback)
    sim.cleanup()
    print("Simulation complete!\n")


if __name__ == "__main__":
    print("\nPyBullet Robot Motion Simulator Examples")
    print("=" * 50)
    print()
    
    # Run examples
    try:
        example_free_fall()
        example_pushed_box()
        example_rolling_motion()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
