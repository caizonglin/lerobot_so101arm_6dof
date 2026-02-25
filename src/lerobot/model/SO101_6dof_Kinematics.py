import numpy as np
import math
from typing import List, Union, Tuple, Dict
import os

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not found. Please run: pip install pybullet")
    p = None
    pybullet_data = None

class SO101_6dof_Kinematics_PyBullet:
    """
    A class to represent the kinematics of a SO101 robot arm with 6 degrees of freedom,
    using pybullet for all calculations.
    """
    URDF_PATH = "examples/phone_to_so100/SO101/d2lrobot_so101_yaw.urdf"

    def __init__(self):
        if p is None:
            raise ImportError("pybullet is required. Please ensure it is installed correctly.")
        
        # Connect to pybullet in DIRECT mode (no GUI)
        self.client = p.connect(p.DIRECT)

        try:
            # Load the URDF. Pybullet will now find the 'assets' directory relative to the URDF file.
            self.robot_id = p.loadURDF(self.URDF_PATH, useFixedBase=True, physicsClientId=self.client)
            print(f"Successfully loaded URDF into pybullet from: {self.URDF_PATH}")
        except p.error as e:
            raise RuntimeError(f"Failed to load URDF into pybullet from {self.URDF_PATH}: {e}")

        # Build a mapping from joint name to joint index
        self.joint_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_name = info[1].decode('UTF-8')
            self.joint_name_to_id[joint_name] = i
        
        print(f"Active joints found by pybullet: {list(self.joint_name_to_id.keys())}")

        # MAPPING from our Python motor names to URDF joint names
        self.motor_to_urdf_joint_map: Dict[str, str] = {
            "shoulder_pan": "joint1",
            "shoulder_lift": "joint2",
            "elbow_flex": "joint3",
            "wrist_flex": "joint4",
            "wrist_yaw": "joint5",
            "wrist_roll": "joint6",
            "gripper": "joint7"
        }
        
        # Find the end-effector link index
        self.ee_link_id = -1
        ee_link_name = "gripper_body"
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            link_name = info[12].decode('UTF-8')
            if link_name == ee_link_name:
                self.ee_link_id = i
                break
        if self.ee_link_id == -1:
            raise RuntimeError(f"Could not find end-effector link named '{ee_link_name}' in URDF.")
        
        print(f"End-effector link '{ee_link_name}' found with index: {self.ee_link_id}")

        # Define a rest pose for the IK solver
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        self.rest_poses = [0.0] * self.num_joints

        # Define joint damping to encourage use of specific joints
        # Lower values mean the joint is "cheaper" to move for the solver.
        # We set a lower value for shoulder_lift (joint2, index 1) to encourage its movement.
        self.joint_damping = [0.1] * self.num_joints
        shoulder_lift_joint_id = self.joint_name_to_id.get("joint2")
        if shoulder_lift_joint_id is not None:
            self.joint_damping[shoulder_lift_joint_id] = 0.01


    def forward_kinematics(self, motor_angles_degrees: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the end-effector pose for a given set of motor angles.
        """
        # Reset all joint states to the target angles
        for motor_name, angle_deg in motor_angles_degrees.items():
            urdf_joint_name = self.motor_to_urdf_joint_map.get(motor_name)
            joint_id = self.joint_name_to_id.get(urdf_joint_name)
            if joint_id is not None:
                p.resetJointState(self.robot_id, joint_id, targetValue=np.radians(angle_deg), physicsClientId=self.client)

        # Get the state of the end-effector link
        # The result includes world position and orientation (quaternion)
        link_state = p.getLinkState(self.robot_id, self.ee_link_id, computeForwardKinematics=True, physicsClientId=self.client)
        position = np.array(link_state[0])
        orientation_quat = link_state[1] # (x, y, z, w)
        
        # Convert quaternion to RPY
        rpy = p.getEulerFromQuaternion(orientation_quat, physicsClientId=self.client)
        
        return position, np.array(rpy)


    def inverse_kinematics(self, target_position: List[float], target_orientation_rpy: List[float], custom_rest_poses_deg: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate the joint angles required to reach a target end-effector pose.
        
        Args:
            target_position: Target [x, y, z] position.
            target_orientation_rpy: Target [roll, pitch, yaw] orientation in radians.
            custom_rest_poses_deg: Optional dictionary of motor names to angles in degrees.
                                   If provided, these are used as the rest pose for the IK solver.
        """
        target_orientation_quat = p.getQuaternionFromEuler(target_orientation_rpy)

        active_rest_poses = self.rest_poses
        if custom_rest_poses_deg is not None:
            # Create a temporary list for the rest poses in pybullet's joint order
            dynamic_rest_poses_rad = list(self.rest_poses) 
            for motor_name, angle_deg in custom_rest_poses_deg.items():
                urdf_joint_name = self.motor_to_urdf_joint_map.get(motor_name)
                if urdf_joint_name:
                    joint_id = self.joint_name_to_id.get(urdf_joint_name)
                    if joint_id is not None:
                         dynamic_rest_poses_rad[joint_id] = np.radians(angle_deg)
            active_rest_poses = dynamic_rest_poses_rad

        # PyBullet's IK solver. We provide a rest pose to get a more stable solution.
        joint_poses_rad = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_id,
            target_position,
            targetOrientation=target_orientation_quat,
            restPoses=active_rest_poses,
            jointDamping=self.joint_damping,
            physicsClientId=self.client
        )
        
        solved_motor_angles_deg = {}
        for motor_name, urdf_joint_name in self.motor_to_urdf_joint_map.items():
            joint_id = self.joint_name_to_id.get(urdf_joint_name)
            if joint_id is not None and joint_id < len(joint_poses_rad): # Ensure index is valid for joint_poses_rad
                solved_motor_angles_deg[motor_name] = np.degrees(joint_poses_rad[joint_id])
                
        return solved_motor_angles_deg

    def disconnect(self):
        p.disconnect(self.client)

if __name__ == "__main__":
    kin = None
    try:
        kin = SO101_6dof_Kinematics_PyBullet()
        print("\nKinematics class initialized successfully with pybullet.")
        
        dummy_motor_angles = {
            "shoulder_pan": 0, "shoulder_lift": 45, "elbow_flex": -90,
            "wrist_flex": 0, "wrist_yaw": 0, "wrist_roll": 0, "gripper": 0
        }
        position, rpy = kin.forward_kinematics(dummy_motor_angles)
        print(f"\nForward Kinematics for angles (degrees):")
        print(f"  Position: {position}")
        print(f"  Orientation (RPY degrees): {np.degrees(rpy)}")

        target_pos = [0.2, 0.1, 0.3]
        target_rpy_deg = [0.0, 90.0, 0.0]
        target_rpy_rad = np.radians(target_rpy_deg).tolist()
        
        solved_motor_angles = kin.inverse_kinematics(target_pos, target_rpy_rad)
        print(f"\nInverse Kinematics for target pos {target_pos}, rpy {target_rpy_deg} (degrees):")
        print(f"  Solved angles (degrees):")
        # Pretty print the dictionary
        for motor, angle in solved_motor_angles.items():
            print(f"    {motor}: {angle:.2f}")

    except ImportError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if kin:
            kin.disconnect()