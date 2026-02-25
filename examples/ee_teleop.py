#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run end-effector space teleoperation between two 6-DOF SO-101 robots.
This script uses the kinematics classes to map the leader's end-effector pose to the follower.

Example usage:
python examples/ee_teleop.py --robot.type=so101_6dof_follower --robot.port=/dev/ttyACM0 --teleop.type=so101_6dof_leader --teleop.port=/dev/ttyUSB1
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so101_6dof_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so101_6dof_leader,
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.utils.utils import init_logging


@dataclass
class EE_TeleopConfig:
    """Config for the 6-DOF End-Effector Teleoperation script."""
    robot: RobotConfig
    teleop: TeleoperatorConfig
    # Control frequency in Hz
    frequency: float = 20.0

@parser.wrap()
def ee_teleop(cfg: EE_TeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 初始化logger实例，修复原代码中未定义logger的问题
    logger = logging.getLogger(__name__)
    logger.info("Initializing 6-DOF End-Effector Teleoperation")

    # Instantiate the 6-DOF leader and follower robots
    follower_robot = make_robot_from_config(cfg.robot)
    leader_teleop = make_teleoperator_from_config(cfg.teleop)

    try:
        logger.info("Connecting to the follower robot...")
        follower_robot.connect()
        logger.info("Connecting to the leader teleoperator...")
        leader_teleop.connect()

        logger.info("Starting end-effector pose teleoperation loop. Press Ctrl+C to exit.")

        # Get the initial pose of the leader to start
        initial_pos, _ = leader_teleop.get_ee_pose()
        # Get the initial pose of the follower
        follower_initial_pos, _ = follower_robot.get_ee_pose()

        while True:
            start_time = time.perf_counter()
        
            # 1. Get full action from leader (includes gripper angle)
            leader_action = leader_teleop.get_action()
            leader_motor_angles = {key.replace(".pos", ""): value for key, value in leader_action.items()}
            
            # 2. Use FK to get leader's EE pose for the arm part
            leader_pos, leader_rpy = leader_teleop.kinematics.forward_kinematics(leader_motor_angles)
        
            # 3. Calculate target pose for the follower arm
            target_pos = follower_initial_pos + (leader_pos - initial_pos)
            target_rpy = leader_rpy
            
            # 4. Get the IK solution for the follower's arm joints, using the leader's pose as the rest pose
            follower_arm_angles_deg = follower_robot.kinematics.inverse_kinematics(
                target_pos.tolist(), target_rpy.tolist(), custom_rest_poses_deg=leader_motor_angles
            )
            
            # 5. MERGE the arm IK solution with the leader's gripper angle
            #    Reverse the gripper action (100 - value) to fix the reversed open/close behavior.
            follower_arm_angles_deg["gripper"] = 100.0 - leader_motor_angles.get("gripper", 0.0)
        
            # 6. Create the final action dictionary and send it to the follower
            final_action_to_send = {f"{motor}.pos": angle for motor, angle in follower_arm_angles_deg.items()}
            
            # --- DEBUG PRINT: Show the calculated IK solution ---
            ik_solution_str = ", ".join([f"{motor}: {angle:.1f}" for motor, angle in follower_arm_angles_deg.items()])
            print(f"IK Solution (deg): {ik_solution_str}".ljust(150), end="\r")
            
            follower_robot.send_action(final_action_to_send)
            
            # Maintain control frequency
            elapsed_time = time.perf_counter() - start_time
            sleep_time = 1.0 / cfg.frequency - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print()
        logger.info("Caught KeyboardInterrupt. Shutting down.")
    except Exception as e:
        logger.error(f"An exception occurred: {e}", exc_info=True)
    finally:
        logger.info("Disconnecting devices...")
        if follower_robot and follower_robot.is_connected:
            follower_robot.disconnect()
        if leader_teleop and leader_teleop.is_connected:
            leader_teleop.disconnect()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    ee_teleop()

