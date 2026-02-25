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
A script for joint-space teleoperation, specifically for a follower with a
parallel gripper whose motion is inverted and scaled relative to the leader.

Example usage:
python examples/joint_teleop_parallel_gripper.py --robot.port=/dev/ttyACM0 --teleop.port=/dev/ttyUSB1
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

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
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig # noqa: F401
from lerobot.utils.utils import init_logging


@dataclass
class Joint_Teleop_Parallel_Config:
    """Config for the 6-DOF Joint-Space Teleoperation script with parallel gripper."""
    robot: RobotConfig
    teleop: TeleoperatorConfig
    # Control frequency in Hz
    frequency: float = 50.0
    # Gripper scaling factor to match physical movement ranges.
    # > 1.0 -> follower gripper moves more
    # < 1.0 -> follower gripper moves less
    gripper_scale: float = 1.0


@parser.wrap()
def joint_teleop_parallel(cfg: Joint_Teleop_Parallel_Config):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    logger = logging.getLogger(__name__)
    logger.info("Initializing 6-DOF Joint-Space Teleoperation for Parallel Gripper")

    # Instantiate the 6-DOF leader and follower robots
    follower_robot = make_robot_from_config(cfg.robot)
    leader_teleop = make_teleoperator_from_config(cfg.teleop)

    try:
        logger.info("Connecting to the follower robot...")
        follower_robot.connect()
        logger.info("Connecting to the leader teleoperator...")
        leader_teleop.connect()

        logger.info("Starting joint-space teleoperation loop. Press Ctrl+C to exit.")

        while True:
            start_time = time.perf_counter()

            # --- JOINT SPACE CONTROL WITH GRIPPER REMAPPING ---
            # 1. Get joint angles from leader
            leader_action = leader_teleop.get_action()

            # Create a copy to modify
            follower_action = leader_action.copy()

            # 2. Remap the gripper value
            if "gripper.pos" in follower_action:
                # The value from get_action is normalized between 0 and 100
                original_gripper_value = follower_action["gripper.pos"]

                # a) Invert the motion
                inverted_gripper_value = 100.0 - original_gripper_value

                # b) Apply scaling around the midpoint (50)
                mid_point = 50.0
                scaled_gripper_value = mid_point + (inverted_gripper_value - mid_point) * cfg.gripper_scale

                # c) Clamp the value to ensure it stays within the valid [0, 100] range
                final_gripper_value = max(0.0, min(100.0, scaled_gripper_value))

                follower_action["gripper.pos"] = final_gripper_value

            # 3. Send the modified action to the follower
            follower_robot.send_action(follower_action)

            # Maintain control frequency
            elapsed_time = time.perf_counter() - start_time
            sleep_time = 1.0 / cfg.frequency - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_s = time.perf_counter() - start_time
            hz_value = 1 / loop_s if loop_s > 0 else 0
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({hz_value:.0f} Hz)", end="\r")

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
    joint_teleop_parallel()

