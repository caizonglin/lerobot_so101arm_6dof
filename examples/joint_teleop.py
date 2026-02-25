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
A script to run direct joint-space teleoperation between two 6-DOF SO-101 robots,
with optional Rerun visualization.

Example usage:
# Without visualization
python examples/joint_teleop.py --robot.port=/dev/ttyACM0 --teleop.port=/dev/ttyUSB1

# With visualization
python examples/joint_teleop.py --robot.port=/dev/ttyACM0 --teleop.port=/dev/ttyUSB1 --display_data=true
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr
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
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class Joint_TeleopConfig:
    """Config for the 6-DOF Joint-Space Teleoperation script."""
    robot: RobotConfig
    teleop: TeleoperatorConfig
    # Control frequency in Hz
    frequency: float = 50.0
    # Gripper scaling factor to match physical movement ranges.
    gripper_scale: float = 1.0
    # Display all cameras and robot data on screen using Rerun
    display_data: bool = False


@parser.wrap(config_path="configs", config_name="joint_teleop_config")
def joint_teleop(cfg: Joint_TeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="joint_space_teleop")

    logger = logging.getLogger(__name__)
    logger.info("Initializing 6-DOF Joint-Space Teleoperation")

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

            leader_action = leader_teleop.get_action()
            follower_action = leader_action.copy()

            if "gripper.pos" in follower_action:
                original_gripper_value = follower_action["gripper.pos"]
                inverted_gripper_value = 100.0 - original_gripper_value
                mid_point = 50.0
                scaled_gripper_value = mid_point + (inverted_gripper_value - mid_point) * cfg.gripper_scale
                final_gripper_value = max(0.0, min(100.0, scaled_gripper_value))
                follower_action["gripper.pos"] = final_gripper_value

            follower_robot.send_action(follower_action)

            if cfg.display_data:
                # Get observation from follower for visualization
                obs = follower_robot.get_observation()
                log_rerun_data(observation=obs, action=follower_action)

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
        if cfg.display_data:
            rr.rerun_shutdown()
        if follower_robot and follower_robot.is_connected:
            follower_robot.disconnect()
        if leader_teleop and leader_teleop.is_connected:
            leader_teleop.disconnect()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    import draccus
    import os
    
    # Construct the path to the default config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the configs directory is at the project root, relative to the examples/ dir
    config_path = os.path.join(current_dir, "..", "configs")
    config_name = "joint_teleop_config"
    
    # Manually create a lightweight parser to load the config
    cfg = draccus.parse(config_class=Joint_TeleopConfig, config_path=config_path, config_name=config_name)
    
    # Call the main function with the loaded config
    joint_teleop(cfg)

