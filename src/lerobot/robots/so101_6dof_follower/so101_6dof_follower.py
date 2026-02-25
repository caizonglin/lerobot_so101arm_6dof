#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ...model.SO101_6dof_Kinematics import SO101_6dof_Kinematics_PyBullet
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_6dof_follower import SO101_6dof_FollowerConfig

logger = logging.getLogger(__name__)


class SO101_6dof_Follower(Robot):
    """
    SO-101 Follower Arm with 6-DOF, with added kinematics for end-effector control.
    """

    config_class = SO101_6dof_FollowerConfig
    name = "so101_6dof_follower"

    def __init__(self, config: SO101_6dof_FollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "wrist_yaw": Motor(7, "sts3215", norm_mode_body), # ADDED 6th DOF
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        # Lazily initialize kinematics only when needed
        self._kinematics = None

    @property
    def kinematics(self) -> SO101_6dof_Kinematics_PyBullet:
        """Lazy loader for the kinematics object."""
        if self._kinematics is None:
            logger.info("Initializing kinematics for SO101_6dof_Follower...")
            try:
                self._kinematics = SO101_6dof_Kinematics_PyBullet()
            except ImportError as e:
                logger.error(f"Failed to initialize kinematics: {e}")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during kinematics initialization: {e}")
                raise
        return self._kinematics

    def set_ee_pose(self, target_position: List[float], target_orientation_rpy: List[float]):
        """
        Set the end-effector to a target pose using inverse kinematics.

        Args:
            target_position: Target [x, y, z] position.
            target_orientation_rpy: Target [roll, pitch, yaw] orientation in radians.
        """
        logger.debug(f"Setting EE pose to pos:{target_position}, rpy:{target_orientation_rpy}")

        # Use inverse kinematics to get the target joint angles in degrees
        motor_angles_deg = self.kinematics.inverse_kinematics(target_position, target_orientation_rpy)

        # The action dictionary for send_action expects keys like "motor_name.pos"
        action = {f"{motor}.pos": angle for motor, angle in motor_angles_deg.items()}

        # Send the calculated joint angles to the motors
        self.send_action(action)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current end-effector pose using forward kinematics.

        Returns:
            A tuple (position, orientation_rpy) where position is [x, y, z]
            and orientation_rpy is [roll, pitch, yaw] in radians.
        """
        # Get current joint angles from the robot
        current_obs = self.get_observation()
        motor_angles_degrees = {key.replace(".pos", ""): value for key, value in current_obs.items() if ".pos" in key}

        # Use forward kinematics to calculate the pose
        position, rpy = self.kinematics.forward_kinematics(motor_angles_degrees)
        return position, rpy

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"Running calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion. Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        print("即将发送的马达指令:", goal_pos)
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        # Disconnect kinematics if it was initialized
        if self._kinematics is not None:
            self._kinematics.disconnect()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

