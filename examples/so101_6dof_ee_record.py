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
A script to record end-effector space teleoperation between two 6-DOF SO-101 robots,
with keyboard controls for the recording session.
This script is a hybrid of 'examples/ee_teleop.py' (for kinematics) and
'src/lerobot/scripts/lerobot_record.py' (for recording session handling).

Example usage:
python examples/so101_6dof_ee_record.py \
    --follower.type=so101_6dof_follower \
    --follower.port=/dev/ttyACM0 \
    --follower.cameras="{camera_top: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}" \
    --leader.type=so101_6dof_leader \
    --leader.port=/dev/ttyUSB1 \
    --dataset.repo_id="lerobot/so101_ee_teleop_demo" \
    --dataset.single_task="Follow the leader's end-effector" \
    --display_data
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import numpy as np
from lerobot.configs import parser
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.datasets.utils import (
    build_dataset_frame,
    hw_to_dataset_features,
    combine_feature_dicts,
)
from lerobot.utils.constants import ACTION, OBS_STR

# Needed to register the robots, teleops and cameras
from lerobot.robots import so101_6dof_follower
from lerobot.teleoperators import so101_6dof_leader
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@dataclass
class EERecordConfig:
    """Config for the 6-DOF End-Effector Recording script."""

    follower: RobotConfig
    leader: TeleoperatorConfig
    dataset: DatasetRecordConfig
    frequency: float = 30.0
    display_data: bool = False
    resume: bool = False


@parser.wrap()
def main(cfg: EERecordConfig):
    """Main function to run the end-effector teleoperation recording."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="so101_6dof_ee_record")

    follower_robot = make_robot_from_config(cfg.follower)
    leader_teleop = make_teleoperator_from_config(cfg.leader)

    # Define dataset features
    obs_features = hw_to_dataset_features(
        follower_robot.observation_features, OBS_STR, use_video=cfg.dataset.video
    )
    action_features = {
        ACTION: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
    }
    dataset_features = combine_feature_dicts(obs_features, action_features)

    # Create the dataset
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=cfg.dataset.repo_id,
            fps=cfg.frequency,
            features=dataset_features,
            root=cfg.dataset.root,
            robot_type=follower_robot.name,
            use_videos=cfg.dataset.video,
            image_writer_threads=(
                cfg.dataset.num_image_writer_threads_per_camera * len(follower_robot.cameras)
                if follower_robot.cameras
                else 0
            ),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    listener = None
    try:
        log_say("正在连接设备...")
        follower_robot.connect()
        leader_teleop.connect()

        listener, events = init_keyboard_listener()
        log_say(
            "准备就绪。请开始遥操作。使用键盘控制录制流程:\n"
            "- 右箭头 (->): 完成当前 episode\n"
            "- 左箭头 (<-): 重新录制当前 episode\n"
            "- ESC:      退出程序"
        )

        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"等待录制第 {dataset.num_episodes} 个 episode...")

            record_episode(cfg, follower_robot, leader_teleop, dataset, events)

            if events["rerecord_episode"]:
                log_say("重新录制当前 episode。")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if not events["stop_recording"]:
                dataset.save_episode()
                recorded_episodes += 1
                log_say(f"第 {recorded_episodes} 个 episode 已保存。")

                # Add a reset period after saving an episode
                if recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                    log_say(f"请在 {cfg.dataset.reset_time_s} 秒内重置环境...")
                    time.sleep(cfg.dataset.reset_time_s)

            # Reset exit_early flag for the next episode
            if events["exit_early"]:
                events["exit_early"] = False


    except KeyboardInterrupt:
        print()
        logging.info("捕获到 KeyboardInterrupt。正在关闭。")
    except Exception as e:
        logging.error(f"发生异常: {e}", exc_info=True)
    finally:
        log_say("正在断开设备连接...")
        if follower_robot and follower_robot.is_connected:
            follower_robot.disconnect()
        if leader_teleop and leader_teleop.is_connected:
            leader_teleop.disconnect()

        # Clean up empty image directories if videos were used
        if 'dataset' in locals() and dataset and cfg.dataset.video:
            images_dir = dataset.root / "images"
            if images_dir.is_dir():
                # Walk the directory tree from the bottom up and remove empty directories
                for dirpath, dirnames, filenames in os.walk(images_dir, topdown=False):
                    if not dirnames and not filenames:
                        try:
                            os.rmdir(dirpath)
                            logging.info(f"已移除空的临时目录: {dirpath}")
                        except OSError as e:
                            logging.warning(f"无法移除目录 {dirpath}: {e}")

        dataset.finalize()
        if listener:
            listener.stop()

        log_say("关闭完成。")
        if cfg.dataset.push_to_hub:
            log_say("正在将数据集推送到 Hub...")
            dataset.push_to_hub()
            log_say("数据集已推送到 Hub。")


def record_episode(
    cfg: EERecordConfig,
    follower: Any,
    leader: Any,
    dataset: LeRobotDataset,
    events: dict,
):
    """Record a single episode of teleoperation."""
    initial_pos, _ = leader.get_ee_pose()
    follower_initial_pos, _ = follower.get_ee_pose()

    start_time = time.time()
    while not events["exit_early"] and not events["stop_recording"]:
        loop_start_time = time.perf_counter()

        leader_action = leader.get_action()
        leader_motor_angles = {key.replace(".pos", ""): value for key, value in leader_action.items()}

        leader_pos, leader_rpy = leader.kinematics.forward_kinematics(leader_motor_angles)

        target_pos = follower_initial_pos + (leader_pos - initial_pos)
        target_rpy = leader_rpy

        follower_arm_angles_deg = follower.kinematics.inverse_kinematics(
            target_pos.tolist(), target_rpy.tolist(), custom_rest_poses_deg=leader_motor_angles
        )

        gripper_openness_leader = leader_motor_angles.get("gripper", 0.0)
        follower_arm_angles_deg["gripper"] = 100.0 - gripper_openness_leader

        final_action_to_send = {f"{motor}.pos": angle for motor, angle in follower_arm_angles_deg.items()}
        follower.send_action(final_action_to_send)

        obs = follower.get_observation()

        # Construct a flat dictionary for action_to_log for build_dataset_frame
        action_pose_vec = np.concatenate((target_pos, target_rpy))
        action_to_log = dict(zip(['x', 'y', 'z', 'roll', 'pitch', 'yaw'], map(float, action_pose_vec)))
        action_to_log['gripper'] = float((100.0 - gripper_openness_leader) / 100.0)

        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, action_to_log, prefix=ACTION)
        frame = {**observation_frame, **action_frame, "task": cfg.dataset.single_task}
        dataset.add_frame(frame)

        if cfg.display_data:
            log_rerun_data(observation=obs, action=action_to_log)

        elapsed_time = time.perf_counter() - loop_start_time
        sleep_time = 1.0 / cfg.frequency - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if (time.time() - start_time) > cfg.dataset.episode_time_s:
            log_say("Episode 时间限制已到。")
            break


if __name__ == "__main__":
    main()

