#!/usr/bin/env python3
"""
This script records teleoperation data for a Bi-SO100 setup.

It uses a BiSO100Leader as the master device and a BiSO100Follower as the robot,
and saves the trajectory data to a Hugging Face dataset.

This script performs direct joint-to-joint mapping without any complex
kinematic processing pipelines.

NOTE: It is assumed that the robot has been calibrated beforehand.
"""

import logging

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.robots.bi_so100_follower import BiSO100Follower, BiSO100FollowerConfig
from lerobot.processor import make_default_processors
from lerobot.utils.control_utils import init_keyboard_listener
from huggingface_hub.errors import RepositoryNotFoundError

from lerobot.utils.utils import log_say

# --- Configuration ---
# TODO: Update these ports to match your robot connections.
# [HARDWARE] Ports for the master device (leader)
LEADER_LEFT_ARM_PORT = "/dev/ttyACM1"
LEADER_RIGHT_ARM_PORT = "/dev/ttyACM0"
# [HARDWARE] Ports for the robot (follower)
FOLLOWER_LEFT_ARM_PORT = "/dev/ttyACM2"
FOLLOWER_RIGHT_ARM_PORT = "/dev/ttyACM3"

# TODO: Set your Hugging Face Hub dataset repository ID.
# [REQUIRED] This is where the dataset will be uploaded. e.g. "my-username/my-dataset"
HF_REPO_ID = "zonglin11/bi_so100_test_dataset_3"

# [OPTIONAL] Recording parameters
NUM_EPISODES = 3
FPS = 30
EPISODE_TIME_SEC = 40
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Controlling a bimanual SO100 robot to pick blue and yellow lego into cup."

# [HARDWARE] Camera settings
# If you have no cameras, set this to an empty dictionary: `camera_configs = {}`
camera_configs = {
    # You can name your cameras anything you like.
    "front_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
    "left_wrist_camera": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=FPS),
    "right_wrist_camera": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=FPS),
}

# --- Visualization ---
# [OPTIONAL] Whether to display camera feeds and other data in a Rerun window.
DISPLAY_DATA = True
# ---------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the recording loop."""
    log_say("Bi-SO100 Data Recording Script")
    log_say("=" * 50)

    if DISPLAY_DATA:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(session_name="recording")

    # 1. Initialize leader and follower
    leader_config = BiSO100LeaderConfig(
        id="bi_so100_leader",  # Used to generate unique IDs for left/right arms
        left_arm_port=LEADER_LEFT_ARM_PORT,
        right_arm_port=LEADER_RIGHT_ARM_PORT,
    )
    leader = BiSO100Leader(leader_config)

    follower_config = BiSO100FollowerConfig(
        id="bi_so100_follower",  # Used to generate unique IDs for left/right arms
        left_arm_port=FOLLOWER_LEFT_ARM_PORT,
        right_arm_port=FOLLOWER_RIGHT_ARM_PORT,
        cameras=camera_configs,  # Pass the camera configs here
    )
    follower = BiSO100Follower(follower_config)

    # 2. Create the dataset
    # Convert raw hardware features to LeRobot dataset feature format
    action_features = hw_to_dataset_features(leader.action_features, prefix="action")
    observation_features = hw_to_dataset_features(
        follower.observation_features, prefix="observation", use_video=bool(follower.cameras)
    )
    dataset_features = {**action_features, **observation_features}

    # When resuming, we load the existing dataset instead of creating a new one.
    try:
        # Attempt to load the existing dataset
        dataset = LeRobotDataset(HF_REPO_ID)
        dataset.meta.metadata_buffer_size = 1  # Force flush metadata to disk after each episode
        log_say(f"Successfully loaded existing dataset. Resuming from episode {dataset.meta.total_episodes}.")
    except (FileNotFoundError, NotADirectoryError, RepositoryNotFoundError):
        # This block is executed if the dataset doesn't exist, either locally or on the Hub.
        
        # The attempt to load might have created an empty directory. Clean it up before creating.
        from lerobot.utils.constants import HF_LEROBOT_HOME
        import shutil
        repo_path = HF_LEROBOT_HOME / HF_REPO_ID
        if repo_path.exists():
            log_say(f"Found an incomplete local directory at {repo_path}. Cleaning it up before starting fresh.")
            shutil.rmtree(repo_path)

        log_say("No existing dataset found. Creating a new one.")
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            features=dataset_features,
            fps=FPS,
            robot_type=follower.name,
            # Set use_videos to False if you don't have cameras configured
            use_videos=bool(follower.cameras),
        )
        dataset.meta.metadata_buffer_size = 1  # Force flush metadata to disk after each episode
    log_say(f"Dataset will be saved to Hugging Face Hub repo: {HF_REPO_ID}")

    # 3. Connect to devices (calibration is skipped)
    leader.connect(calibrate=False)
    follower.connect(calibrate=False)

    if not leader.is_connected or not follower.is_connected:
        raise ConnectionError("Leader or Follower failed to connect.")

    # Check calibration status
    logger.info("Checking calibration status of all arms...")
    if not leader.is_calibrated or not follower.is_calibrated:
        logger.error("One or more arms are not calibrated.")
        logger.error("Please run the 'lerobot-calibrate' script for all 4 arms first.")
        logger.warning(
            "Example for a leader arm: lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/tty... --teleop.id=leader_left"
        )
        logger.warning(
            "Example for a follower arm: lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/tty... --robot.id=follower_left"
        )
        raise RuntimeError("Arms not calibrated. Exiting.")
    logger.info("All arms are calibrated and ready.")

    # 4. Initialize keyboard listener for recording controls (e.g., stop)
    listener, events = init_keyboard_listener()

    # 5. Create default processors for direct joint-to-joint recording
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 6. Start recording loop
    # Start from the next episode index if resuming
    start_episode_idx = dataset.meta.total_episodes
    # Adjust NUM_EPISODES to be the target total number of episodes
    log_say(f"Will record until a total of {NUM_EPISODES} episodes are saved.")
    episode_idx = start_episode_idx
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1}")

        # Main record loop from lerobot scripts
        record_loop(
            robot=follower,
            events=events,
            fps=FPS,
            teleop=leader,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_data=DISPLAY_DATA,
        )

        # Reset phase between episodes
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop=leader,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                display_data=DISPLAY_DATA,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording the last episode.")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save the completed episode
        dataset.save_episode()
        episode_idx += 1

    # 6. Clean up
    log_say("Recording finished")
    leader.disconnect()
    follower.disconnect()
    listener.stop()

    # Finalize and push dataset to the Hub
    log_say("Finalizing and uploading dataset to the Hub...")
    dataset.finalize()
    dataset.push_to_hub()
    log_say("Upload complete!")
    log_say("=" * 50)


if __name__ == "__main__":
    main()
