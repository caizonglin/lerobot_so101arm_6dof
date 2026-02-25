#!/usr/bin/env python3
"""
This script runs inference (evaluation) using a trained policy for a Bi-SO100 setup.

It loads a policy from the Hugging Face Hub, runs it on the robot, and saves
the resulting episodes to a new evaluation dataset on the Hub.
"""

import logging

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots.bi_so100_follower import BiSO100Follower, BiSO100FollowerConfig
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.utils.utils import log_say

# --- Configuration ---
# TODO: Update these ports to match your robot connections.
# [HARDWARE] Ports for the robot (follower)
FOLLOWER_LEFT_ARM_PORT = "/dev/ttyACM2"  # <-- 请务必修改这里
FOLLOWER_RIGHT_ARM_PORT = "/dev/ttyACM3" # <-- 请务必修改这里

# [REQUIRED] The policy model to be loaded and evaluated.
HF_MODEL_ID = "zonglin11/lerobot_bi_so100_pickup_lego"

# [REQUIRED] Used to load statistics for data normalization. Must match the dataset used for training the policy.
HF_TRAINING_DATASET_ID = "zonglin11/lerobot_bi_so100_pickup_lego"

# [REQUIRED] A new or existing repo where the evaluation episodes will be uploaded.
HF_EVAL_DATASET_ID = "zonglin11/lerobot_bi_so100_pickup_lego_eval" # 您可以按需修改

# [OPTIONAL] Evaluation parameters
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Evaluating a trained policy on a bimanual SO100 robot."

# [HARDWARE] Camera settings (ensure they match the training setup)
# The camera names (keys) here MUST match the names expected by the policy's preprocessor.
camera_configs = {
    "image": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
    "image2": OpenCVCameraConfig(index_or_path=6, width=320, height=240, fps=FPS),
    "empty_camera_0": OpenCVCameraConfig(index_or_path=8, width=320, height=240, fps=FPS),
}
# ---------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the evaluation loop."""
    log_say("Bi-SO100 Evaluation Script")
    log_say("=" * 50)

    # 1. Initialize robot
    follower_config = BiSO100FollowerConfig(
        id="bi_so100_follower",  # Used to generate unique IDs for left/right arms
        left_arm_port=FOLLOWER_LEFT_ARM_PORT,
        right_arm_port=FOLLOWER_RIGHT_ARM_PORT,
        cameras=camera_configs,
    )
    robot = BiSO100Follower(follower_config)

    # 2. Load policy from Hub
    log_say(f"Loading policy from: {HF_MODEL_ID}")
    policy = make_policy(HF_MODEL_ID)

    # 3. Load training dataset stats for processors
    log_say(f"Loading stats from training dataset: {HF_TRAINING_DATASET_ID}")
    training_dataset = LeRobotDataset.from_hub(HF_TRAINING_DATASET_ID)

    # 4. Create pre- and post-processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=training_dataset.stats,
    )

    # 5. Create or load the evaluation dataset
    log_say(f"Evaluation episodes will be saved to: {HF_EVAL_DATASET_ID}")
    try:
        # Attempt to load an existing evaluation dataset to resume.
        eval_dataset = LeRobotDataset(HF_EVAL_DATASET_ID)
        eval_dataset.meta.metadata_buffer_size = 1  # Force flush metadata to disk after each episode
        log_say(f"Successfully loaded existing eval dataset. Resuming from episode {eval_dataset.meta.total_episodes}.")
    except (FileNotFoundError, NotADirectoryError, RepositoryNotFoundError):
        # This block is executed if the dataset doesn't exist.
        from lerobot.utils.constants import HF_LEROBOT_HOME
        import shutil
        repo_path = HF_LEROBOT_HOME / HF_EVAL_DATASET_ID
        if repo_path.exists():
            log_say(f"Found an incomplete local directory for eval dataset. Cleaning it up.")
            shutil.rmtree(repo_path)

        log_say("No existing eval dataset found. Creating a new one.")
        eval_dataset = LeRobotDataset.create(
            repo_id=HF_EVAL_DATASET_ID,
            features=training_dataset.features,
            fps=FPS,
            robot_type=robot.name,
            use_videos=bool(robot.cameras),
        )
        eval_dataset.meta.metadata_buffer_size = 1  # Force flush metadata to disk after each episode

    # 6. Connect to the robot (calibration is skipped)
    robot.connect(calibrate=False)
    if not robot.is_connected:
        raise ConnectionError("Robot failed to connect.")

    # Check calibration status
    logger.info("Checking calibration status of follower arms...")
    if not robot.is_calibrated:
        logger.error("Follower arms are not calibrated.")
        logger.error("Please run the 'lerobot-calibrate' script for the 2 follower arms first.")
        logger.warning(
            "Example for a follower arm: lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/tty... --robot.id=follower_left"
        )
        raise RuntimeError("Arms not calibrated. Exiting.")
    logger.info("All arms are calibrated and ready.")

    # 7. Initialize keyboard listener for controls
    listener, events = init_keyboard_listener()

    # 8. Create default processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 9. Start evaluation loop
    log_say(f"Starting evaluation for {NUM_EPISODES} episodes.")
    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

        # Main loop for inference, passing the policy and processors
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if events["rerecord_episode"]:
            log_say("Re-recording the last episode.")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            eval_dataset.clear_episode_buffer()
            continue

        # Save the completed evaluation episode
        eval_dataset.save_episode()
        episode_idx += 1

    # 9. Clean up
    log_say("Evaluation finished. Cleaning up...")
    robot.disconnect()
    listener.stop()

    # Finalize and push evaluation dataset to the Hub
    log_say("Finalizing and uploading evaluation dataset to the Hub...")
    eval_dataset.finalize()
    eval_dataset.push_to_hub()
    log_say("Upload complete!")
    log_say("=" * 50)


if __name__ == "__main__":
    main()
