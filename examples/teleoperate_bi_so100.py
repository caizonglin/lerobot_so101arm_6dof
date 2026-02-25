#!/usr/bin/env python3
"""
This script runs real-time teleoperation for a Bi-SO100 setup.

It uses a BiSO100Leader as the master device and a BiSO100Follower as the robot,
continuously sending actions from the leader to the follower.

NOTE: This script does NOT record any data. For that, use 'record_bi_so100.py'.
It is also assumed that the robot has been calibrated beforehand, for example,
by running 'calibrate_bi_so100.py'.
"""

import logging
import time
import traceback

from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.robots.bi_so100_follower import BiSO100Follower, BiSO100FollowerConfig

# --- Configuration ---
# TODO: Update these ports to match your robot connections.
# You can typically find these by running `ls /dev/ttyACM*` in your terminal.

# [HARDWARE] Ports for the master device (leader)
LEADER_LEFT_ARM_PORT = "/dev/ttyACM1"
LEADER_RIGHT_ARM_PORT = "/dev/ttyACM0"

# [HARDWARE] Ports for the robot (follower)
FOLLOWER_LEFT_ARM_PORT = "/dev/ttyACM2"
FOLLOWER_RIGHT_ARM_PORT = "/dev/ttyACM3"

# [OPTIONAL] Teleoperation frequency in Hz
TELEOP_FREQUENCY = 50
# ---------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the teleoperation loop."""
    logger.info("Bi-SO100 Teleoperation Script (Live Control)")
    logger.info("=" * 50)

    leader = None
    follower = None

    try:
        # 1. Configure and instantiate the leader (master device)
        logger.info("Configuring leader arms...")
        leader_config = BiSO100LeaderConfig(
            id="bi_so100_leader",  # Used to generate unique IDs for left/right arms
            left_arm_port=LEADER_LEFT_ARM_PORT,
            right_arm_port=LEADER_RIGHT_ARM_PORT,
        )
        leader = BiSO100Leader(leader_config)

        # 2. Configure and instantiate the follower (robot)
        logger.info("Configuring follower arms...")
        follower_config = BiSO100FollowerConfig(
            id="bi_so100_follower",  # Used to generate unique IDs for left/right arms
            left_arm_port=FOLLOWER_LEFT_ARM_PORT,
            right_arm_port=FOLLOWER_RIGHT_ARM_PORT,
        )
        follower = BiSO100Follower(follower_config)

        # 3. Connect to both devices (calibration is skipped)
        logger.info("Connecting to leader...")
        leader.connect(calibrate=False)
        logger.info("Leader connected.")

        logger.info("Connecting to follower...")
        follower.connect(calibrate=False)
        logger.info("Follower connected.")

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

        logger.info("\n>>> Ready for teleoperation. Press Ctrl+C to exit. <<<")

        # 4. Main teleoperation loop
        dt = 1.0 / TELEOP_FREQUENCY
        while True:
            # Get action from the leader device
            action = leader.get_action()

            # Send action to the follower robot
            follower.send_action(action)

            # Maintain control frequency
            time.sleep(dt)

    except KeyboardInterrupt:
        logger.info("Teleoperation stopped by user.")

    except Exception as e:
        logger.error(f"An error occurred during teleoperation: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Please check device connections and configurations.")

    finally:
        # 5. Disconnect all devices
        if leader and leader.is_connected:
            logger.info("Disconnecting leader...")
            leader.disconnect()

        if follower and follower.is_connected:
            logger.info("Disconnecting follower...")
            follower.disconnect()

        logger.info("=" * 50)
        logger.info("Teleoperation script finished.")


if __name__ == "__main__":
    main()
