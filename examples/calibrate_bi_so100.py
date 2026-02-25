#!/usr/bin/env python3
"""
This script calibrates a Bi-SO100 Follower robot.

It uses the dedicated BiSO100Follower class to connect to both arms and
then calls the main calibrate method.

Please ensure that the robot arms have a clear workspace before running.
"""

import logging
import traceback

from lerobot.robots.bi_so100_follower import BiSO100Follower, BiSO100FollowerConfig

# --- Configuration ---
# TODO: Update these ports to match your robot connections.
# You can typically find these by running `ls /dev/ttyACM*` in your terminal.
LEFT_ARM_PORT = "/dev/ttyACM2"
RIGHT_ARM_PORT = "/dev/ttyACM3"
# ---------------------

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the calibration process."""
    logger.info("Bi-SO100 Follower Calibration Script")
    logger.info("=" * 50)

    robot = None
    try:
        # 1. Create a configuration object for the bimanual robot
        logger.info(f"Configuring left arm on '{LEFT_ARM_PORT}' and right arm on '{RIGHT_ARM_PORT}'.")
        config = BiSO100FollowerConfig(
            left_arm_port=LEFT_ARM_PORT,
            right_arm_port=RIGHT_ARM_PORT,
        )

        # 2. Instantiate the bimanual robot
        robot = BiSO100Follower(config)

        # 3. Connect to the robot
        # We set calibrate=False because we will call it explicitly next.
        logger.info("Connecting to both arms...")
        robot.connect(calibrate=False)
        logger.info("Successfully connected to both arms.")

        # 4. Calibrate the robot
        logger.info("Starting calibration for both arms...")
        robot.calibrate()
        logger.info("Calibration completed for both arms.")

    except Exception as e:
        logger.error(f"An error occurred during the calibration process: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Please check the following:")
        logger.warning("1. Are both robots properly connected?")
        logger.warning(f"2. Are the USB ports '{LEFT_ARM_PORT}' and '{RIGHT_ARM_PORT}' correct?")
        logger.warning("3. Do you have sufficient permissions to access USB devices?")
        logger.warning("4. Is the workspace around the robots clear?")

    finally:
        # 5. Disconnect the robot
        if robot and robot.is_connected:
            logger.info("Disconnecting both arms...")
            robot.disconnect()
            logger.info("Disconnected successfully.")

        logger.info("=" * 50)
        logger.info("Bi-SO100 calibration script finished.")


if __name__ == "__main__":
    main()
