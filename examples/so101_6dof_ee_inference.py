#!/usr/bin/env python

#
# 最终版：这是一个根据您 `lerobot-record` 脚本中的完整逻辑定制的推理脚本。
# 它精确复现了数据处理、模型调用和动作执行的流程。
#
# 使用示例:
# python examples/so101_6dof_ee_inference.py \
#   --policy.path="zonglin11/act-so101_ee_6dof_pickblock" \
#   --dataset.repo_id="zonglin11/so101_ee_6dof_pickblock" \
#   --robot.type=so101_6dof_follower \
#   --robot.port="/dev/ttyACM0"
#


import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import hydra
import torch
import numpy as np
from omegaconf import DictConfig

# 导入所有需要的模块，参照 lerobot-record.py
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.constants import OBS_STR

# 导入您自己的机器人驱动
from lerobot.robots import so101_6dof_follower  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig # noqa: F401


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    repo_id: str
    root: str | None = None


@dataclass
class InferenceConfig:
    robot: RobotConfig
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    control_freq: int = 10
    device: str = "cuda"
    display_data: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@parser.wrap()
def main(cfg: InferenceConfig):
    init_logging()
    logger.info("--- 最终版推理脚本 ---")
    logger.info(pformat(asdict(cfg)))

    if cfg.policy is None:
        raise ValueError("必须通过 --policy.path 提供一个模型路径")

    if cfg.display_data:
        init_rerun(session_name="final_inference")

    # 1. 加载数据集以获取元数据
    log_say("正在加载数据集元数据...")
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
    ds_meta = dataset.meta

    # 2. 初始化机器人
    log_say("正在初始化机器人...")
    cfg.robot.control_mode = "ee"
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # 3. 加载AI策略模型
    log_say(f"正在加载策略模型: {cfg.policy.pretrained_path} ...")
    policy = make_policy(cfg.policy, ds_meta=ds_meta)

    # 4. 创建所有需要的数据处理器
    # 完全仿照 lerobot-record.py 的逻辑
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=ds_meta.stats,
    )
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    log_say("准备就绪！开始推理循环。按 Ctrl+C 退出。")

    try:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        robot_observation_processor.reset()
        robot_action_processor.reset()

        while True:
            start_loop_t = time.perf_counter()

            # 步骤 A: 获取原始观察值
            raw_obs = robot.get_observation()

            # 步骤 B: 处理原始观察值 (仿照 lerobot-record)
            obs_processed = robot_observation_processor(raw_obs)

            # 步骤 C: 构建模型输入帧
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            # 步骤 D: 使用 predict_action 辅助函数进行预测
            # 这个函数内部会正确地调用 preprocessor
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(cfg.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=None, # 推理时通常不需要 task
                robot_type=robot.robot_type,
            )

            # 步骤 E: 将模型输出的Tensor转换为带名字的动作字典
            action_dict = make_robot_action(action_values, dataset.features)

            # 步骤 F: 从动作字典中提取末端位姿和夹爪
            # `action_dict` 本身就是我们需要的扁平字典, e.g., {'x': ..., 'y': ...}
            target_pos = np.array([action_dict['x'], action_dict['y'], action_dict['z']])
            target_rpy = np.array([action_dict['roll'], action_dict['pitch'], action_dict['yaw']])
            gripper_action = action_dict['gripper']

            # 步骤 G: 调用逆运动学解算出臂部的关节角度
            # **核心修改**: 从当前观察值中提取关节角度,作为IK解算器的"种子",以避免解的跳变
            current_joint_angles = {key.removesuffix(".pos"): val for key, val in raw_obs.items() if key.endswith(".pos")}
            follower_arm_angles_deg = robot.kinematics.inverse_kinematics(
                target_pos.tolist(),
                target_rpy.tolist(),
                custom_rest_poses_deg=current_joint_angles
            )

            # 步骤 H: 将夹爪的动作合并进去
            # 注意：这里的 'gripper' 必须和您在 so101_6dof_follower.py 中定义马达时用的名字一致
            # 注意：policy输出的gripper范围可能是0-1，而马达需要0-100。
            # 您遥操作脚本中是 100 - value，这里我们先假设 policy 输出的 0-1 值对应夹爪闭合度，再转为 0-100
            follower_arm_angles_deg["gripper"] = gripper_action * 100

            # 步骤 I: 构建最终发送给 send_action 的指令字典
            robot_action_to_send = {f"{motor}.pos": angle for motor, angle in follower_arm_angles_deg.items()}

            # ----------- 调试打印 -----------
            print(f"末端目标姿态 (x,y,z,r,p,y): {np.round(np.concatenate([target_pos, target_rpy]), 2)} | 最终关节角度: { {k: round(v, 1) for k, v in robot_action_to_send.items()} }", end='\r')
            # --------------------------------

            # 步骤 J: 发送最终的关节角度指令给机器人
            robot.send_action(robot_action_to_send)

            if cfg.display_data:
                log_rerun_data(observation=obs_processed, action=action_values)

            # 维持循环频率
            loop_duration = time.perf_counter() - start_loop_t
            sleep_duration = max(0, (1 / cfg.control_freq) - loop_duration)
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        log_say("捕获到中断信号，正在关闭...")
    finally:
        robot.disconnect()
        log_say("程序已安全退出。")


if __name__ == "__main__":
    main()
