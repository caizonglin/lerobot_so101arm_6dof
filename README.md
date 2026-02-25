# 准备工作

安装pybullet

```plain&#x20;text
pip install pybullet
```

下载代码：

```plain&#x20;text
git clone https://github.com/caizonglin/lerobot_so101arm_6dof.git
```

# 调试机械臂

## 设置舵机ID

设置从臂舵机ID

lerobot-setup-motors --robot.type=so101\_6dof\_follower --robot.port=/dev/ttyACM0

![](<images/Lerobot 6自由度-图片-6.png>)

设置主臂舵机ID

lerobot-setup-motors --teleop.type=so101\_6dof\_leader --teleop.port=/dev/ttyACM0

![](<images/Lerobot 6自由度-图片-4.png>)



## Calibrate

从臂Calibrate

lerobot-calibrate --robot.type=so101\_6dof\_follower --robot.port=/dev/ttyACM0

![](<images/Lerobot 6自由度-图片-7.png>)

主臂Calibrate

lerobot-calibrate --teleop.type=so101\_6dof\_leader --teleop.port=/dev/ttyACM1

![](<images/Lerobot 6自由度-图片-5.png>)



# 摇操作

## 末端摇从操作（逆解）

```plain&#x20;text
python examples/ee_teleop.py --robot.type=so101_6dof_follower --robot.port=/dev/ttyACM0 --robot.use_degrees=true --teleop.type=so101_6dof_leader --teleop.port=/dev/ttyACM1 --teleop.use_degrees=true
```

> 由于平行夹爪和SO101夹爪结构不一致，可能存在开合相反的方向，如果出现这个情况请在ee\_teleop.py中替换合适的代码：
>
> SO101夹爪：follower\_arm\_angles\_deg\["gripper"] = leader\_motor\_angles.get("gripper", 0.0)
>
> 平行夹爪（目前代码）：follower\_arm\_angles\_deg\["gripper"] = 100.0 - leader\_motor\_angles.get("gripper", 0.0)

![](<images/Lerobot 6自由度-图片.png>)



## 关节摇操作（正解SO101夹爪）

```javascript
lerobot-teleoperate \
    --teleop.type=so101_6dof_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.type=so101_6dof_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=false
```



## 带摄像头的关节摇操作（正解SO101夹爪）

查找相机

lerobot-find-cameras

> Note：请根据摄像头实际高度和宽度调整width和height

运行命令

```javascript
lerobot-teleoperate \
    --teleop.type=so101_6dof_leader \
    --teleop.port=/dev/ttyACM1 \
    --robot.type=so101_6dof_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true
```

![](<images/Lerobot 6自由度-图片-1.png>)

![](<images/Lerobot 6自由度-图片-2.png>)



# Record数据

如果已有数据，先删除

```plain&#x20;text
rm -rf /home/zach/.cache/huggingface/lerobot/zonglin11/so101_6dof_pickblock
```

## 关节空间采集数据

```sql
lerobot-record \
    --robot.type=so101_6dof_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.use_degrees=true \
    --robot.cameras='{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_6dof_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.use_degrees=true \
    --dataset.repo_id=zonglin11/so101_6dof_pickblock \
    --dataset.single_task="pickblock" \
    --dataset.num_episodes=20 \
    --display_data=true \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=6

```

![](<images/Lerobot 6自由度-图片-3.png>)



## 末端逆解采集数据

如果文件已存在，先删除

```plain&#x20;text
rm -rf /home/zach/.cache/huggingface/lerobot/zonglin11/so101_ee_6dof_pickblock
```

运行

```sql
python examples/so101_6dof_ee_record.py \
    --follower.type=so101_6dof_follower \
    --follower.port=/dev/ttyACM0 \
    --follower.cameras="{camera_top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 25}}" \
    --leader.type=so101_6dof_leader \
    --leader.port=/dev/ttyACM1 \
    --dataset.repo_id=zonglin11/so101_ee_6dof_pickblock \
    --dataset.single_task="pickblock" \
    --display_data=true \
    --dataset.episode_time_s=25 \
    --dataset.num_episodes=20 \
    --dataset.reset_time_s=10
```

> 参数解释：
>
> * python examples/so101\_6dof\_ee\_record.py: 这是执行您的 Python 脚本的命令。
>
> * \--follower.type=so101\_6dof\_follower: 指定从臂机器人的类型。
>
> * \--follower.port=/dev/ttyACM0: 请替换为您的 从臂机器人实际连接的串行端口。例如 /dev/ttyUSB0 或 /dev/ttyACM0。
>
> * \--follower.cameras="{camera\_top: {type: opencv, index\_or\_path: 0, width: 320, height: 240, fps: 30}}": 配置从臂的摄像头。您可以根据需要修改摄像头类型、索引、分辨率和帧率。
>
>   * index\_or\_path: 0: 通常 0 指的是系统的默认摄像头，您可以根据您的摄像头实际索引进行调整。
>
> * \--leader.type=so101\_6dof\_leader: 指定主臂遥操作器的类型。
>
> * \--leader.port=/dev/ttyACM0: 请替换为您的 主臂遥操作器实际连接的串行端口。
>
> * \--dataset.repo\_id="\<hf\_username>/\<dataset\_repo\_id>": 请替换为您希望将录制的数据集上传到的 Hugging Face Hub 仓库 ID。例如 "your\_username/my\_so101\_ee\_teleop\_dataset"。
>
> * \--dataset.single\_task="Follow the leader's end-effector": 对您录制任务的简短描述。这会保存在数据集中。
>
> * \--display\_data: 这是一个布尔标志。如果您在命令中包含它，它将启用可视化功能（例如 rerun），显示机器人观察和遥操作数据。如果您不希望显示可视化，请从命令中移除此参数。



> 逆解运行流程：
>
> 1. 运行上述命令。
>
> 2. 脚本会连接主臂和从臂机器人。
>
> 3) 脚本会等待您开始遥操作。
>
> 4) 您通过操作 主臂 来控制 从臂 的末端执行器。
>
> 5. 在录制过程中，您可以使用键盘管理录制会话：
>
>    * 右箭头 (`->`): 完成当前 episode 的录制并开始下一个 episode。
>
>    * 左箭头 (`<-`): 放弃当前 episode 的录制，并重新开始录制当前 episode。
>
>    * ESC 键: 停止所有录制，退出脚本，并执行数据集的最终化和上传（如果配置了）。



# 训练-ACT

## 关节空间训练

```sql
lerobot-train \
  --dataset.repo_id=zonglin11/so101_6dof_pickblock \
  --dataset.root=~/data/so101_6dof_pickblock \
  --dataset.streaming=false \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=output_lerobot_train/act2 \
  --job_name=pickblock \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=zonglin11/act-so101_6dof_pickblock \
  --policy.push_to_hub=true \
  --steps=20000 \
  --batch_size=8
```

> &#x20; 参数说明：
>
> * \--policy.type=act：指定使用 ACT 策略模型。
>
> * \--dataset.repo\_id="zonglin11/so101\_6dof\_pickblock"：指向您之前录制的数据集。
>
> * \--policy.output\_dir="outputs/train/act-so101\_6dof\_pickblock"：训练过程中产生的模型检查点和日志会保存到这个本地文件夹。
>
> * \--policy.repo\_id="\<your\_hf\_username>/act-so101\_6dof\_pickblock"：训练结束后，最终的模型会推送到 Hugging Face Hub 上的这个仓库ID。请务必将 \<your\_hf\_username> 替换为您的实际用户名。
>
> * \--policy.device=cuda：指定训练设备为 GPU。如果您的机器没有 GPU，或者 GPU 内存不足，请将其更改为 `--policy.device=cpu`。 请注意，CPU 训练会非常慢。
>
> * \--steps=100000：总共训练 10 万步。这是一个常见的起始值，可能需要根据数据集大小和模型性能进行调整。
>
> * \--batch\_size=8：每次训练使用的样本数量。
>
> * \--policy.push\_to\_hub=true：训练结束后将模型推送到 Hugging Face Hub。

## 末端训练

```sql
lerobot-train \
  --dataset.repo_id=zonglin11/so101_ee_6dof_pickblock \
  --dataset.root=~/data/so101_ee_6dof_pickblock \
  --dataset.streaming=false \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir=output_lerobot_train/act \
  --job_name=pickblock \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=zonglin11/act-so101_ee_6dof_pickblock \
  --policy.push_to_hub=true \
  --steps=20000 \
  --batch_size=16
```

# 推理

## 关节空间推理

```json
lerobot-record \
    --robot.type=so101_6dof_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{ "front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --robot.id=zonglin11_follower_arm \
    --display_data=false \
    --dataset.repo_id=zonglin11/eval_so101_6dof_pickblock \
    --dataset.single_task="PickBlock" \
    --dataset.episode_time_s=1000 \
    --policy.path=zonglin11/act-so101_6dof_pickblock
```



## 末端推理

> Note：逆解的推理要小心，如发现舵机乱转，需要马上拔掉电源，以免损坏舵机

```json
python examples/so101_6dof_ee_inference.py \
    --robot.type=so101_6dof_follower \
    --robot.port="/dev/ttyACM0" \
    --robot.cameras='{ "camera_top": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --policy.path=zonglin11/act-so101_ee_6dof_pickblock \
    --dataset.repo_id=zonglin11/so101_ee_6dof_pickblock \
    --display_data=false
```



写在最后



&#x20;  \* 如果代表 “目标姿态” 的那个坐标系本身就在疯狂抖动，那就说明问题出在AI模型（很可能是单摄像头导致的视觉信息不足）。

&#x20;  \* 如果 “目标姿态” 的运动轨迹是平滑的，但代表机器人实际姿态的坐标系或关节在乱动，那就说明问题出在逆运动学解算或URDF模型不精确。
