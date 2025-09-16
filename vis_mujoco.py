import mujoco
import mujoco.viewer
import time
import argparse
import numpy as np
from utils.mujoco_util import MujocoModel
from omegaconf import OmegaConf

def vis_mujoco(config, motion_file_path):
    print("Loading Mujoco model...")
    mj_model = MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values())
    motion_data = mj_model.load_motion_data(motion_file_path)
    dt = 1.0 / motion_data['fps']

    # 在仿真循环前设置
    mj_model.model.opt.gravity = (0,0,0)          # 关闭重力
    mj_model.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # 禁用碰撞
    mj_model.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_PASSIVE  # 禁用被动力

    # 启动交互式可视化窗口
    with mujoco.viewer.launch_passive(mj_model.model, mj_model.data) as viewer:
        viewer.cam.distance = 4  # 相机距离
        viewer.cam.azimuth = 180  # 水平旋转角度

        # 主循环
        for jpos in motion_data['joint_pos']:
            mj_model.set_joint_pos(jpos)
            viewer.sync()
            time.sleep(dt)  # 控制循环频率


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config path")
    parser.add_argument("--robot", type=str, help="Path to the robot motion file", default=None)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    vis_mujoco(config, args.robot)
