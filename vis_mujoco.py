import mujoco
import mujoco.viewer
import time
import argparse
import numpy as np
from utils.mujoco_util import MujocoModel
from omegaconf import OmegaConf
import mediapy as media
import joblib


def vis_mujoco(config, motion_file_path):
    print("Loading Mujoco model...")
    mj_model = MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values())
    robot_data = joblib.load(motion_file_path)
    dt = 1.0 / robot_data['fps']

    # 在仿真循环前设置
    mj_model.model.opt.gravity = (0,0,0)          # 关闭重力
    mj_model.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # 禁用碰撞
    mj_model.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_PASSIVE  # 禁用被动力

    # 启动交互式可视化窗口
    # frames = []
    # renderer = mujoco.Renderer(mj_model.model, 480, 640)  # 分辨率

    with mujoco.viewer.launch_passive(mj_model.model, mj_model.data) as viewer:
        viewer.cam.distance = 4  # 相机距离
        viewer.cam.azimuth = 180  # 水平旋转角度

        # 主循环
        for root_pos, root_rot, dof_pos in zip(robot_data['root_pos'], robot_data['root_rot'], robot_data['dof_pos']):
            mj_model.set_pose(root_pos=root_pos, root_rot=root_rot, joint_pos=dof_pos)
            viewer.sync()
            time.sleep(2*dt)  # 控制循环频率
    #         renderer.update_scene(mj_model.data)
    #         frame = renderer.render()
    #         frames.append(frame)
    # media.write_video("output.mp4", frames, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config path")
    parser.add_argument("--data", type=str, help="Path to the robot motion file", default=None)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    vis_mujoco(config, args.data)
