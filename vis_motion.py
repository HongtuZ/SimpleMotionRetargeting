import argparse
import torch
import numpy as np
import joblib
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import helper
from utils.smpl_util import SmplModel
from utils.mujoco_util import MujocoModel

def vis_motion(config, data_path):
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    # smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=config.device, retarget_fps=config.motion_retargeting.fps)
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=config.device, shape_file=config.motion_retargeting.shape_file, retarget_fps=config.motion_retargeting.fps)
    mj_model= MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values(), config.mujoco.T_pose_joints, device=config.device)
    # human_data = smpl_model.load_motion_data('dataset/SFU/0005/0005_Walking001_stageii.npz')
    robot_data = joblib.load(data_path)
    helper.show_motions(human_data=None, robot_data=robot_data)
    # helper.show_motions(human_data=human_data, robot_data=None)

    # helper.show_motions(motion_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("--data", default=None, type=str, help="Path to the data file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    vis_motion(config, args.data)