import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import helper
from utils.smpl_util import SmplModel
from utils.mujoco_util import MujocoModel

def vis_motion(config, human_motion_path, robot_motion_path):
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    print("Loading SMPLX model...")
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl.rotation, config.smpl_robot_mapping.keys())
    print("Loading Mujoco model...")
    mj_model = MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values())

    human_motion_data, robot_motion_data = None, None
    if human_motion_path is not None:
        # Load motion capture data
        print("Loading human motion capture data...")
        human_motion_data = smpl_model.load_motion_data(human_motion_path, config.motion_retargeting.shape_file_path)
    if robot_motion_path is not None:
        robot_motion_data = mj_model.load_motion_data(robot_motion_path, regenerate=True)

    helper.show_motions(human_motion_data, robot_motion_data)


    # helper.show_motions(motion_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("--human", default=None, type=str, help="Path to the data file")
    parser.add_argument("--robot", default=None, type=str, help="Path to the data file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    vis_motion(config, args.human, args.robot)