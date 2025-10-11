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
    data = joblib.load(data_path)
    human_motion_data = data['motion_data']
    robot_motion_data = data['robot_data']
    helper.show_motions(human_motion_data, robot_motion_data)

    # helper.show_motions(motion_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("--data", default=None, type=str, help="Path to the data file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    vis_motion(config, args.data)