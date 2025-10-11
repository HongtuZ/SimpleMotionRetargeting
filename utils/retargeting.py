import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import helper
from utils.smpl_util import SmplModel
from utils.mujoco_util import MujocoModel
from pathlib import Path

class Retargeting:
    def __init__(self, config: OmegaConf, device='cpu'):
        device = torch.device(device)
        # Load the SMPLX model and rotate it to the desired orientation xforward-zup
        self.smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=device)