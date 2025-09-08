import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import helper
from utils.smpl_util import SmplModel
from utils.mujoco_util import MujocoModel
from pathlib import Path

def smpl_shape_fit(config: OmegaConf, device='cpu'):
    device = torch.device(device)
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl.rotation, config.smpl_robot_mapping.keys())
    mj_model= MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values(), config.mujoco.T_pose_joints)
    # helper.show_joints(smpl_model, mj_model, selected=True)

    # Shape fitting
    shape_new = torch.nn.Parameter(torch.zeros([1, smpl_model.num_betas], device=device))
    scale = torch.nn.Parameter(torch.ones([1], device=device))
    optimizer_shape = torch.optim.AdamW([shape_new, scale],lr=0.003)

    best_shape = None
    best_scale = None
    best_loss = np.inf
    for iteration in tqdm(range(10000), desc='Fitting shape and scale'):
        mj_link_pos = torch.from_numpy(mj_model.selected_link_pose[None, ...]).float().to(device)[..., :3]
        smpl_link_pose = smpl_model.selected_link_pose(betas=shape_new)
        smpl_link_pos = smpl_link_pose[..., :3] * scale
        loss = (smpl_link_pos - mj_link_pos).norm(dim=-1).mean()
        if loss < best_loss:
            best_loss = loss
            best_shape = shape_new.clone().detach()
            best_scale = scale.clone().detach()
        if iteration % 100 == 0:
            tqdm.write(f'Iteration {iteration}, Loss: {loss.item()*1000:.4f}, Best Loss: {best_loss.item()*1000:.4f}, Best Scale: {best_scale.item():.4f}')
        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()
    # Save the shape
    mj_xml_path = Path(config.mujoco.xml_path)
    save_path = Path('best_shape') / mj_xml_path.stem / f'best_shape_{config.smpl.gender}.npz'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f'Saving best shape and scale to {save_path}')
    np.savez_compressed(save_path, shape=best_shape.numpy(), scale=best_scale.numpy())
    # Visualize the results
    helper.show_joints(smpl_model, mj_model, betas=best_shape, scale=best_scale.item(), selected=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    smpl_shape_fit(config, device='cpu')