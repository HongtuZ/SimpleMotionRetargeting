import argparse
import torch
import numpy as np
import joblib
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import helper
from utils.smpl_util import SmplModel
from utils.mujoco_util import MujocoModel
from pathlib import Path
from collections import deque

def smpl_shape_fit(config: OmegaConf, device='cpu'):
    device = torch.device(device)
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=device)
    mj_model= MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values(), config.mujoco.T_pose_joints, device=device)
    # helper.show_joints(smpl_model, mj_model, selected=True)

    # Relative link rotation from SMPL to MuJoCo model
    smpl2robot_rot_mat = np.tile(np.eye(3), (len(smpl_model.link_names), 1, 1))[None]
    smpl_rot = smpl_model.selected_link_pose()[..., :3, :3].detach().cpu().numpy()
    mj_rot = mj_model.selected_link_pose[..., :3, :3]
    selected_smpl2robot_rot_mat = np.matmul(np.swapaxes(smpl_rot, -1, -2), mj_rot)
    smpl2robot_rot_mat[:, smpl_model.selected_link_ids] = selected_smpl2robot_rot_mat

    # Shape fitting
    shape_new = torch.nn.Parameter(torch.zeros([1, smpl_model.num_betas], device=device))
    scale = torch.nn.Parameter(torch.ones([1], device=device))
    optimizer_shape = torch.optim.AdamW([shape_new, scale],lr=0.01)

    best_shape = None
    best_scale = None
    best_loss = np.inf
    loss_history = deque(maxlen=100)
    for iteration in tqdm(range(10000), desc='Fitting shape and scale'):
        mj_link_pos = torch.from_numpy(mj_model.selected_link_pose).float().to(device)[..., :3, 3]
        smpl_link_pos = smpl_model.selected_link_pose(betas=shape_new)[..., :3, 3] * scale
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
        loss_history.append(best_loss.item())
        if len(loss_history) == loss_history.maxlen and np.std(loss_history) < 1e-8:
            print("Early stopping due to convergence.")
            break
    # Save the shape
    mj_xml_path = Path(config.mujoco.xml_path)
    save_path = Path('best_shape') / f'{mj_xml_path.stem}_shape.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f'Saving best shape and scale to {save_path}')
    joblib.dump({
        'smpl2robot_rot_mat': smpl2robot_rot_mat,
        'betas': best_shape.detach().cpu().numpy(),
        'scale': best_scale.detach().cpu().numpy(),
    }, str(save_path))
    # Visualize the results
    helper.show_joints(smpl_model, mj_model, betas=best_shape, scale=best_scale.item(), selected=True, smpl2robot_rot_mat=smpl2robot_rot_mat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    smpl_shape_fit(config, device=config.device)