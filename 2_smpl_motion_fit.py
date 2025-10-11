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
from scipy.spatial.transform import Rotation as R

def smpl_motion_fit(config: OmegaConf, data_path: str, device='cpu'):
    device = torch.device(device)
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    print("Loading SMPLX model...")
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=device)
    print("Loading Mujoco model...")
    mj_model = MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values(), device=device)

    # Load motion capture data
    print("Loading motion capture data...")
    mocap_data = smpl_model.load_motion_data(data_path, shape_file_path=config.motion_retargeting.shape_file, fps=config.motion_retargeting.fps, device=device)
    # helper.show_motions(human_data=mocap_data)

    smpl_local_body_pose_batch = mocap_data['local_body_pose'][:, mocap_data['matching_ids']]  # [N, L, 4, 4]
    batch_size = smpl_local_body_pose_batch.shape[0]

    # Motion fitting
    batch_joints = torch.zeros((batch_size, len(mj_model.joint_names)), dtype=torch.float32, device=device)
    batch_joints = torch.nn.Parameter(batch_joints)
    optimizer = torch.optim.AdamW([batch_joints], lr=0.008)

    best_loss = float('inf')
    best_batch_joints = None
    best_batch_link_pose = None

    joint_limits = torch.tensor(mj_model.joint_limits, dtype=torch.float32, device=device)
    beta = 0.8
    loss_history = deque(maxlen=10)

    # Forward kinematics under default coordinate system
    # mj_link_pose_batch = mj_model.fk_batch(batch_joints)
    # helper.show_frame(smpl_local_body_pose_batch[0].cpu().numpy(), mj_link_pose_batch[0].detach().cpu().numpy())
    # return

    for iteration in tqdm(range(1000), desc='Fitting motion'):
        mj_link_pose_batch = mj_model.fk_batch(batch_joints)
        # Compute loss
        loss_pos = torch.norm((mj_link_pose_batch[..., :3, 3] - smpl_local_body_pose_batch[..., :3, 3]), dim=-1).mean()
        # loss_smooth = torch.mean((batch_joints)**2)
        # loss_smooth = torch.mean((batch_joints[1:] - batch_joints[:-1])**2)
        loss_smooth = torch.mean((batch_joints[2:] - 2*batch_joints[1:-1] + batch_joints[:-2])**2)
        total_loss = beta * loss_pos + (1-beta) * loss_smooth
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            batch_joints.clamp_(min=joint_limits[0], max=joint_limits[1])

        loss_value = total_loss.item()
        if loss_value < best_loss:
            best_loss = loss_value
            with torch.no_grad():
                best_batch_joints = batch_joints.clone().detach()
                best_batch_link_pose = mj_link_pose_batch.clone().detach()
        if iteration % 10 == 0:
            loss_history.append(loss_value)
            tqdm.write(f'Iteration {iteration}, Pos Loss: {loss_pos:.4f}, Smooth Loss: {loss_smooth:.4f}, Total Loss: {loss_value:.4f}, Best Loss: {best_loss:.4f}')
        if loss_history and iteration > 50:
            if abs(loss_history[0] - loss_history[-1]) < 1e-5:
                print("Early stopping due to minimal loss improvement.")
                break
    # Save the best qpos_batch
    save_path = Path('retargeted_data') / Path(data_path).with_suffix('.pkl').name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f'Saving retargeted mocap data to {str(save_path)}')
    # joblib.dump({
    #     'mocap_data': mocap_data,
    #     'robot_data': {
    #         'fps': mocap_data['fps'],
    #         'root_pose': mocap_data['root_pose'],
    #         'local_body_pose': best_batch_link_pose,
    #         'dof_pos': best_batch_joints,
    #         'link_body_list': mj_model.selected_link_names,
    #     }
    # }, str(save_path))
    joblib.dump({
        'fps': mocap_data['fps'],
        'root_pose': mocap_data['root_pose'].cpu().numpy(),
        'local_body_pose': best_batch_link_pose.cpu().numpy(),
        'dof_pos': best_batch_joints.cpu().numpy(),
        'link_body_list': mj_model.selected_link_names,
    }, str(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("--data", required=True, type=str, help="Path to the human data file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    smpl_motion_fit(config, data_path=args.data, device='mps')