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

def smpl_motion_fit(data_path: str, smpl_model, mj_model):
    # Load motion capture data
    print("Loading motion capture data...")
    mocap_data = smpl_model.load_motion_data(data_path)
    if mocap_data is None:
        return None
    # helper.show_motions(human_data=mocap_data)

    smpl_local_body_pose_batch = mocap_data['local_body_pose'][:, mocap_data['matching_ids']]  # [N, L, 4, 4]
    batch_size = smpl_local_body_pose_batch.shape[0]
    if batch_size < 3:
        tqdm.write(f"Not enough frames ({batch_size}) for motion fitting, skipping file.")
        return None

    # Motion fitting
    batch_joints = torch.zeros((batch_size, len(mj_model.joint_names)), dtype=torch.float32, device=smpl_model.device)
    batch_joints = torch.nn.Parameter(batch_joints)
    optimizer = torch.optim.AdamW([batch_joints], lr=0.008)

    best_loss = float('inf')
    best_batch_joints = None
    best_batch_link_pose = None

    joint_limits = torch.tensor(mj_model.joint_limits, dtype=torch.float32, device=mj_model.device)
    beta = 0.05
    loss_history = deque(maxlen=10)

    # Forward kinematics under default coordinate system
    # mj_link_pose_batch = mj_model.fk_batch(batch_joints)
    # helper.show_frame(smpl_local_body_pose_batch[0].cpu().numpy(), mj_link_pose_batch[0].detach().cpu().numpy())
    # return

    for iteration in tqdm(range(10000), desc='Fitting motion'):
        mj_link_pose_batch = mj_model.fk_batch(batch_joints)
        # Compute loss
        loss_pos = torch.norm((mj_link_pose_batch[:, mj_model.selected_link_ids, :3, 3] - smpl_local_body_pose_batch[..., :3, 3]), dim=-1).mean()
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
            loss_history.append(loss_pos)
            tqdm.write(f'Iteration {iteration}, Pos Loss: {loss_pos:.4f}, Smooth Loss: {loss_smooth:.4f}, Total Loss: {loss_value:.4f}, Best Loss: {best_loss:.4f} Beta: {beta:.4f}')
        if loss_history and iteration > 50:
            if abs(loss_history[0] - loss_history[-1]) < 1e-3:
                print("Early stopping due to minimal loss improvement.")
                break
    # Calculate the offset z
    root_pose = mocap_data['root_pose']
    global_pose = torch.matmul(root_pose, best_batch_link_pose)
    z_offset = torch.min(global_pose[..., 2, 3], dim=1, keepdim=True)
    root_pose[..., 2, 3] -= z_offset.values
    try:
        results = {
            'fps': mocap_data['fps'],
            'root_pos': root_pose.squeeze().cpu().numpy()[..., :3, 3],
            'root_rot': R.from_matrix(root_pose.squeeze()[..., :3, :3].cpu().numpy()).as_quat(),
            'local_body_pos': best_batch_link_pose.squeeze().cpu().numpy()[..., :3, 3],
            'dof_pos': best_batch_joints.cpu().numpy(),
            'link_body_list': mj_model.link_names,
            'dof_list': mj_model.joint_names,
        }
    except:
        return None
    return results

def smpl_motion_fit_from_file(config_path: str, data_file: str):
    config = OmegaConf.load(config_path)
    device = torch.device(config.device)
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    print("Loading SMPLX model...")
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=device, shape_file=config.motion_retargeting.shape_file, retarget_fps=config.motion_retargeting.fps)
    print("Loading Mujoco model...")
    mj_model = MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values(), device=device)
    results = smpl_motion_fit(data_path=data_file, smpl_model=smpl_model, mj_model=mj_model)
    if results is None:
        return
    save_path = Path('retargeted_data').joinpath(*Path(data_file).with_suffix('.pkl').parts[1:])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f'Saving retargeted mocap data to {str(save_path)}')
    joblib.dump(results, str(save_path))
    # output format for beyond mimic
    root_pos = results['root_pos']
    root_pos[..., 2] += 0.08
    root_rot = results['root_rot']
    dof_pos = results['dof_pos']
    traj = np.concatenate([root_pos, root_rot, dof_pos], axis=-1)
    np.savetxt(str(save_path).replace('.pkl', '.csv'), traj, delimiter=',')




def smpl_motion_fit_from_directory(config_path: str, data_dir: str):
    config = OmegaConf.load(config_path)
    device = torch.device(config.device)
    # Load the SMPLX model and rotate it to the desired orientation xforward-zup
    print("Loading SMPLX model...")
    smpl_model = SmplModel(config.smpl.model_path, config.smpl.model_type, config.smpl.gender, config.smpl.ext, config.smpl_robot_mapping.keys(), device=device, shape_file=config.motion_retargeting.shape_file, retarget_fps=config.motion_retargeting.fps)
    print("Loading Mujoco model...")
    mj_model = MujocoModel(config.mujoco.xml_path, config.mujoco.root, config.smpl_robot_mapping.values(), device=device)

    data_dir = Path(data_dir)
    mocap_files = []
    for mocap_file in data_dir.rglob('*stageii.npz'):
    # for mocap_file in data_dir.rglob('*poses.npz'):
    # for mocap_file in data_dir.rglob('*.pkl'):
        save_path = Path('retargeted_data').joinpath(*mocap_file.with_suffix('.pkl').parts[1:])
        if save_path.exists():
            print(f'Already retargeted, Skipping existing file: {str(save_path)}')
            continue
        mocap_files.append(mocap_file)
    print(f"Found {len(mocap_files)} mocap files in {str(data_dir)}")
    for mocap_file in tqdm(mocap_files, desc="Processing mocap files"):
        results = smpl_motion_fit(data_path=str(mocap_file), smpl_model=smpl_model, mj_model=mj_model)
        if results is None:
            continue
        save_path = Path('retargeted_data').joinpath(*mocap_file.with_suffix('.pkl').parts[1:])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tqdm.write(f'Saving retargeted mocap data to {str(save_path)}')
        joblib.dump(results, str(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("--data", required=True, type=str, help="Path to the human data file")
    args = parser.parse_args()
    if Path(args.data).is_dir():
        smpl_motion_fit_from_directory(args.config, args.data)
    else:
        smpl_motion_fit_from_file(args.config, args.data)