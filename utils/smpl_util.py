import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES, SMPL_JOINT_NAMES, SMPLH_JOINT_NAMES
from scipy.interpolate import interp1d
from pathlib import Path

class SmplModel:
    def __init__(self, model_path, model_type, gender='neutral', ext='npz', rotate_rpy=None, selected_link_names=None, device='cpu'):
        self.model_path = model_path
        self.model_type = model_type
        self.base_rotation = R.from_euler('xyz', [0, 0, 0], degrees=True) if rotate_rpy is None else R.from_euler('xyz', rotate_rpy, degrees=True)
        global_orient = torch.tensor(self.base_rotation.as_rotvec()).float().view(1, -1)
        self.body_model = smplx.create(
            model_path,
            model_type,
            gender=gender,
            use_pca=False,
            ext=ext,
            global_orient=global_orient
        ).to(device)
        self.link_parent_ids = self.body_model.parents
        if model_type == 'smplx':
            self.link_names = JOINT_NAMES[: len(self.link_parent_ids)]
        elif model_type == 'smpl':
            self.link_names = SMPL_JOINT_NAMES[: len(self.link_parent_ids)]
        elif model_type == 'smplh':
            self.link_names = SMPLH_JOINT_NAMES[: len(self.link_parent_ids)]
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types are: smplx, smpl, smplh")

        self.selected_link_names = selected_link_names if selected_link_names is not None else self.link_names
        self.selected_link_ids = []
        for link_name in self.selected_link_names:
            if link_name not in self.link_names:
                raise ValueError(f"Selected link {link_name} not found in link names.")
            self.selected_link_ids.append(self.link_names.index(link_name))
        self.align_rotations = [R.from_quat([1, 0, 0, 0], scalar_first=True)] * len(self.link_names)

    def align_to_mujoco(self, mj_model):
        smpl_quats = self.selected_link_pose()[0, :, 3:]
        mj_quats = mj_model.selected_link_pose[:, 3:]
        for i, link_name in enumerate(self.selected_link_names):
            align_rot = R.from_quat(mj_quats[i], scalar_first=True) * R.from_quat(smpl_quats[i], scalar_first=True).inv()
            self.align_rotations[self.link_names.index(link_name)] = align_rot

    @property
    def num_betas(self):
        return self.body_model.num_betas

    def link_pose(self, **kwargs):
        res = self.body_model(**kwargs, return_full_pose=True)
        full_pose = res.full_pose.reshape(-1, 3)[:len(self.link_names)].detach()
        joints = res.joints[:, :len(self.link_names)]
        joints -= joints[:, 0].clone().detach()
        joint_orientations = []
        quats = []
        for i in range(len(self.link_names)):
            if i == 0:
                rot = R.from_matrix(np.eye(3))
                align_rot = self.align_rotations[i] * rot
            else:
                rot = joint_orientations[self.link_parent_ids[i]] * R.from_rotvec(
                   full_pose[i].cpu().numpy()
                )
                align_rot = self.align_rotations[i] * rot
            joint_orientations.append(rot)
            quats.append(align_rot.as_quat(scalar_first=True))
        quats = torch.tensor(np.stack(quats, axis=0), dtype=torch.float32, device=joints.device).unsqueeze(0)
        pose = torch.cat([joints, quats], dim=-1)
        return pose

    def selected_link_pose(self, **kwargs):
        return self.link_pose(**kwargs)[:, self.selected_link_ids]

    def load_motion_data(self, data_path, shape_file_dir=None, device='cpu'):
        smplx_data = np.load(data_path, allow_pickle=True)
        gender = str(smplx_data["gender"])
        if shape_file_dir is not None:
            shape_file_path = Path(shape_file_dir)/f'best_shape_{gender}.npz'
            print(f"Loading shape file data from {str(shape_file_path)}")
            shape_data = np.load(str(shape_file_path))
            betas = torch.from_numpy(shape_data['shape']).float().to(device)
            scale = torch.from_numpy(shape_data['scale']).float().to(device)
        betas = torch.from_numpy(smplx_data['betas']).float().to(device) if shape_file_path is None else betas
        body_model = smplx.create(
            self.model_path,
            self.model_type,
            gender=gender,
            use_pca=False,
        ).to(device)

        num_frames = smplx_data["pose_body"].shape[0]
        global_orient = torch.tensor(self.base_rotation.as_rotvec(), dtype=torch.float32, device=device).repeat(num_frames, 1)
        smplx_output = body_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=torch.tensor(smplx_data["pose_body"], dtype=torch.float32, device=device), # (N, 63)
            transl=torch.tensor(smplx_data["trans"], dtype=torch.float32, device=device), # (N, 3)
            left_hand_pose=torch.zeros((num_frames, 45), dtype=torch.float32, device=device),
            right_hand_pose=torch.zeros((num_frames, 45), dtype=torch.float32, device=device),
            jaw_pose=torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
            leye_pose=torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
            reye_pose=torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        human_height = 1.66 + 0.1 * betas[0, 0].item()

        full_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)[:, :len(self.link_names)].detach()
        joints = smplx_output.joints[:, :len(self.link_names)].detach()
        joints -= joints[:, :1].clone()
        if shape_file_path is not None:
            joints *= scale
        joint_orientations = []
        quats = []
        for i in range(len(self.link_names)):
            if i == 0:
                rot = R.from_matrix(np.eye(3))
                align_rot = self.align_rotations[i] * rot
                quats.append(align_rot.as_quat(scalar_first=True)[None, ...].repeat(num_frames, axis=0))
            else:
                rot = joint_orientations[self.link_parent_ids[i]] * R.from_rotvec(
                   full_pose[:, i].cpu().numpy()
                )
                align_rot = self.align_rotations[i] * rot
                quats.append(align_rot.as_quat(scalar_first=True))
            joint_orientations.append(rot)
        quats = torch.tensor(np.stack(quats, axis=1), dtype=torch.float32, device=joints.device)
        link_pose = torch.cat([joints, quats], dim=-1)

        return {
            'link_pose': link_pose,
            'selected_link_ids': self.selected_link_ids,
            'link_parent_ids': body_model.parents,
            'fps': smplx_data['mocap_frame_rate']
        }