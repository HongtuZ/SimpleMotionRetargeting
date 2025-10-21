import numpy as np
import smplx
import torch
import joblib
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES, SMPLH_JOINT_NAMES
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_matrix
from itertools import permutations, product
from utils import pytorch_util as ptu
from pathlib import Path

class SmplModel:
    def __init__(self, model_path, model_type, gender='neutral', ext='npz', selected_link_names=None, device='cpu', shape_file=None, retarget_fps=None):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist. Please modify the path in config file.")
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        global_orient = torch.from_numpy(R.from_euler('xyz', [np.pi/2, 0, np.pi/2], degrees=False).as_rotvec()).float().to(self.device).view(1,-1)
        self.body_model = smplx.create(
            model_path,
            model_type,
            global_orient=global_orient,
            gender=gender,
            use_pca=False,
            ext=ext,
        ).to(device)
        self.link_parent_ids = self.body_model.parents
        if model_type == 'smplx':
            self.link_names = JOINT_NAMES[: len(self.link_parent_ids)]
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

        self.link_rot_mats = torch.from_numpy(R.from_quat([0.5, -0.5, -0.5, -0.5], scalar_first=True).as_matrix()).float().to(device).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        self.base_rot_transform = torch.eye(4).float().to(self.device).unsqueeze(0).unsqueeze(0)  # (1,1,4,4)
        self.base_rot_transform[..., :3, :3] = torch.from_numpy(R.from_euler('xyz', [np.pi/2, 0, np.pi/2], degrees=False).as_matrix()).float().to(self.device)

        # Load shape data for mocap retargeting if provided
        if shape_file and Path(shape_file).exists():
            shape_data = joblib.load(str(shape_file))
            self.betas = torch.from_numpy(shape_data['betas']).float().to(device)
            self.scale = torch.from_numpy(shape_data['scale']).float().to(device)
            self.smpl2robot_rot_mat = torch.from_numpy(shape_data['smpl2robot_rot_mat']).float().to(device)
        else:
            self.betas = None
            self.scale = None
            self.smpl2robot_rot_mat = None
        self.fps = retarget_fps



    @property
    def num_betas(self):
        return self.body_model.num_betas

    def link_pose(self, **kwargs):
        res = self.body_model(**kwargs, return_full_pose=True)
        full_pose = res.full_pose.reshape(1, -1, 3)[:, :len(self.link_names)]
        rot_mats = axis_angle_to_matrix(full_pose)
        pos_xyz = res.joints[:, :len(self.link_names)] - res.joints[:, 0].clone().detach()
        parents = self.body_model.parents
        tgt_rot_mats = torch.zeros_like(rot_mats)
        for i in range(len(parents)):
            if parents[i] == -1:
                tgt_rot_mats[:, i] = rot_mats[:, i]
                continue
            tgt_rot_mats[:, i] = tgt_rot_mats[:, parents[i]].clone().detach() @ rot_mats[:, i]

        transformation_matrices = torch.zeros((1, len(self.link_names), 4, 4), dtype=torch.float32, device=pos_xyz.device)
        transformation_matrices[..., :3, :3] = tgt_rot_mats @ self.link_rot_mats
        transformation_matrices[..., :3, 3] = pos_xyz
        transformation_matrices[..., 3, 3] = 1.0

        return transformation_matrices

    def selected_link_pose(self, **kwargs):
        return self.link_pose(**kwargs)[:, self.selected_link_ids]

    def load_motion_data(self, data_path):
        # 1. Load SMPL-X data
        try:
            smplx_data = np.load(data_path, allow_pickle=True)
        except:
            print(f"Failed to load SMPL-X data from {data_path}")
            return None
        poses = torch.from_numpy(smplx_data['poses']).float().to(self.device)   # [N, 165]
        trans = torch.from_numpy(smplx_data['trans']).float().to(self.device)   # [N, 3]
        num_frames = poses.shape[0]
        betas = self.betas if self.betas is not None else torch.from_numpy(smplx_data['betas']).float().to(self.device).reshape(1, -1)
        scale = self.scale if self.scale is not None else torch.ones((1,), dtype=torch.float32, device=self.device)
        smpl2robot_rot_mat = self.smpl2robot_rot_mat if self.smpl2robot_rot_mat is not None else torch.eye(3).unsqueeze(0).to(self.device).repeat(len(self.link_names), 1, 1).unsqueeze(0)  # [1, L, 4, 4]

        # 2. Forward pass
        with torch.no_grad():
            chunk_size = 2048
            res = {'full_pose': [], 'joints': []}
            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                chunk_frames = end - start
                chunk_res = self.body_model(
                    betas=betas,
                    global_orient= poses[start:end, :3],
                    transl=trans[start:end],
                    body_pose=poses[start:end, 3:66],
                    left_hand_pose=poses[start:end, 66:111],
                    right_hand_pose=poses[start:end, 111:156],
                    jaw_pose=torch.zeros((chunk_frames, 3), dtype=torch.float32, device=self.device),
                    leye_pose=torch.zeros((chunk_frames, 3), dtype=torch.float32, device=self.device),
                    reye_pose=torch.zeros((chunk_frames, 3), dtype=torch.float32, device=self.device),
                    expression=torch.zeros((chunk_frames, 10), dtype=torch.float32, device=self.device),
                    return_full_pose=True,
                )
                res['full_pose'].append(chunk_res['full_pose'])
                res['joints'].append(chunk_res['joints'])
            for key in res.keys():
                res[key] = torch.cat(res[key], dim=0)

        full_pose = res['full_pose'].reshape(num_frames, -1, 3)[:, :len(self.link_names)]
        rot_mats = axis_angle_to_matrix(full_pose)
        root_pos = res['joints'][:, :1]
        pos_xyz = (res['joints'][:, :len(self.link_names)] - root_pos) * scale + root_pos
        parents = self.body_model.parents
        tgt_rot_mats = torch.zeros_like(rot_mats)
        for i in range(len(parents)):
            if parents[i] == -1:
                tgt_rot_mats[:, i] = rot_mats[:, i]
                continue
            tgt_rot_mats[:, i] = tgt_rot_mats[:, parents[i]] @ rot_mats[:, i]

        transformation_matrices = torch.zeros((num_frames, len(self.link_names), 4, 4), dtype=torch.float32, device=self.device)
        transformation_matrices[..., :3, :3] = tgt_rot_mats @ self.link_rot_mats
        transformation_matrices[..., :3, 3] = pos_xyz
        transformation_matrices[..., 3, 3] = 1.0

        transformation_matrices = self.to_zup(transformation_matrices)
        root_transform = transformation_matrices[:, :1].clone()
        transformation_matrices = self.to_local(transformation_matrices)

        # Sample down to target fps
        data_fps = -1
        for fps_key in ['mocap_frame_rate', 'mocap_framerate']:
            if fps_key in smplx_data:
                data_fps = smplx_data[fps_key]
                break
        if data_fps < 0:
            raise ValueError(f"Frame rate information not found in the SMPL-X data: {data_path}, available keys: {list(smplx_data.keys())}")
        fps = self.fps
        if fps is not None and fps < data_fps:
            transformation_matrices = ptu.interpolate_mocap_se3(
                transformation_matrices, data_fps, fps
            )
            root_transform = ptu.interpolate_mocap_se3(
                root_transform, data_fps, fps
            )
            num_frames = transformation_matrices.shape[0]
        else:
            fps = data_fps

        return {
            'fps': fps,
            "root_pose": root_transform,
            "local_body_pose": transformation_matrices,
            'parent_ids': parents,
            'matching_ids': self.selected_link_ids,
        }

    def to_zup(self, transformation_matrices):
        root_pos = transformation_matrices[0, 0, :3, 3]
        head_pos = transformation_matrices[0, 15, :3, 3]
        up_vec = (head_pos - root_pos).cpu().numpy()
        u = up_vec / np.linalg.norm(up_vec)
        ez = np.array([0., 0., 1.])
        ex = np.array([1., 0., 0.])
        # 24 rotation mat
        best_R = None
        best_primary = -np.inf
        best_secondary = -np.inf
        I = np.eye(3)
        axes = list(permutations(range(3)))          # 6 个排列
        signs = list(product([1,-1], repeat=3))      # 8 种符号
        for p in axes:
            P = I[:, p]                              # 置换矩阵（按列重排）
            for s in signs:
                S = np.diag(s)
                R = P @ S
                if np.isclose(np.linalg.det(R), 1.0):
                    Ru = R @ u
                    primary = float(Ru @ ez)           # 想让 z 分量最大
                    secondary = float((R @ ex) @ ex)   # 平局破：尽量保持 x 不变
                    if (primary > best_primary) or (np.isclose(primary, best_primary) and secondary > best_secondary):
                        best_primary = primary
                        best_secondary = secondary
                        best_R = R
        rotation_mat = torch.eye(4).float().to(self.device).unsqueeze(0).unsqueeze(0)  # (1,1,4,4)
        rotation_mat[..., :3, :3] = torch.from_numpy(best_R).float().to(self.device)
        transformation_matrices = rotation_mat @ transformation_matrices
        return transformation_matrices

    def to_local(self, transformation_matrices):
        root_transform = transformation_matrices[:, :1].clone()
        transformation_matrices = ptu.invert_se3(root_transform) @ transformation_matrices
        return transformation_matrices
