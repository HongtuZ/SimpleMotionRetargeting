import mujoco
import torch
import numpy as np
import joblib
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

class MujocoModel:
    def __init__(self, xml_file: str, root: str, foot_link_names = None, selected_link_names = None, T_pose_joints = None, device='cpu'):
        self.device = device
        self.root = root
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.chain = pk.build_chain_from_mjcf(xml_file, root).to(device=device)

        self.joint_names = self.chain.get_joint_parameter_names()
        self.joint_limits = self.chain.get_joint_limits()
        self.joint_qpos_adr = {name: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names}
        self.link_names = self.chain.get_link_names()
        self.link_xpos_adr = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.link_names}

        if root not in self.link_names:
            raise ValueError(f"Root link name {root} not found in the mujoco model. Available links are: {self.link_names}")

        self.selected_link_names = list(selected_link_names) if selected_link_names is not None else self.link_names
        self.selected_link_ids = [self.link_names.index(link) for link in self.selected_link_names]
        self.foot_link_ids = [self.link_names.index(link) for link in foot_link_names] if foot_link_names is not None else []

        if T_pose_joints is not None:
            for joint_name, joint_pos in T_pose_joints.items():
                if joint_name not in self.joint_names:
                    raise ValueError(f"Joint name {joint_name} not found in the mujoco model. Available joints are: {list(self.joint_names.keys())}")
                qpos_adr = self.joint_qpos_adr[joint_name]
                print(f"Setting joint {joint_name} (qpos_adr: {qpos_adr}) to position {joint_pos}")
                self.data.qpos[qpos_adr] = joint_pos/180*np.pi
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_pos(self):
        return self.data.qpos[list(self.joint_qpos_adr.values())]

    @property
    def link_pose(self):
        root_pos = self.data.xpos[self.link_xpos_adr[self.root]]
        link_pos = self.data.xpos[list(self.link_xpos_adr.values())] - root_pos
        link_quat = self.data.xquat[list(self.link_xpos_adr.values())]
        rot_mats = R.from_quat(link_quat, scalar_first=True).as_matrix()
        transformation_matrices = np.zeros((1, len(self.link_names), 4, 4))
        transformation_matrices[..., :3, :3] = rot_mats
        transformation_matrices[..., :3, 3] = link_pos
        transformation_matrices[..., 3, 3] = 1.0
        return transformation_matrices

    @property
    def selected_link_pose(self):
        return self.link_pose[:, self.selected_link_ids]

    def set_pose(self, root_pos=None, root_rot=None, joint_pos=None):
        if root_pos is not None:
            self.data.qpos[:3] = root_pos
        if root_rot is not None:
            self.data.qpos[3:7] = np.roll(root_rot, 1)  # xyzw -> wxyz
        if joint_pos is not None:
            self.data.qpos[list(self.joint_qpos_adr.values())] = joint_pos
        mujoco.mj_forward(self.model, self.data)

    def fk_batch(self, joints):
        frame_ids = self.chain.get_frame_indices(*self.link_names)
        results = self.chain.forward_kinematics(joints, frame_ids)
        matrices = torch.stack([results[name].get_matrix() for name in self.link_names], dim=1)
        root_pos = results[self.root].get_matrix()[:, :3, 3].clone().detach().unsqueeze(1)
        matrices[..., :3, 3] -= root_pos
        return matrices

    def load_motion_data(self, data_path, regenerate=False):
        data = joblib.load(data_path)
        link_pose = data['link_pose']
        joints = data['batch_joints']
        if regenerate:
            # Regenerate the motion data use mujoco forward
            batch_link_pose = []
            for joint_pos in joints:
                self.set_joint_pos(joint_pos)
                batch_link_pose.append(self.selected_link_pose)
            link_pose = np.stack(batch_link_pose)
        return {
            'link_pose': link_pose,
            'fps': data['fps'],
            'joint_pos': joints
        }

    def __str__(self):
        out = '---------------------------------\n'
        out += 'Joints:\n'
        for name in self.joint_names:
            out += f"  {name}\n"

        out += '---------------------------------\n'
        out += 'Links:\n'
        for name in self.link_names:
            out += f"  {name}\n"

        out += '---------------------------------\n'
        out += 'Selected Links:\n'
        for name in self.selected_links:
            out += f"  {name}\n"

        out += '---------------------------------\n'

        return out
