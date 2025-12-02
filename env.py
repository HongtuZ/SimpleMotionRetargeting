import joblib
import numpy as np
import mujoco, mujoco.viewer

from scipy.spatial.transform import Rotation as R
from scipy import signal

urdf_dof_names = ['lleg_joint1', 'lleg_joint2', 'lleg_joint3', 'lleg_joint4', 'lleg_joint5', 'lleg_joint6', 'rleg_joint1', 'rleg_joint2', 'rleg_joint3', 'rleg_joint4', 'rleg_joint5', 'rleg_joint6', 'waist_yaw_joint', 'head_joint1', 'head_joint2', 'head_joint3', 'larm_joint1', 'larm_joint2', 'larm_joint3', 'larm_joint4', 'larm_joint5', 'larm_joint6', 'rarm_joint1', 'rarm_joint2', 'rarm_joint3', 'rarm_joint4', 'rarm_joint5', 'rarm_joint6']

def smooth(x, box_pts):
    """
    使用移动平均对信号进行平滑处理
    
    参数:
    x: numpy数组, 形状为(n_samples, n_channels)
    box_pts: 整数, 平滑窗口大小
    device: 为了兼容性保留的参数, NumPy版本中不使用
    
    返回:
    smoothed: 平滑后的numpy数组, 形状与x相同
    """
    box = np.ones(box_pts) / box_pts
    num_channels = x.shape[1]
    # 对每个通道分别进行卷积
    smoothed_channels = []
    for i in range(num_channels):
        # 使用scipy的卷积函数，mode='same'保持输出长度不变
        smoothed_channel = signal.convolve(x[:, i], box, mode='same')
        smoothed_channels.append(smoothed_channel)
    # 将结果堆叠回原始形状
    smoothed = np.column_stack(smoothed_channels)
    return smoothed

def get_nonfree_joint_order(model):
    qpos_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_names.append((model.jnt_qposadr[i], name))
    qpos_names.sort(key=lambda x: x[0])
    return [n for _, n in qpos_names]

class OrcaEnv():
    def __init__(self, xml_path, motion_path, dt=0.001, decimation=20, visualization=False, action_names=None) -> None:
        self.xml_path = xml_path
        self.simulation_dt = dt
        self.decimation = decimation

        # Load robot model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.simulation_dt
        self.data = mujoco.MjData(self.model)

        self.torque_lower_limits = np.array([self.model.actuator_ctrlrange[i, 0] for i in range(self.model.nu)]) * 0.9
        self.torque_upper_limits = np.array([self.model.actuator_ctrlrange[i, 1] for i in range(self.model.nu)]) * 0.9

        ctrl_joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0]) for i in range(self.model.nu)]
        dof_names = get_nonfree_joint_order(self.model)
        action_names = dof_names if action_names is None else action_names
        print('-----------mujoco dof names-----------')
        print(dof_names)
        print('-----------policy action names-----------')
        print(action_names)

        self.motion_path = motion_path
        self.motion, self.init_qpos, self.init_qvel = self.load_motion(motion_path)
        self.motion_step = 0


        self.idx_action2mj = [action_names.index(name) for name in dof_names]
        self.idx_mj2action = [dof_names.index(name) for name in action_names]
        self.idx_mj2ctl = [dof_names.index(name) for name in ctrl_joint_names]
        self.idx_mj2urdf = [dof_names.index(name) for name in urdf_dof_names]

        self.viewer = None
        if visualization:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            self.viewer.cam.distance = 2.0

        # [waist_yaw_joint], [larm1-6, rarm1-6], [head1-3], [lleg1-6, rleg1-6]
        kps = [50] + [100]*12 + [0]*3 + [75, 50, 50, 75, 30, 5, 75, 50, 50, 75, 30, 5] 
        kds = [50] + [100]*12 + [0]*3 + [3, 3, 3, 3, 2, 1, 3, 3, 3, 3, 2, 1]
        self.kps = np.array(kps)
        self.kds = np.array(kds)

    def load_motion(self, motion_path):
        motion = joblib.load(motion_path)
        print('-----------ref motion-----------')
        print(motion['dof_list'])
        fps = motion['fps']
        root_pos = motion['root_pos']
        root_rot = motion['root_rot']
        root_ang = R.from_quat(root_rot).as_euler('xyz', degrees=False) # Base orientation
        root_vel = np.zeros_like(root_pos)
        root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
        root_vel[-1, :] = root_vel[-2, :]
        root_vel = smooth(root_vel, 19)
        root_ang_vel = np.zeros_like(root_pos)
        root_ang_vel[:-1, :] = fps * (R.from_quat(root_rot[1:]) * R.from_quat(root_rot[:-1]).inv()).as_rotvec()
        root_ang_vel[-1, :] = root_ang_vel[-2, :]
        root_ang_vel = smooth(root_ang_vel, 19)
        dof_pos = motion['dof_pos']
        dof_vel = np.zeros_like(dof_pos)
        dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
        dof_vel[-1, :] = dof_vel[-2, :]
        dof_vel = smooth(dof_vel, 19)
        ref_motion = np.concatenate([root_pos[:, 2:3], root_ang, root_vel, root_ang_vel[:, 2:3], dof_pos], axis=-1)
        init_qpos = np.concatenate([[0,0,root_pos[0,2]+0.13], root_rot[0][[3,0,1,2]], dof_pos[0]] )
        init_qvel = np.concatenate([root_vel[0], root_ang_vel[0]*0.8, dof_vel[0]*0.8])
        return ref_motion, init_qpos, init_qvel


    def pd_control(self, target_q, q, dq):
        """PD controller calculation"""
        tq = target_q * 0.25
        torque = (tq - q) * self.kps  - dq * self.kds
        torque = np.clip(torque, self.torque_lower_limits, self.torque_upper_limits)
        return torque

    def reset(self):
        self.motion_step = 0
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            root_pos = self.data.xpos[self.model.body("base_link").id]
            self.viewer.cam.lookat = root_pos
            self.viewer.sync()
        root_ang = R.from_quat(self.data.sensor('orientation').data.astype(np.float32)[[1,2,3,0]]).as_euler('xyz', degrees=False) # Base orientation
        root_ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        joints_pos = self.data.qpos[7:][self.idx_mj2action] # Joint positions
        joints_vel = self.data.qvel[6:][self.idx_mj2action] # Joint vel
        return {
            'ref_motion': np.zeros((36,)),
            'root_ang': np.zeros((3,)),
            'root_ang_vel': np.zeros((3,)),
            'joints_pos': np.zeros((28,)),
            'joints_vel': np.zeros((28,)),
        }

    def step(self, action):
        # Get control commands
        target_dof_pos = action[self.idx_action2mj]
        for _ in range(self.decimation):
            # Apply PD control
            tau = self.pd_control(
                target_dof_pos,
                self.data.qpos[7:],
                self.data.qvel[6:]
            )
            self.data.ctrl[:] = tau[self.idx_mj2ctl]
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            root_pos = self.data.xpos[self.model.body("base_link").id]
            self.viewer.cam.lookat = root_pos
            self.viewer.sync()

        root_ang = R.from_quat(self.data.sensor('orientation').data.astype(np.float32)[[1,2,3,0]]).as_euler('xyz', degrees=False) # Base orientation
        root_ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)[[0,2,1]]
        joints_pos = self.data.qpos[7:][self.idx_mj2action] # Joint positions
        joints_vel = self.data.qvel[6:][self.idx_mj2action] # Joint vel
        self.motion_step += 1
        self.motion_step %= self.motion.shape[0]
        return {
            'ref_motion': self.motion[self.motion_step],
            'root_ang': root_ang,
            'root_ang_vel': root_ang_vel,
            'joints_pos': joints_pos,
            'joints_vel': joints_vel,
        }

    def close(self):
        if self.viewer is not None:
            self.viewer.close()