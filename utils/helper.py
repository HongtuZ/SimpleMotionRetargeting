import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

def show_joints(smpl_model, mj_model, betas=None, scale=1.0, selected=False, smpl2robot_rot_mat=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if betas is not None and type(betas) is np.ndarray:
        betas = torch.from_numpy(betas).to(smpl_model.body_model.betas.device)
    smpl_pose = smpl_model.selected_link_pose(betas=betas).detach().cpu().numpy().squeeze() if selected \
        else smpl_model.link_pose(betas=betas).detach().cpu().numpy().squeeze()[:22]

    axis_len = 0.05
    positions = smpl_pose[..., :3, 3]*scale
    Rmat = smpl_pose[..., :3, :3]
    if smpl2robot_rot_mat is not None:
        Rmat = np.matmul(smpl2robot_rot_mat[:, smpl_model.selected_link_ids].squeeze(), Rmat)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='k', s=30)
    # 坐标轴 (X=红, Y=绿, Z=蓝)
    for i in range(positions.shape[0]):
        p = positions[i]
        rot = Rmat[i]
        ax.plot([p[0], p[0] + rot[0, 0]*axis_len], [p[1], p[1] + rot[1, 0]*axis_len], [p[2], p[2] + rot[2, 0]*axis_len], c='r')
        ax.plot([p[0], p[0] + rot[0, 1]*axis_len], [p[1], p[1] + rot[1, 1]*axis_len], [p[2], p[2] + rot[2, 1]*axis_len], c='g')
        ax.plot([p[0], p[0] + rot[0, 2]*axis_len], [p[1], p[1] + rot[1, 2]*axis_len], [p[2], p[2] + rot[2, 2]*axis_len], c='b')

    # 连接父子关节
    smpl_pose = smpl_model.link_pose(betas=betas).detach().cpu().numpy().squeeze()[:22]
    for i, parent in enumerate(smpl_model.link_parent_ids[:smpl_pose.shape[0]]):
        pos = smpl_pose[i, :3, 3]*scale
        ax.text(pos[0], pos[1], pos[2], smpl_model.link_names[i], fontsize=8)
        if parent != -1:
            ppos = smpl_pose[parent, :3, 3]*scale
            ax.plot([pos[0], ppos[0]],
                    [pos[1], ppos[1]],
                    [pos[2], ppos[2]], 'r-')
    mj_pose = mj_model.selected_link_pose if selected else mj_model.link_pose
    mj_xnames = mj_model.selected_link_names if selected else mj_model.link_names
    # 前3维是位置，后4维是四元数(w, x, y, z)
    positions = mj_pose.squeeze()[..., :3, 3]
    Rmat = mj_pose.squeeze()[..., :3, :3]

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', s=30)
    # 坐标轴 (X=红, Y=绿, Z=蓝)
    for i in range(positions.shape[0]):
        p = positions[i]
        rot = Rmat[i]
        name = mj_xnames[i]
        ax.plot([p[0], p[0] + rot[0, 0]*axis_len], [p[1], p[1] + rot[1, 0]*axis_len], [p[2], p[2] + rot[2, 0]*axis_len], c='r')
        ax.plot([p[0], p[0] + rot[0, 1]*axis_len], [p[1], p[1] + rot[1, 1]*axis_len], [p[2], p[2] + rot[2, 1]*axis_len], c='g')
        ax.plot([p[0], p[0] + rot[0, 2]*axis_len], [p[1], p[1] + rot[1, 2]*axis_len], [p[2], p[2] + rot[2, 2]*axis_len], c='b')
        ax.text(p[0], p[1], p[2], name, fontsize=8)
    #显示坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  
    # 设置坐标轴范围
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    plt.show()

def show_motions(human_data=None, robot_data=None):
    """
    叠加显示 human 与 robot 关节/骨骼动画 (overlay 模式固定).
    期望数据字典字段:
    human_data{
        'fps': int,
        "root_pose": (T, 1, 4, 4)  Tensor 或 ndarray,
        "local_body_pose": (T, J, 4, 4)  Tensor 或 ndarray,
        'parent_ids': list 或 ndarray,
        'matching_ids': list 或 ndarray,
    }
    robot_data{
        'fps': int,
        'root_pos': (T, 3)  Tensor 或 ndarray,
        'root_rot': (T, 4)  Tensor 或 ndarray,
        'local_body_pos': (T, J, 3)  Tensor 或 ndarray,
        'dof_pos': (T, D)  Tensor 或 ndarray,
        'link_body_list': (N,)  list of str,
        'dof_list': (J,)  list of str,
    }
    """
    if human_data is None and robot_data is None:
        print("无可视化数据")
        return

    def to_np(x):
        if x is None:
            return None
        if hasattr(x, "cpu"):
            return x.cpu().numpy()
        return np.asarray(x)

    def quat_wxyz_to_mat4(q):
        """
        q: (T,4) quaternion (w,x,y,z)
        returns: (T,4,4) homogeneous matrices
        """
        q = np.asarray(q)
        if q.shape[-1] != 4:
            raise ValueError("Quaternion must have shape (..., 4)")
        # SciPy expects (x, y, z, w)
        q_xyzw = q[..., [1, 2, 3, 0]]
        rot = R.from_quat(q_xyzw)
        rot_mats = rot.as_matrix()  # (T,3,3)
        T = rot_mats.shape[0]
        mats = np.tile(np.eye(4), (T, 1, 1))
        mats[:, :3, :3] = rot_mats
        return mats

    def make_homogeneous_translations(t):
        """
        t: (...,3)
        returns: (...,4,4) homogeneous transforms (identity rotation + translation)
        """
        t = np.asarray(t)
        orig_shape = t.shape[:-1]
        t = t.reshape(-1, 3)
        M = np.tile(np.eye(4), (t.shape[0], 1, 1))
        M[:, :3, 3] = t
        return M.reshape((*orig_shape, 4, 4))

    def prep_human(d):
        if d is None:
            return None
        root_pose = to_np(d.get("root_pose"))
        local_body_pose = to_np(d.get("local_body_pose"))
        if root_pose is None or local_body_pose is None:
            raise ValueError("human_data must contain 'root_pose' and 'local_body_pose'")
        T = local_body_pose.shape[0]
        if root_pose.shape[0] == 1:
            root_pose = np.repeat(root_pose, T, axis=0)
        if root_pose.ndim == 3:
            root_pose = root_pose[:, None, :, :]
        link_pose = root_pose @ local_body_pose

        parents = to_np(d.get("parent_ids"))
        if parents is None:
            parents = np.ones(link_pose.shape[1], dtype=int) * -1
        else:
            parents = np.asarray(parents)[:link_pose.shape[1]]
        sel = to_np(d.get("matching_ids"))
        if sel is None:
            sel = np.arange(link_pose.shape[1], dtype=int)
        fps = int(d.get("fps", 30))
        return dict(link_pose=link_pose, parents=parents, sel_ids=sel, fps=fps)

    def prep_robot(d):
        if d is None:
            return None
        root_pos = to_np(d.get("root_pos"))
        root_rot = to_np(d.get("root_rot"))
        local_body_pos = to_np(d.get("local_body_pos"))

        if root_pos is None or root_rot is None or local_body_pos is None:
            raise ValueError("robot_data must contain root_pos, root_rot, local_body_pos")

        T = root_pos.shape[0]
        root_T = quat_wxyz_to_mat4(root_rot)
        root_T[:, :3, 3] = root_pos

        if local_body_pos.ndim == 2:
            J = local_body_pos.shape[0]
            local_T = np.repeat(make_homogeneous_translations(local_body_pos)[None, ...], T, axis=0)
        else:
            local_T = make_homogeneous_translations(local_body_pos)

        link_pose = root_T[:, None, :, :] @ local_T

        parents = to_np(d.get("parent_ids"))
        if parents is None:
            parents = np.ones(link_pose.shape[1], dtype=int) * -1
        sel = to_np(d.get("matching_ids"))
        if sel is None:
            sel = np.arange(link_pose.shape[1], dtype=int)
        fps = int(d.get("fps", 30))
        return dict(link_pose=link_pose, parents=parents, sel_ids=sel, fps=fps)

    # --- Prepare both
    H = prep_human(human_data) if human_data is not None else None
    Rb = prep_robot(robot_data) if robot_data is not None else None

    # --- Determine frames and fps
    if H and Rb:
        frames = min(len(H["link_pose"]), len(Rb["link_pose"]))
        fps = min(H["fps"], Rb["fps"])
    elif H:
        frames, fps = len(H["link_pose"]), H["fps"]
    else:
        frames, fps = len(Rb["link_pose"]), Rb["fps"]

    # --- Compute bounds
    pts_all = []
    if H: pts_all.append(H["link_pose"].reshape(-1, 4, 4)[:, :3, 3])
    if Rb: pts_all.append(Rb["link_pose"].reshape(-1, 4, 4)[:, :3, 3])
    all_pts = np.concatenate(pts_all, axis=0)
    center = all_pts.mean(0)
    span = (all_pts.max(0) - all_pts.min(0)).max()
    if span == 0: span = 1.0
    half = span * 0.55
    lim_min = center - half
    lim_max = center + half

    # --- Plot setup
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])
    ax.set_title("Overlay Human & Robot")

    # --- Human init
    if H:
        H_sel0 = H["link_pose"][0, H["sel_ids"], :3, 3]
        human_scat = ax.scatter(H_sel0[:,0], H_sel0[:,1], H_sel0[:,2],
                                c="#ff3030", s=40, label="Human")
        human_lines = []
        for i, p in enumerate(H["parents"]):
            if p != -1:
                l, = ax.plot([], [], [], color="#0055ff", lw=2)
                human_lines.append((i, int(p), l))
    else:
        human_scat, human_lines = None, []

    # --- Robot init
    if Rb:
        R_sel0 = Rb["link_pose"][0, Rb["sel_ids"], :3, 3]
        robot_scat = ax.scatter(R_sel0[:,0], R_sel0[:,1], R_sel0[:,2],
                                c="#00c070", s=40, label="Robot")
        robot_lines = []
        for i, p in enumerate(Rb["parents"]):
            if p != -1:
                l, = ax.plot([], [], [], color="#ff9900", lw=2)
                robot_lines.append((i, int(p), l))
    else:
        robot_scat, robot_lines = None, []

    if H or Rb:
        ax.legend(loc="upper right", fontsize=9)
    time_text = fig.text(0.02, 0.95, "", fontsize=9)

    # --- Update function
    def update(f):
        t = f / fps
        time_text.set_text(f"Frame {f}/{frames-1}  Time {t:.2f}s")

        if H:
            sel = H["link_pose"][f, H["sel_ids"], :3, 3]
            human_scat._offsets3d = (sel[:,0], sel[:,1], sel[:,2])
            for i, p, l in human_lines:
                a = H["link_pose"][f, p, :3, 3]; b = H["link_pose"][f, i, :3, 3]
                l.set_data([a[0], b[0]], [a[1], b[1]])
                l.set_3d_properties([a[2], b[2]])

        if Rb:
            sel = Rb["link_pose"][f, Rb["sel_ids"], :3, 3]
            robot_scat._offsets3d = (sel[:,0], sel[:,1], sel[:,2])
            for i, p, l in robot_lines:
                a = Rb["link_pose"][f, p, :3, 3]; b = Rb["link_pose"][f, i, :3, 3]
                l.set_data([a[0], b[0]], [a[1], b[1]])
                l.set_3d_properties([a[2], b[2]])
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
    ani.save('animation.mp4', writer='ffmpeg', fps=fps, dpi=300)
    # plt.tight_layout()
    # plt.show()
    return ani


def show_frame(human_transformations, robot_transformations, ax=None, title="Human vs Robot Points"):
    """
    可视化 4x4 变换矩阵的平移部分（即点）和每个点的 orientation（坐标轴方向），区分人类骨骼点和机器人关节点。
    
    参数:
      human_transformations: (Nh,4,4) 数组或列表，人类骨骼的变换矩阵
      robot_transformations: (Nr,4,4) 数组或列表，机器人关节的变换矩阵
      ax: 可选，传入已有的 3D Axes
      title: 标题
    """
    # 转成 numpy
    human_T = np.asarray(human_transformations)
    robot_T = np.asarray(robot_transformations)
    if human_T.ndim == 2: human_T = human_T[None, ...]
    if robot_T.ndim == 2: robot_T = robot_T[None, ...]

    # 提取位置和旋转
    human_pts = human_T[:, :3, 3]
    human_R = human_T[:, :3, :3]
    robot_pts = robot_T[:, :3, 3]
    robot_R = robot_T[:, :3, :3]

    # 创建画布
    if ax is None:
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # 画散点
    if len(human_pts) > 0:
        ax.scatter(human_pts[:,0], human_pts[:,1], human_pts[:,2],
                   c='tab:blue', marker='o', label='Human')
        # 画 orientation
        axis_len = 0.05
        for p, Rmat in zip(human_pts, human_R):
            ax.plot([p[0], p[0] + Rmat[0,0]*axis_len], [p[1], p[1] + Rmat[1,0]*axis_len], [p[2], p[2] + Rmat[2,0]*axis_len], c='r')
            ax.plot([p[0], p[0] + Rmat[0,1]*axis_len], [p[1], p[1] + Rmat[1,1]*axis_len], [p[2], p[2] + Rmat[2,1]*axis_len], c='g')
            ax.plot([p[0], p[0] + Rmat[0,2]*axis_len], [p[1], p[1] + Rmat[1,2]*axis_len], [p[2], p[2] + Rmat[2,2]*axis_len], c='b')

    if len(robot_pts) > 0:
        ax.scatter(robot_pts[:,0], robot_pts[:,1], robot_pts[:,2],
                   c='tab:orange', marker='^', label='Robot')
        # 画 orientation
        axis_len = 0.05
        for p, Rmat in zip(robot_pts, robot_R):
            ax.plot([p[0], p[0] + Rmat[0,0]*axis_len], [p[1], p[1] + Rmat[1,0]*axis_len], [p[2], p[2] + Rmat[2,0]*axis_len], c='r')
            ax.plot([p[0], p[0] + Rmat[0,1]*axis_len], [p[1], p[1] + Rmat[1,1]*axis_len], [p[2], p[2] + Rmat[2,1]*axis_len], c='g')
            ax.plot([p[0], p[0] + Rmat[0,2]*axis_len], [p[1], p[1] + Rmat[1,2]*axis_len], [p[2], p[2] + Rmat[2,2]*axis_len], c='b')

    # 设置等比例
    all_pts = np.vstack([human_pts, robot_pts]) if (len(human_pts)+len(robot_pts))>0 else np.zeros((1,3))
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (mins + maxs)/2
    span = np.max(maxs - mins)
    span = 1.0 if span < 1e-6 else span
    r = span/2 * 1.2
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    try: ax.set_box_aspect((1,1,1))
    except: pass

    ax.legend()
    plt.tight_layout()
    plt.show()