import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import numpy as np

def show_joints(smpl_model, mj_model, betas=None, scale=1.0, selected=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xpos = smpl_model.selected_link_pose(betas=betas).detach().cpu().numpy().squeeze()*scale if selected \
        else smpl_model.link_pose(betas=betas).detach().cpu().numpy().squeeze()[:22]*scale

    axis_len = 0.05
    # 前3维是位置，后4维是四元数(w, x, y, z)
    positions = xpos[:, :3]
    quats = xpos[:, 3:]  # shape (joint_num, 4)
    Rmat = R.from_quat(quats, scalar_first=True).as_matrix()  # shape (joint_num, 3, 3)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='k', s=30)
    # 坐标轴 (X=红, Y=绿, Z=蓝)
    for i in range(positions.shape[0]):
        p = positions[i]
        rot = Rmat[i]
        ax.plot([p[0], p[0] + rot[0, 0]*axis_len], [p[1], p[1] + rot[1, 0]*axis_len], [p[2], p[2] + rot[2, 0]*axis_len], c='r')
        ax.plot([p[0], p[0] + rot[0, 1]*axis_len], [p[1], p[1] + rot[1, 1]*axis_len], [p[2], p[2] + rot[2, 1]*axis_len], c='g')
        ax.plot([p[0], p[0] + rot[0, 2]*axis_len], [p[1], p[1] + rot[1, 2]*axis_len], [p[2], p[2] + rot[2, 2]*axis_len], c='b')

    # 连接父子关节
    xpos = smpl_model.link_pose(betas=betas).detach().cpu().numpy().squeeze()[:22]*scale
    for i, parent in enumerate(smpl_model.link_parent_ids[:xpos.shape[0]]):
        ax.text(xpos[i, 0], xpos[i, 1], xpos[i, 2], smpl_model.link_names[i], fontsize=8)
        if parent != -1:
            pxpos = xpos[parent]
            ax.plot([xpos[i, 0], pxpos[0]],
                    [xpos[i, 1], pxpos[1]],
                    [xpos[i, 2], pxpos[2]], 'r-')
    mj_xpos = mj_model.selected_link_pose if selected else mj_model.link_pose
    mj_xnames = mj_model.selected_link_names if selected else mj_model.link_names
    # 前3维是位置，后4维是四元数(w, x, y, z)
    positions = mj_xpos[:, :3]
    quats = mj_xpos[:, 3:]  # shape (joint_num, 4)
    Rmat = R.from_quat(quats, scalar_first=True).as_matrix()  # shape (joint_num, 3, 3)

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

def show_motions(human_data, robot_data=None):
    """
    叠加显示 human 与 robot 关节/骨骼动画 (overlay 模式固定).
    期望数据字典字段:
      link_pose: (T, J, 7)  Tensor 或 ndarray
      link_parent_ids: (J,)
      selected_link_ids: (K,)  (可选, 不提供则使用全部前22个)
      fps: int
    """
    if human_data is None and robot_data is None:
        print("无可视化数据")
        return

    def to_np(x):
        if x is None: return None
        if hasattr(x, 'cpu'): return x.cpu().numpy()
        return np.asarray(x)

    def prep(d):
        if d is None: return None
        link_pose = to_np(d['link_pose'])
        link_pose = link_pose[:, :22]  # 截取前22
        if 'link_parent_ids' in d:
            parents = to_np(d['link_parent_ids'])[:link_pose.shape[1]]
        else:
            parents = np.ones(link_pose.shape[1]) * -1
        if 'selected_link_ids' in d:
            sel = to_np(d['selected_link_ids'])
        else:
            sel = np.arange(link_pose.shape[1])
        fps = d.get('fps', 30)
        return dict(link_pose=link_pose, parents=parents, sel_ids=sel, fps=fps)

    H = prep(human_data)
    R = prep(robot_data)

    frames = 0
    fps = 30
    if H and R:
        frames = min(len(H['link_pose']), len(R['link_pose']))
        fps = min(H['fps'], R['fps'])
    elif H:
        frames = len(H['link_pose']); fps = H['fps']
    else:
        frames = len(R['link_pose']); fps = R['fps']

    # 计算显示范围
    pts_all = []
    if H: pts_all.append(H['link_pose'].reshape(-1, 7))
    if R: pts_all.append(R['link_pose'].reshape(-1, 7))
    all_pts = np.concatenate(pts_all, axis=0)[:, :3]
    center = all_pts.mean(0)
    span = (all_pts.max(0) - all_pts.min(0)).max()
    if span == 0: span = 1.0
    half = span * 0.55
    lim_min = center - half
    lim_max = center + half

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])
    ax.set_title("Overlay Human & Robot")

    # 初始化 human
    if H:
        H_sel0 = H['link_pose'][0, H['sel_ids']]
        human_scat = ax.scatter(H_sel0[:,0], H_sel0[:,1], H_sel0[:,2],
                                c='#ff3030', s=40, label='Human Joints')
        human_lines = []
        for i, p in enumerate(H['parents']):
            if p != -1:
                l, = ax.plot([], [], [], color='#0055ff', lw=2)
                human_lines.append((i, p, l))
    else:
        human_scat = None; human_lines = []

    # 初始化 robot
    if R:
        R_sel0 = R['link_pose'][0, R['sel_ids']]
        robot_scat = ax.scatter(R_sel0[:,0], R_sel0[:,1], R_sel0[:,2],
                                c='#00c070', s=40, label='Robot Joints')
        robot_lines = []
        for i, p in enumerate(R['parents']):
            if p != -1:
                l, = ax.plot([], [], [], color='#ff9900', lw=2)
                robot_lines.append((i, p, l))
    else:
        robot_scat = None; robot_lines = []

    if H or R:
        ax.legend(loc='upper right', fontsize=9)

    time_text = fig.text(0.02, 0.95, '', fontsize=9)

    def update(f):
        t = f / fps
        time_text.set_text(f'Frame {f}/{frames-1}  Time {t:.2f}s')

        if H:
            sel = H['link_pose'][f, H['sel_ids']]
            human_scat._offsets3d = (sel[:,0], sel[:,1], sel[:,2])
            for i, p, l in human_lines:
                a = H['link_pose'][f, p]; b = H['link_pose'][f, i]
                l.set_data([a[0], b[0]], [a[1], b[1]])
                l.set_3d_properties([a[2], b[2]])

        if R:
            sel = R['link_pose'][f, R['sel_ids']]
            robot_scat._offsets3d = (sel[:,0], sel[:,1], sel[:,2])
            for i, p, l in robot_lines:
                a = R['link_pose'][f, p]; b = R['link_pose'][f, i]
                l.set_data([a[0], b[0]], [a[1], b[1]])
                l.set_3d_properties([a[2], b[2]])
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
    plt.tight_layout()
    plt.show()
    # save
    # ani.save('animation.mp4', writer='ffmpeg', fps=60)
    return ani