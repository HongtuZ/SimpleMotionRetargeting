import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_smplx(npz_path, model_path, device='cpu'):
    # 1. 读取数据
    data = np.load(npz_path, allow_pickle=True)
    poses = torch.from_numpy(data['poses']).float().to(device)   # [N, 165]
    betas = torch.from_numpy(data['betas']).float().to(device)
    trans = torch.from_numpy(data['trans']).float().to(device)
    num_frames = poses.shape[0]

    # 2. 创建 SMPL-X 模型
    model = smplx.create(model_path, model_type='smplx', use_pca=False).to(device)

    # 3. 准备 matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [], c='r', s=20)
    lines = []

    # 限制坐标轴范围（可按需要调整）
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    parents = model.parents
    for _ in range(len(parents)):
        line, = ax.plot([], [], [], c='b')
        lines.append(line)

    # 4. 更新每一帧
    def update(frame):
        body_pose = poses[frame, 3:66].unsqueeze(0)
        global_orient = poses[frame, :3].unsqueeze(0)
        left_hand = poses[frame, 66:111].unsqueeze(0)
        right_hand = poses[frame, 111:156].unsqueeze(0)
        jaw = poses[frame, 156:159].unsqueeze(0)
        leye = poses[frame, 159:162].unsqueeze(0)
        reye = poses[frame, 162:165].unsqueeze(0)

        output = model(
            betas=betas.unsqueeze(0),
            body_pose=body_pose,
            global_orient=global_orient,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            jaw_pose=jaw,
            leye_pose=leye,
            reye_pose=reye,
            transl=trans[frame].unsqueeze(0)
        )

        joints = output.joints[0].detach().cpu().numpy()
        scat._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])

        for i, p in enumerate(parents):
            if p >= 0:
                lines[i].set_data([joints[i, 0], joints[p, 0]],
                                  [joints[i, 1], joints[p, 1]])
                lines[i].set_3d_properties([joints[i, 2], joints[p, 2]])
        return scat, *lines

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    plt.show()


if __name__ == "__main__":
    npz_path = "data/DanceDB/Andria_Satisfied_v2_C3D_stageii.npz"   # 这里换成你的 npz 数据路径
    model_path = "assets/body_models"                # SMPL-X 模型目录，里面要有 SMPLX_NEUTRAL.pkl 等文件
    animate_smplx(npz_path, model_path)
