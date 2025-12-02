import mujoco
import mujoco.viewer
import time
import argparse
import numpy as np
import joblib
from omegaconf import OmegaConf

def get_nonfree_joint_order(model):
    qpos_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_names.append((model.jnt_qposadr[i], name))
    qpos_names.sort(key=lambda x: x[0])
    return [n for _, n in qpos_names]

def vis_mujoco(xml_path, motion_file_path):
    print('Loading motion data...')
    motion_data = joblib.load(motion_file_path)
    dt = 1.0 / motion_data['fps']

    print("Loading Mujoco model...")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = dt
    mj_data = mujoco.MjData(mj_model)

    dof_names = get_nonfree_joint_order(mj_model)
    print('----------------mujoco dof names----------------')
    for i, name in enumerate(dof_names):
        print(i, name)
    print('----------------motion dof names----------------')
    for i, name in enumerate(motion_data['dof_list']):
        print(i, name)
    print('------------------------------------------------')

    print('Replay motion ...')

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 2  # 相机距离
        # viewer.cam.azimuth = 180  # 水平旋转角度
        # 主循环
        for root_pos, root_rot, dof_pos in zip(motion_data['root_pos'], motion_data['root_rot'], motion_data['dof_pos']):
            root_pos[2] += 0.1
            t1 = time.perf_counter()
            mj_data.qpos[:3] = root_pos
            mj_data.qpos[3:7] = np.roll(root_rot, 1)  # xyzw -> wxyz
            mj_data.qpos[7:] = dof_pos
            mujoco.mj_forward(mj_model, mj_data)
            viewer.cam.lookat = root_pos
            viewer.sync()
            time.sleep(max(0, dt - (time.perf_counter() - t1))) # 控制循环频率


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config path")
    parser.add_argument("--data", type=str, help="Path to the robot motion file", default=None)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    vis_mujoco(config.mujoco.xml_path, args.data)
