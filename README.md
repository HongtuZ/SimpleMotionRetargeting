# SimpleMotionRetargeting
A simple and clean motion retargeting repo extracted from the H2O motion retargeting module

## Requirements
1. You need to download the SMPL model from [SMPL website](https://smpl.is.tue.mpg.de/) and place it in the `assets/body_models/smplx` directory. The structure should look like this:
    ```
    assets/body_models/smplx
    └── SMPLX_NEUTRAL.npz
    └── SMPLX_MALE.npz
    └── SMPLX_FEMALE.npz
    ```

2. Download the AMASS dataset from [AMASS website](https://amass.is.tue.mpg.de/).

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

    Install `pytorch3d` package:
    ```bash
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
    ```

## Usage
1. Create a config file in the `config` directory. You can refer to `config/orca.yaml` for an example.
    * because the SMPL model is in T-pose, and the root coordinate is y-up/z-forward by default, you need to adjust the 'smpl.rotation' to match your robot root coordinate system.
    * you should specify the joint angles in the 'mujoco.T_pose_joints' field in the config file according to your robot model.

2. Fit the SMPL shape to the robot by running:
    ```bash
        python 1_smpl_shape_fit.py config/your_config.yaml
    ```

3. Retarget motion by running:
    ```bash
        python 2_smpl_motion_fit.py config/your_config.yaml --data path/to/your/amass/motion.npz
    ```

4. Visualize the retargeted motion by running:
    ```bash
        python vis_mujoco.py config/your_config.yaml --data retargeted_data/path/to/your/amass/motion.pkl
    ```
