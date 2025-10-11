import torch
from pytorch3d.transforms import so3_log_map, se3_exp_map


def invert_se3(T):
    """
    高效求 SE(3) 逆：
    T: [..., 4, 4]
    返回 T^{-1}: [..., 4, 4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    inv = torch.zeros_like(T)
    inv[..., :3, :3] = Rt
    inv[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    inv[..., 3, 3] = 1.0
    return inv

def _skew(w: torch.Tensor) -> torch.Tensor:
    """w: (...,3) -> (...,3,3) 反对称矩阵"""
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack([
        O,   -wz,  wy,
        wz,   O,  -wx,
       -wy,  wx,   O
    ], dim=-1).reshape(w.shape[:-1] + (3,3))

@torch.no_grad()
def interpolate_mocap_se3(
    data: torch.Tensor,   # (T, J, 4, 4)
    src_fps: float,
    target_fps: float,
) -> torch.Tensor:
    """
    使用 SE(3) 对数/指数插值：T(s)=T0*exp(s*log(T0^{-1}T1))
    注意：不直接调用 pytorch3d.transforms.se3_log_map（它要求平移为0），
    而是用 so3_log_map + 线性方程 V v = t_rel 求 v。
    """
    print(f"Interpolating motion from {src_fps} fps to {target_fps} fps...")
    assert data.ndim == 4 and data.shape[-2:] == (4, 4), "data 形状应为 (T, J, 4, 4)"
    T, J = data.shape[:2]
    if T < 2 or target_fps <= 0:
        return data.clone()

    device, dtype = data.device, data.dtype

    # 时间轴
    t_src = torch.arange(T, device=device, dtype=dtype) / float(src_fps)
    duration = t_src[-1].item()
    dt = 1.0 / float(target_fps)
    T_tgt = int(torch.floor(torch.tensor(duration / dt)).item()) + 1
    t_tgt = torch.arange(T_tgt, device=device, dtype=dtype) * dt

    idx_right = torch.searchsorted(t_src, t_tgt, right=True).clamp(1, T-1)
    idx_left  = idx_right - 1
    tl, tr = t_src[idx_left], t_src[idx_right]
    alpha = ((t_tgt - tl) / (tr - tl).clamp_min(1e-12)).view(T_tgt, 1, 1)  # (T_tgt,1,1)

    # 相邻帧
    T0 = data[idx_left,  ...]   # (T_tgt, J, 4, 4)
    T1 = data[idx_right, ...]

    R0, t0 = T0[..., :3, :3], T0[..., :3, 3]
    R1, t1 = T1[..., :3, :3], T1[..., :3, 3]

    # 相对变换：T_rel = T0^{-1} T1
    R0T = R0.transpose(-1, -2)
    R_rel = R0T @ R1                                   # (T_tgt, J, 3, 3)
    t_rel = (R0T @ (t1 - t0).unsqueeze(-1)).squeeze(-1)  # (T_tgt, J, 3)

    # SO(3) 对数 -> w
    w = so3_log_map(R_rel.reshape(-1, 3, 3)).reshape(T_tgt, J, 3)  # (T_tgt,J,3)

    # 构造 V 并解 V v = t_rel
    theta = w.norm(dim=-1, keepdim=True)  # (T_tgt,J,1)
    K = _skew(w)                          # (T_tgt,J,3,3)

    # 系数 A,B；小角度用泰勒展开稳定
    eps = 1e-8
    theta2 = theta * theta
    A = torch.empty_like(theta)
    B = torch.empty_like(theta)

    small = (theta < 1e-4)
    large = ~small

    # 大角度
    A[large] = (1 - torch.cos(theta[large])) / (theta2[large] + eps)
    B[large] = (theta[large] - torch.sin(theta[large])) / (theta[large]**3 + eps)

    # 小角度泰勒
    # A ≈ 1/2 - θ^2/24 + θ^4/720
    # B ≈ 1/6 - θ^2/120 + θ^4/5040
    th2 = theta2[small]
    th4 = th2 * th2
    A[small] = 0.5 - th2/24.0 + th4/720.0
    B[small] = (1.0/6.0) - th2/120.0 + th4/5040.0

    I = torch.eye(3, dtype=dtype, device=device).view(1,1,3,3).expand_as(K)
    V = I + A.view(T_tgt, J, 1, 1) * K + B.view(T_tgt, J, 1, 1) * (K @ K)   # (T_tgt,J,3,3)

    # 解线性系统求 v（比写解析逆稳健）
    v = torch.linalg.solve(V.reshape(-1,3,3), t_rel.reshape(-1,3)).reshape(T_tgt, J, 3)

    # 组合 twist 并插值
    xi = torch.cat([w, v], dim=-1)                       # (T_tgt,J,6)
    a = alpha.expand(-1, J, -1)                          # (T_tgt,J,1)
    xi_a = a * xi                                        # (T_tgt,J,6)

    # exp 回到 SE(3) 增量
    T_inc = se3_exp_map(xi_a.reshape(-1,6)).reshape(T_tgt, J, 4, 4)  # (T_tgt,J,4,4)

    # 左乘回到绝对位姿
    T_interp = T0 @ T_inc
    T_interp[..., 3, :] = torch.tensor([0,0,0,1], dtype=dtype, device=device)
    print(f"  Interpolated to {T_tgt} frames, duration {duration:.2f} s.")
    return T_interp