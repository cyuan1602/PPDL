import torch
import torch.nn.functional as F

# ---------- 平滑差分 ----------
def smooth_diff(x, win=5, order=2, dx=1.0):
    """
    x: (B,T,J,C)  任意维度，差分沿 dim=1
    return: 同 shape 的导数
    """
    # 用卷积实现 Savitzky-Golay 滤波差分，kernel 只支持 5 点
    if win == 5 and order == 2:
        kernel = torch.tensor([-2, -1, 0, 1, 2], dtype=x.dtype, device=x.device)
        kernel = kernel / (10 * dx)          # 一阶导系数
        kernel = kernel.view(1, 1, 5, 1, 1)  # (out_c, in_c, kT, kJ, kC)
        # pad 以便输出长度不变
        x_pad = F.pad(x, (0, 0, 0, 0, 2, 2), mode='replicate')
        # 沿 T 维度卷积
        grad = F.conv3d(x_pad.unsqueeze(1), kernel, stride=1).squeeze(1)
        return grad
    else:
        # fallback：中心差分
        return (x[:, 2:] - x[:, :-2]) / (2 * dx)

# ---------- 加速度一致性损失 ----------
def acc_consistency_loss(p_rng, v_dop, limb_idx=None, beta=1.0, win=5):
    """
    p_rng: (B,T,J,3)  range 分支重建的 pose
    v_dop: (B,T,J,3)  doppler 分支重建的 velocity
    limb_idx: list    只对哪些关节计算，None=全部
    beta:  loss 权重
    return: scalar
    """
    if limb_idx is not None:
        p_rng = p_rng[..., limb_idx, :]
        v_dop = v_dop[..., limb_idx, :]

    a_rng = smooth_diff(p_rng, win=win, dx=1.0)          # 二阶导
    a_dop = smooth_diff(v_dop, win=win, dx=1.0)          # 一阶导

    # 对齐长度（差分会缩短）
    minT = min(a_rng.shape[1], a_dop.shape[1])
    a_rng, a_dop = a_rng[:, :minT], a_dop[:, :minT]

    loss = F.mse_loss(a_rng, a_dop)
    return beta * loss