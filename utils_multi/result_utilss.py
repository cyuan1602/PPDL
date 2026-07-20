"""
result_utils_hupr14.py - 修正版（支持真实面积和可见性，对齐“单实例/单GT”COCOeval风格）
修复点：
1) areas / vis 强制对齐到每一帧实例（B*T），避免 AP 输入错配
2) scores 也强制对齐到 (B*T)
3) 增加严格的 shape 断��与更清晰的 warning
4) des_ls（ID）也对齐到 (B*T)，便于排查每帧
5) mpjpe_mm() 支持 [B,J,2] 或 [B,T,J,2]
"""

import os
import torch
import numpy as np

# -------------------- 毫米转换参数 --------------------
# 物理标定：512 像素 = 2000 mm
PIXEL_TO_MM = 2000.0 / 512.0          # 3.90625 mm/像素
IMG_SIZE = 256                         # 图像尺寸（像素）

# COCO 14关键点的sigma值（HuPR数据集）
KPT_OKS_SIGMAS = np.array(
    [1.07, .87, .89, 1.07, .87, .89, 1., 1., .79, .72, .62, .79, .72, .62],
    dtype=np.float32
) / 10.0


# -------------------- 基础指标 --------------------
def mpjpe_mm(output, target):
    """
    计算毫米单位的2D MPJPE
    output, target: [B, J, 2] 或 [B, T, J, 2] 归一化坐标 [0,1]
    返回: [B] 或 [B, T] 的均值（按关节平均）
    """
    dist_norm = torch.norm(output - target, dim=-1)   # [..., J]
    dist_pixel = dist_norm * IMG_SIZE                  # 像素距离
    dist_mm = dist_pixel * PIXEL_TO_MM                 # 毫米距离
    # 对最后一个维度J做平均，保留前面的维度
    return dist_mm.mean(dim=-1).cpu().numpy()


def compute_correlation(x, y):
    """皮尔逊相关系数"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 2:
        return 0.0
    x_mean = np.mean(x_clean)
    y_mean = np.mean(y_clean)
    cov = np.sum((x_clean - x_mean) * (y_clean - y_mean))
    std_x = np.sqrt(np.sum((x_clean - x_mean) ** 2))
    std_y = np.sqrt(np.sum((y_clean - y_mean) ** 2))
    if std_x == 0 or std_y == 0:
        return 0.0
    corr = cov / (std_x * std_y)
    return float(np.clip(corr, -1.0, 1.0))


# -------------------- COCO标准OKS计算 --------------------
def compute_oks(dt_kpts, gt_kpts, area, vis, sigmas=KPT_OKS_SIGMAS):
    """
    计算单个实例的OKS（遵循COCO公式的核心部分）
    dt_kpts: [J, 2] 检测关键点（像素坐标）
    gt_kpts: [J, 2] 真实关键点（像素坐标）
    area: 像素面积（来自bbox的 w*h）
    vis: [J] 可见性标志（1=可见，0=不可见）(若你是0/1/2语义，可在这里改成 vis>0 或 vis==2)
    """
    area = max(float(area), 1e-8)
    vars_ = (sigmas * 2.0) ** 2  # [J]
    visible = np.asarray(vis).astype(bool)
    if not np.any(visible):
        return 0.0

    dt_kpts = np.asarray(dt_kpts, dtype=np.float32)
    gt_kpts = np.asarray(gt_kpts, dtype=np.float32)

    dx = dt_kpts[:, 0] - gt_kpts[:, 0]
    dy = dt_kpts[:, 1] - gt_kpts[:, 1]
    d2 = dx ** 2 + dy ** 2

    e = d2 / (2.0 * area * vars_ + 1e-8)
    e_visible = e[visible]
    oks = float(np.sum(np.exp(-e_visible)) / max(len(e_visible), 1))
    return oks


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _expand_area_to_bt(area, B, T):
    """
    将 area 统一扩展为 [B*T]
    支持：标量、(B,), (B,T), (B*T,)
    """
    area = _to_numpy(area)
    if area is None:
        return None
    area = np.asarray(area)
    if area.ndim == 0:
        return np.full((B * T,), float(area), dtype=np.float32)
    if area.shape == (B,):
        return np.repeat(area.astype(np.float32), T, axis=0)
    if area.shape == (B, T):
        return area.astype(np.float32).reshape(B * T)
    if area.shape == (B * T,):
        return area.astype(np.float32)
    raise ValueError(f"Unsupported area shape: {area.shape}. Expected scalar, (B,), (B,T) or (B*T,).")


def _expand_vis_to_btj(vis, B, T, J):
    """
    将 vis 统一扩展为 [B*T, J]
    支持：(J,), (B,J), (B,T,J), (B*T,J)
    """
    vis = _to_numpy(vis)
    if vis is None:
        return None
    vis = np.asarray(vis)
    if vis.ndim == 1 and vis.shape == (J,):
        return np.tile(vis.astype(np.int32)[None, :], (B * T, 1))
    if vis.ndim == 2 and vis.shape == (B, J):
        return np.repeat(vis.astype(np.int32), T, axis=0)
    if vis.ndim == 3 and vis.shape == (B, T, J):
        return vis.astype(np.int32).reshape(B * T, J)
    if vis.ndim == 2 and vis.shape == (B * T, J):
        return vis.astype(np.int32)
    raise ValueError(f"Unsupported vis shape: {vis.shape}. Expected (J,), (B,J), (B,T,J) or (B*T,J).")


def compute_ap_coco_style(predictions, ground_truth, scores=None, areas=None, vis=None):
    """
    “单实例/单GT”简化版 COCO 风格 OKS-AP：
    - 每一帧当成一个实例
    - 每一帧只有一个预测与一个GT，不做多目标匹配

    predictions, ground_truth: [N, T, J, 2] 归一化坐标
    scores: [N, T] 或 [N*T]
    areas: [N*T]（每帧一个 area）
    vis: [N*T, J]
    """
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    assert predictions.shape == ground_truth.shape, f"predictions shape {predictions.shape} != gt shape {ground_truth.shape}"
    assert predictions.ndim == 4, f"Expected [N,T,J,2], got {predictions.shape}"

    N, T, J, C = predictions.shape
    assert C == 2, f"Last dim must be 2, got {C}"
    n_samples = N * T

    # 展平到 [N*T, J, 2]
    pred_flat = predictions.reshape(n_samples, J, 2)
    gt_flat = ground_truth.reshape(n_samples, J, 2)

    # 转换为像素坐标
    pred_pixel = pred_flat * IMG_SIZE
    gt_pixel = gt_flat * IMG_SIZE

    # areas
    if areas is None:
        print("Warning: areas not provided, estimating from keypoints (may be inaccurate).")
        areas_est = []
        for i in range(n_samples):
            gt_kpts = gt_pixel[i]
            # 注意：这里用 !=0 只是兜底；强烈建议用真实 vis
            visible = np.any(gt_kpts != 0, axis=1)
            if np.sum(visible) > 0:
                visible_kpts = gt_kpts[visible]
                x_min, y_min = visible_kpts.min(axis=0)
                x_max, y_max = visible_kpts.max(axis=0)
                padding = 10
                x_min = max(0.0, x_min - padding)
                y_min = max(0.0, y_min - padding)
                x_max = min(float(IMG_SIZE), x_max + padding)
                y_max = min(float(IMG_SIZE), y_max + padding)
                width = max(x_max - x_min, 10.0)
                height = max(y_max - y_min, 10.0)
                area = max(width * height, 400.0)
            else:
                area = float(IMG_SIZE * IMG_SIZE) / 4.0
            areas_est.append(area)
        areas = np.asarray(areas_est, dtype=np.float32)
    else:
        areas = np.asarray(areas, dtype=np.float32).reshape(-1)
        assert len(areas) == n_samples, f"areas长度必须等于样本数 {n_samples}，但得到 {len(areas)}"

    # vis
    if vis is None:
        print("Warning: vis not provided, assuming all keypoints visible.")
        vis = np.ones((n_samples, J), dtype=np.int32)
    else:
        vis = np.asarray(vis).reshape(n_samples, J)

    # OKS per instance
    oks_all = np.zeros((n_samples,), dtype=np.float32)
    for i in range(n_samples):
        oks_all[i] = compute_oks(pred_pixel[i], gt_pixel[i], areas[i], vis[i])

    # scores
    if scores is None:
        scores = oks_all.copy()
    else:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        assert len(scores) == n_samples, f"scores维度不匹配: {len(scores)} != {n_samples}"

    # 有效样本
    valid_mask = areas > 0
    if not np.any(valid_mask):
        return 0.0, 0.0, 0.0, 0.0

    oks_valid = oks_all[valid_mask]
    scores_valid = scores[valid_mask]
    n_gt = len(oks_valid)

    # COCOeval thresholds
    iou_thrs = np.linspace(0.5, 0.95, 10)
    rec_thrs = np.linspace(0.0, 1.0, 101)

    # sort by score desc
    sort_inds = np.argsort(-scores_valid)
    oks_sorted = oks_valid[sort_inds]

    aps = []
    for thr in iou_thrs:
        tp = (oks_sorted >= thr).astype(np.float32)
        fp = (oks_sorted < thr).astype(np.float32)

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / max(n_gt, 1)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # 101-point interpolation
        precision_interp = np.zeros_like(rec_thrs, dtype=np.float32)
        for i, r in enumerate(rec_thrs):
            m = recall >= r
            if np.any(m):
                precision_interp[i] = np.max(precision[m])

        aps.append(float(np.mean(precision_interp)))

    ap = float(np.mean(aps))
    ap50 = float(aps[0])
    ap75 = float(aps[5])
    mean_oks = float(np.mean(oks_valid))
    return ap, ap50, ap75, mean_oks


# -------------------- 主评测函数 --------------------
def test_keypoint(data_test, device, model, output_temporal=False):
    """
    主测试函数
    为获得准确的AP，des 建议为 dict，至少包含：
      - 'area': 标量/(B,)/(B,T)/(B*T,) 之一
      - 'vis' : (J,)/(B,J)/(B,T,J)/(B*T,J) 之一
    """
    model.eval()

    mpjpe_ls, pcc_ls, pck_ls, des_ls = [], [], [], []
    predictions_norm, ground_truths_norm = [], []
    scores_list = []
    areas_list = []
    vis_list = []

    with torch.no_grad():
        for batch_idx, (x_batch, x_R_batch, y_batch, des) in enumerate(data_test):
            x_batch = x_batch.float().to(device)
            x_R_batch = x_R_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            y_batch_pred = model(x_batch, x_R_batch)

            B, T, J, _ = y_batch.shape

           
            # ---------- 毫米指标 ----------
            scale_mm = torch.tensor(
                [IMG_SIZE * PIXEL_TO_MM, IMG_SIZE * PIXEL_TO_MM],
                device=device,
                dtype=y_batch.dtype
            ).view(1, 1, 1, 2)

            y_batch_mm = y_batch * scale_mm
            y_batch_pred_mm = y_batch_pred * scale_mm

            mpjpe_mm_tensor = torch.sqrt(((y_batch_pred_mm - y_batch_mm) ** 2).sum(dim=-1))  # [B,T,J]

            # per-joint MPJPE（按帧平均） -> [B,J]
            mpjpe_per_joint = mpjpe_mm_tensor.mean(dim=1)
            mpjpe_ls.append(mpjpe_per_joint.cpu().numpy())

            # ---------- PCC ----------
            pred_flat = y_batch_pred.reshape(-1, 2).detach().cpu().numpy()
            gt_flat = y_batch.reshape(-1, 2).detach().cpu().numpy()
            pcc_x = compute_correlation(pred_flat[:, 0], gt_flat[:, 0])
            pcc_y = compute_correlation(pred_flat[:, 1], gt_flat[:, 1])
            pcc_ls.append((pcc_x + pcc_y) / 2.0)

            # ---------- PCK@50mm ----------
            threshold_mm = 50.0
            correct = (mpjpe_mm_tensor < threshold_mm).float()
            pck_ls.append(correct.mean().cpu().numpy())

            # ---------- 存储用于AP计算 ----------
            predictions_norm.append(y_batch_pred.detach().cpu().numpy())  # [B,T,J,2]
            ground_truths_norm.append(y_batch.detach().cpu().numpy())     # [B,T,J,2]

            # score: [B,T] -> flatten later to [B*T]
            mpjpe_per_frame = mpjpe_mm_tensor.mean(dim=-1)                 # [B,T]
            frame_score = 1.0 / (1.0 + mpjpe_per_frame)                    # 越小误差得分越高
            scores_list.append(frame_score.detach().cpu().numpy())

            # ---------- 收集面积和可见性（强制对齐到 B*T） ----------
            if isinstance(des, dict):
                area_bt = _expand_area_to_bt(des.get("area", None), B, T)
                vis_btj = _expand_vis_to_btj(des.get("vis", None), B, T, J)

                if area_bt is not None:
                    areas_list.append(area_bt)      # list of [B*T]
                if vis_btj is not None:
                    vis_list.append(vis_btj)        # list of [B*T, J]
            else:
                if batch_idx == 0:
                    print("Warning: des is not a dict, area and vis will be estimated/assumed (AP may be inaccurate).")

            # ---------- 存储ID（对齐到 B*T） ----------
            # 优先用 des['imageId']（若是单个标量/字符串，则每帧复用；若是(B,)则按样本复用）
            if isinstance(des, dict) and ("imageId" in des):
                image_id = des["imageId"]
                image_id = _to_numpy(image_id)
                if np.asarray(image_id).ndim == 0:
                    ids_bt = [str(image_id)] * (B * T)
                else:
                    image_id = np.asarray(image_id)
                    if image_id.shape == (B,):
                        ids_bt = []
                        for b in range(B):
                            ids_bt.extend([str(image_id[b])] * T)
                    elif image_id.shape == (B * T,):
                        ids_bt = [str(x) for x in image_id.tolist()]
                    else:
                        # 兜底：不保证严格对齐
                        ids_bt = [str(image_id)] * (B * T)
                des_ls.extend(ids_bt)
            else:
                # 兜底：用 batch/frame 索引
                des_ls.extend([f"batch{batch_idx}_b{b}_t{t}" for b in range(B) for t in range(T)])

    # 合并结果
    mpjpe_array = np.concatenate(mpjpe_ls, axis=0) if mpjpe_ls else np.array([])
    pcc_array = np.array(pcc_ls, dtype=np.float32) if pcc_ls else np.array([])
    pck_array = np.array(pck_ls, dtype=np.float32) if pck_ls else np.array([])

    # ---------- 计算AP ----------
    ap = ap50 = ap75 = mean_oks = 0.0
    if len(predictions_norm) > 0:
        predictions_concat = np.concatenate(predictions_norm, axis=0)  # [N,T,J,2]
        ground_truths_concat = np.concatenate(ground_truths_norm, axis=0)
        scores_concat = np.concatenate(scores_list, axis=0)            # [N,T]

        N, T, J, _ = predictions_concat.shape
        n_samples = N * T

        # scores -> [N*T]
        scores_concat = np.asarray(scores_concat, dtype=np.float32).reshape(n_samples)

        areas_concat = np.concatenate(areas_list, axis=0) if areas_list else None  # [N*T]
        vis_concat = np.concatenate(vis_list, axis=0) if vis_list else None       # [N*T,J]

        if areas_concat is not None:
            areas_concat = np.asarray(areas_concat, dtype=np.float32).reshape(-1)
            assert len(areas_concat) == n_samples, f"areas_concat len {len(areas_concat)} != N*T {n_samples}"
        if vis_concat is not None:
            vis_concat = np.asarray(vis_concat).reshape(n_samples, J)

        ap, ap50, ap75, mean_oks = compute_ap_coco_style(
            predictions_concat,
            ground_truths_concat,
            scores=scores_concat,
            areas=areas_concat,
            vis=vis_concat
        )

    mean_mpjpe = float(np.mean(mpjpe_array)) if mpjpe_array.size > 0 else 0.0
    per_joint_mpjpe = mpjpe_array.mean(axis=0) if mpjpe_array.ndim > 1 and mpjpe_array.size > 0 else np.zeros(14)

    result = {
        'MPJPE': mpjpe_array,
        'PCC': pcc_array,
        'PCK': pck_array,
        'AP': ap,
        'AP50': ap50,
        'AP75': ap75,
        'mAP': ap,
        'mean_OKS': mean_oks,
        'mean_MPJPE': mean_mpjpe,
        'per_joint_MPJPE': per_joint_mpjpe,
        'ID': des_ls
    }

    if output_temporal and len(predictions_norm) > 0:
        pred_mm = predictions_concat * IMG_SIZE * PIXEL_TO_MM
        gt_mm = ground_truths_concat * IMG_SIZE * PIXEL_TO_MM
        result['kpts_pred_mm'] = pred_mm
        result['kpts_gt_mm'] = gt_mm

    return result, des_ls


# -------------------- 其他辅助函数 --------------------
def test_ap_600(data_test, device, model, annot_test_path=None):
    """AP测试函数（仅返回AP相关指标）"""
    result, _ = test_keypoint(data_test, device, model, output_temporal=False)
    return {
        'AP': result.get('AP', 0.0),
        'AP50': result.get('AP50', 0.0),
        'AP75': result.get('AP75', 0.0),
        'mAP': result.get('mAP', 0.0)
    }


def test_keypoint_600(data_test, device, model, annot_test_path=None):
    """600帧测试函数（仅返回结果字典）"""
    return test_keypoint(data_test, device, model, output_temporal=False)[0]


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nPCC_loss(output, target, eps=1e-6):
    """负PCC损失"""
    target_mean = target.mean(dim=1, keepdim=True)
    output_mean = output.mean(dim=1, keepdim=True)
    num = ((target - target_mean) * (output - output_mean)).sum(dim=1)
    den = (target - target_mean).norm(dim=1) * (output - output_mean).norm(dim=1) + eps
    return 1 - num / den


def motion_cal(pred, gt, intervals=[2, 4, 6, 8]):
    """运动一致性损失"""
    motion_loss = 0
    for interval in intervals:
        pred_motion = pred[:, interval:, :, :] - pred[:, :-interval, :, :]
        gt_motion = gt[:, interval:, :, :] - gt[:, :-interval, :, :]
        motion_loss += torch.mean((pred_motion - gt_motion) ** 2)
    return motion_loss / len(intervals)