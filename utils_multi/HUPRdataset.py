"""
HUPRdataset.py
HuPR 14-joint 2D 数据集加载器
- 193 训练 + 21 测试
- 0-based 索引，无硬编码偏移
- 返回 (dop, rng, kpts, des) 4 元组，与训练/评测循环拆包对齐
- des 包含 'area'（每个时间步的面积）和 'vis'（每个时间步的可见性）
改动要点（相对你上一版）：
1) 不对 h5_list 排序：保持 train_ids/test_ids 的原始顺序，确保与 JSON 标注列表顺序对齐
2) H5 只按窗口读取 (start:start+seg_len)，避免每个 segment 都读完整 600 帧（性能大幅提升）
3) medfilt 可开关：默认关闭（建议先关掉，或你可改为 True）
"""
import os
import json
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal

SEQ_LEN  = 600
NUM_JPTS = 14

# ----------- 基础工具 -----------
def medfilt_kpts(kpts, k=5):
    """中值滤波平滑关键点 (k 必须为奇数)"""
    if k is None or k <= 1:
        return kpts
    if k % 2 == 0:
        raise ValueError("median filter k must be odd, got k=%s" % k)

    kpts_f = kpts.copy()
    # kpts: (T, 14, 2)
    for j in range(NUM_JPTS):
        for c in range(2):
            kpts_f[:, j, c] = signal.medfilt(kpts[:, j, c], k)
    return kpts_f

def read_h5_window(path, start, seg_len):
    """
    只读取时间窗口，避免整段读取导致极慢
    Returns:
      dop: (64, seg_len, 2) float32
      rng: (256, seg_len, 2) float32
    """
    with h5py.File(path, 'r') as hf:
        dop = np.asarray(hf['radar_doppler_time'][:, start:start+seg_len, :], dtype=np.float32)
        rng = np.asarray(hf['radar_range_time'][:, start:start+seg_len, :], dtype=np.float32)
    return dop, rng

# ----------- Dataset -----------
class RadarKeypointDataset(Dataset):
    def __init__(
        self,
        h5_list,
        h5_dir,
        kpts_annot,
        mode='train',
        transform=None,
        seg_len=16,
        stride=1,
        kpt_medfilt_k=None,          # None/0/1 表示不滤波；5 表示用 5 帧中值滤波
        img_size=(256, 256),         # (W,H) 用于关键点归一化
        vis_value=2.0                # 你确认 vis 恒为 2
    ):
        # 关键：保持原始顺序，不排序（与你的标注 JSON 列表顺序一致）
        self.h5_list    = list(h5_list)
        self.h5_dir     = h5_dir
        self.kpts_annot = kpts_annot
        self.mode       = mode
        self.transform  = transform

        self.seg_len    = int(seg_len)
        self.stride     = int(stride)
        self.num_seg    = (SEQ_LEN - self.seg_len) // self.stride + 1

        self.kpt_medfilt_k = kpt_medfilt_k
        self.img_w, self.img_h = float(img_size[0]), float(img_size[1])
        self.vis_value = float(vis_value)

        if self.seg_len <= 0 or self.seg_len > SEQ_LEN:
            raise ValueError(f"seg_len must be in [1,{SEQ_LEN}], got {self.seg_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")

        # 强一致性检查（建议保留）
        if len(self.h5_list) != len(self.kpts_annot):
            raise ValueError(
                f"h5_list length ({len(self.h5_list)}) != kpts_annot length ({len(self.kpts_annot)}). "
                "如果你确信顺序一一对应，请保证两者长度一致。"
            )

    def __len__(self):
        return len(self.h5_list) * self.num_seg

    def __getitem__(self, idx):
        file_idx = idx // self.num_seg
        seg_idx  = idx % self.num_seg
        start    = seg_idx * self.stride

        # 安全检查：防止以后改参数导致越界
        if start + self.seg_len > SEQ_LEN:
            raise IndexError(f"Window out of range: start={start}, seg_len={self.seg_len}, SEQ_LEN={SEQ_LEN}")

        fname = self.h5_list[file_idx]
        h5_path = os.path.join(self.h5_dir, fname)

        # 只读窗口雷达数据：dop (64,T,2), rng (256,T,2)
        dop, rng = read_h5_window(h5_path, start, self.seg_len)

        # 获取该文件对应的标注序列（你说严格按列表顺序）
        seq_annot = self.kpts_annot[file_idx]  # len=600，元素是每帧 dict

        # 遍历窗口内每一帧：提取关键点、面积、可见性
        kpts_list = []
        area_list = []
        vis_list  = []

        for t in range(start, start + self.seg_len):
            frame = seq_annot[t]

            joints = np.array(frame['joints'], dtype=np.float32)  # (14,2)
            kpts_list.append(joints)

            bbox = frame['bbox']  # [x, y, w, h]
            area = float(bbox[2]) * float(bbox[3])
            area_list.append(area)

            # 你明确 vis 恒为 2
            vis = np.full(NUM_JPTS, self.vis_value, dtype=np.float32)
            vis_list.append(vis)

        kpts  = np.stack(kpts_list, axis=0)         # (T,14,2)
        areas = np.asarray(area_list, dtype=np.float32)  # (T,)
        vis   = np.stack(vis_list, axis=0)          # (T,14)

        # 可选中值滤波（默认关闭；你想开就把 kpt_medfilt_k=5）
        kpts = medfilt_kpts(kpts, k=self.kpt_medfilt_k)

        # 归一化关键点
        kpts = kpts / np.array([self.img_w, self.img_h], dtype=np.float32)

        # radar tensor: (2, H, T)
        dop = torch.from_numpy(dop.transpose(2, 0, 1))  # (2,64,T)
        rng = torch.from_numpy(rng.transpose(2, 0, 1))  # (2,256,T)

        kpts  = torch.from_numpy(kpts)   # (T,14,2)
        areas = torch.from_numpy(areas)  # (T,)
        vis   = torch.from_numpy(vis)    # (T,14)

        des = {
            'ID': fname.replace('.h5', ''),
            'area': areas,
            'vis': vis
        }

        # 可选：transform hook（如果你未来要做数据增强）
        if self.transform is not None:
            dop, rng, kpts, des = self.transform(dop, rng, kpts, des)

        # 调试打印（仅第一个样本第一个窗口）
        if start == 0 and file_idx == 0:
            print(f"[debug] file={fname}")
            print(f"[debug] dop shape={tuple(dop.shape)}, rng shape={tuple(rng.shape)}")
            print(f"[debug] kpts norm range: [{kpts.min():.3f}, {kpts.max():.3f}]")
            print(f"[debug] area range: [{areas.min():.3f}, {areas.max():.3f}]")

        return dop, rng, kpts, des

# ----------- Sampler -----------
class CoverageSampler(torch.utils.data.Sampler):
    def __init__(self, n_files, segs_per_file, per_file=38):
        self.n_files  = int(n_files)
        self.segs_p_f = int(segs_per_file)
        self.per_file = int(per_file)

        if self.segs_p_f <= 0:
            raise ValueError("segs_per_file must be > 0")
        if self.per_file <= 0:
            raise ValueError("per_file must be > 0")

    def __iter__(self):
        selected = []
        take = min(self.per_file, self.segs_p_f)
        for fid in range(self.n_files):
            base = fid * self.segs_p_f
            idx  = torch.randperm(self.segs_p_f)[:take] + base
            selected.append(idx)
        return iter(torch.cat(selected).tolist())

    def __len__(self):
        take = min(self.per_file, self.segs_p_f)
        return take * self.n_files

def build_dataloaders(
    h5_dir,
    annot_train,
    annot_test,
    batch_size=4,
    workers=4,
    seg_len=16,
    stride=1,
    kpt_medfilt_k=None,   # 默认不滤波；你要滤波传 5
    pin_memory=True
):
    """
    构建训练和测试 DataLoader
    annot_train, annot_test: JSON 文件路径，每个文件包含对应 split 的标注列表，且顺序严格对齐 train_ids/test_ids。
    """
    train_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 35, 36, 37, 43,
                 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59, 60, 61, 62,
                 63, 64, 66, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83,
                 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
                 31, 32, 53, 54, 55, 67, 68, 69, 70, 72, 85, 86, 100, 157, 158,
                 160, 167, 168, 169, 170, 177, 179, 180, 187, 188, 189, 190,
                 228, 229, 230, 259, 260, 263, 264, 269, 270, 273, 274,
                 102, 103, 104, 105, 106, 107, 108, 109, 110, 119, 121, 122,
                 123, 124, 133, 134, 135, 138, 147, 148, 149, 150, 151, 152,
                 153, 154, 155, 162, 163, 165, 166, 171, 172, 173, 174, 175,
                 176, 182, 183, 184, 185, 186, 191, 192, 193, 195, 196, 198,
                 199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211,
                 212, 213, 215, 216, 223, 224, 225, 226, 231, 232, 233, 234,
                 235, 236, 258, 261, 262, 265, 266, 267, 268, 271, 272, 275,
                 276]

    test_ids = [15, 16, 38, 40, 41, 42,
                17, 39, 244, 245, 246, 249, 250, 251, 252, 253, 254,
                247, 248, 255, 256]

    train_h5 = [f'seq_{idx}.h5' for idx in train_ids]
    test_h5  = [f'seq_{idx}.h5' for idx in test_ids]

    with open(annot_train, 'r') as f:
        train_kpts = json.load(f)
    with open(annot_test, 'r') as f:
        test_kpts = json.load(f)

    assert len(train_h5) == len(train_kpts), f"训练集H5文件数({len(train_h5)})与标注数({len(train_kpts)})不匹配"
    assert len(test_h5) == len(test_kpts), f"测试集H5文件数({len(test_h5)})与标注数({len(test_kpts)})不匹配"

    train_set = RadarKeypointDataset(
        train_h5, h5_dir, train_kpts, mode='train',
        seg_len=seg_len, stride=stride, kpt_medfilt_k=kpt_medfilt_k
    )
    test_set  = RadarKeypointDataset(
        test_h5,  h5_dir, test_kpts,  mode='test',
        seg_len=seg_len, stride=stride, kpt_medfilt_k=kpt_medfilt_k
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=CoverageSampler(len(train_h5), train_set.num_seg, per_file=38),
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, test_loader, len(train_set), len(test_set)