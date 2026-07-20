"""
dataloader_multi.py  —— 完全转发到新的 HUPR 关键点数据集
"""
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
# -------------- 只依赖新的 huprdataset --------------
# 假设 huprdataset.py 与本文件在同一目录
from .HUPRdataset import build_dataloaders as _build_hupr_loaders


# ------------------------------------------------------------------------------
# 唯一被上层脚本调用的函数
# ------------------------------------------------------------------------------
def LoadDataset_Keypoint(args):
    """
    完全复用 huprdataset.py 的逻辑，返回旧接口格式的 4 个对象：
    data_train, data_test, radar_test_video, (len_train, len_test)
    其中 radar_test_video 直接用 test_loader 占位（上层若不需要可忽略）
    """
    h5_dir      = args.result.data_dir          # yaml 里指向 processed_h5 文件夹
    annot_trains =args.result.annot_train
    annot_tests =args.result.annot_test
    batch_size  = args.train.batch_size
    num_workers = args.train.num_workers

    train_loader, test_loader, len_train, len_test = _build_hupr_loaders(
        h5_dir       = h5_dir,
        annot_train = annot_trains,   # 新增
        annot_test = annot_tests, 
        batch_size   = batch_size,
        workers      = num_workers
    )

    # 旧接口要求返回 4 个对象
    return train_loader, test_loader, test_loader, (len_train, len_test)



def Analyze_statistics(args):
    raise NotImplementedError("Analyze_statistics 已废弃，请用 huprdataset.py 自检")

def my_collate_fn(batch):
    """新数据集无需特殊 collate，用默认即可。"""
    return tuple(zip(*batch))