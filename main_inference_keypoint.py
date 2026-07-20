"""
- refer to "https://github.com/chinhsuanwu/mobilevit-pytorch/blob/master/mobilevit.py"
"""
import os
import torch
import hydra
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy import signal  # 用于 Savitzky-Golay 滤波器平滑去噪
from scipy.interpolate import make_interp_spline  # 用于高精三次样条平滑插值
import numpy as np
from utils_multi.result_utils import *
from utils_multi.camera import *  # 此处导入了 show3Dpose, resize_keypoint, camera_to_world 等相机工具
from utils_multi import dataloader_multi

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from model import mobileVit_test23

# 用于全局存储自适应图学习模块截获的注意力矩阵 [B, T, J, J]
captured_graph_attn = {
    'doppler': [],  # 存储 Doppler 分支的图注意力矩阵 [B, 16, 17, 17]
    'range': []     # 存储 Range 分支的图注意力矩阵 [B, 4, 17, 17]
}

# MVDoppler-Pose 标准 17 关节物理名称定义
JOINT_NAMES = [
    'Pelvis_C', 'Hip_R', 'Knee_R', 'Foot_R', 'Hip_L', 'Knee_L', 'Foot_L',
    'Spine_L', 'Spine_U', 'Neck', 'Head', 
    'Shoulder_R', 'Elbow_R', 'Hand_R', 
    'Shoulder_L', 'Elbow_L', 'Hand_L'
]

# 解剖学生理分组索引（躯干与头部、右腿、左腿、右臂、左臂），用于对角化热力图
ANATOMICAL_INDICES = [0, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]

# =========================================================================
# 1. 物理先验双通道特征对比可视化工具类（真实隐层特征、原生时间轴、全英文）
# =========================================================================
class PhysicsVisualizer:
    def __init__(self, model):
        self.model = model
        self.captured = {
            'x_doppler': [],
            'doppler_dist': [],
            'x_range': [],
            'range_vel': []
        }
        self.hook_handle = None
        
    def _hook_fn(self, module, inputs, outputs):
        """Hook 回调，安全捕获输入并重跑物理引擎"""
        x_range, x_doppler = inputs
        
        module.phys_calc.eval()
        with torch.no_grad():
            doppler_dist = module.phys_calc.get_doppler_dist(x_doppler)
            range_vel = module.phys_calc.get_range_vel(x_range)
            
        self.captured['x_doppler'].append(x_doppler.detach().cpu())
        self.captured['doppler_dist'].append(doppler_dist.detach().cpu())
        self.captured['x_range'].append(x_range.detach().cpu())
        self.captured['range_vel'].append(range_vel.detach().cpu())

    def register(self):
        """向 CrossModalityFusion 注册 Hook"""
        target_module = self.model.cross_modal_fusion
        self.hook_handle = target_module.register_forward_hook(self._hook_fn)
        
    def remove(self):
        """释放 Hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()

    def finalize(self):
        """将分批捕获的 Tensor 拼接成完整的 NumPy 数组"""
        if len(self.captured['x_doppler']) > 0:
            for key in self.captured.keys():
                self.captured[key] = torch.cat(self.captured[key], dim=0).numpy()

    def _reduce_via_energy(self, feat_3d, flat_idx):
        """使用 L2 范数（欧氏距离）计算特征图的通道能量强度，并对长时序滤波"""
        feat_2d = feat_3d[flat_idx]  # 提取特定样本对应的特征 [T, C]
        trend = np.linalg.norm(feat_2d, axis=-1)  # [T]
        
        # 对 16 帧的长信号使用 Savitzky-Golay 滤波器平滑去噪，突显物理规律
        if len(trend) == 16:
            trend = signal.savgol_filter(trend, window_length=5, polyorder=2)
            
        trend_min, trend_max = trend.min(), trend.max()
        if trend_max - trend_min > 1e-6:
            trend = (trend - trend_min) / (trend_max - trend_min)
        return trend

    def plot_and_save(self, flat_idx, output_dir, clip_idx):
        """
        绘制学术级【物理先验输入与处理后的特征趋势对比图】：
        - Panel 1: Doppler 支路（原始速度特征输入 x_doppler  vs  积分后的距离先验 doppler_dist, T=16）
        - Panel 2: Range 支路（原始距离特征输入 x_range  vs  微分后的速度先验 range_vel, T=4）
        """
        if not isinstance(self.captured['x_doppler'], np.ndarray):
            print("错误：物理先验特征数据未完全捕获。")
            return

        total_samples = len(self.captured['x_doppler'])
        if flat_idx >= total_samples:
            print(f"警告：扁平索引 {flat_idx} 超出范围。")
            return

        # 1. 直接从模型前向传播中捕获的特征张量中提取 1D 时序能量趋势
        dop_vel = self._reduce_via_energy(self.captured['x_doppler'], flat_idx)    # 原始速度特征输入 [16]
        dop_dst = self._reduce_via_energy(self.captured['doppler_dist'], flat_idx)  # 积分距离特征输出 [16]
        rng_dst = self._reduce_via_energy(self.captured['x_range'], flat_idx)       # 原始距离特征输入 [4]
        rng_vel = self._reduce_via_energy(self.captured['range_vel'], flat_idx)       # 微分速度特征输出 [4]

        # 2. 建立原生时间轴并用三次样条（k=3）进行高精平滑（不作下采样或拉伸填充）
        t_16 = np.arange(16)
        t_4 = np.arange(4)
        t_dense_16 = np.linspace(0, 15, 200)
        t_dense_4 = np.linspace(0, 3, 200)

        # 拟合 Doppler 支路（T=16）
        spl_dop_vel = make_interp_spline(t_16, dop_vel, k=3)
        spl_dop_dst = make_interp_spline(t_16, dop_dst, k=3)
        dop_vel_smooth = spl_dop_vel(t_dense_16)
        dop_dst_smooth = spl_dop_dst(t_dense_16)

        # 拟合 Range 支路（T=4）
        spl_rng_dst = make_interp_spline(t_4, rng_dst, k=3)
        spl_rng_vel = make_interp_spline(t_4, rng_vel, k=3)
        rng_dst_smooth = spl_rng_dst(t_dense_4)
        rng_vel_smooth = spl_rng_vel(t_dense_4)

        # 3. 绘制并列的双子图面板
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # Doppler 分支对齐（左图，原生 16 帧时间轴）
        ax1.plot(t_dense_16, dop_vel_smooth, color='#1f77b4', linestyle='-', linewidth=2.5, label='Input Raw Velocity (x_doppler)')
        ax1.plot(t_dense_16, dop_dst_smooth, color='#ff7f0e', linestyle='--', linewidth=2.5, label='Processed Distance Prior (doppler_dist)')
        ax1.set_xlabel('Temporal Axis (Frames, T=16)', fontsize=10)
        ax1.set_ylabel('Normalized Feature Intensity [0, 1]', fontsize=10)
        ax1.set_title("Doppler Prior: Input vs. Processed Alignment", fontsize=11, fontweight='bold', pad=10)
        ax1.set_xlim(-0.5, 15.5)
        ax1.set_xticks(np.arange(0, 16, 2))
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(loc='best', frameon=False)

        # Range 分支对齐（右图，原生 4 帧时间轴）
        ax2.plot(t_dense_4, rng_dst_smooth, color='#2ca02c', linestyle='-', linewidth=2.5, label='Input Raw Distance (x_range)')
        ax2.plot(t_dense_4, rng_vel_smooth, color='#d62728', linestyle='--', linewidth=2.5, label='Processed Velocity Prior (range_vel)')
        ax2.set_xlabel('Temporal Axis (Frames, T=4)', fontsize=10)
        ax2.set_ylabel('Normalized Feature Intensity [0, 1]', fontsize=10)
        ax2.set_title("Range Prior: Input vs. Processed Alignment", fontsize=11, fontweight='bold', pad=10)
        ax2.set_xlim(-0.2, 3.2)
        ax2.set_xticks(np.arange(0, 4, 1))
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(loc='best', frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"range_physics_clip_{clip_idx}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)


# ==========================================
# 2. 动态图双面板时空邻接图可视化（全英文版）
# ==========================================
def plot_dynamic_graph_alignment(joints_3d, attn_matrix, save_path=None):
    """
    绘制双面板自适应时空图协同图：
    - Panel 1: 注意力热力图，标题为 “Adjacency Matrix”
    - Panel 2: 2D 正投影骨架，标题为 “Projected Skeleton Topology”，边粗细及透明度代表对应关节的关联强度
    """
    # 1. 运行相机到世界坐标系的变换
    joints_3d = np.array(joints_3d, dtype='float32')
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32')
    joints_world = camera_to_world(joints_3d, R=rot, t=0)
    joints_2d = joints_world[:, [0, 2]]  # X 轴与 Z 轴投影

    # MVDoppler-Pose 标准 17 关节物理连接对
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    # 2. 对称化并根据解剖生理学分组重排热力图矩阵
    attn_sym = (attn_matrix + attn_matrix.T) / 2
    attn_grouped = attn_sym[ANATOMICAL_INDICES][:, ANATOMICAL_INDICES]
    joint_names_grouped = [JOINT_NAMES[idx] for idx in ANATOMICAL_INDICES]

    # 创建并列的双面板布局
    fig, (ax_heat, ax_skel) = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={'width_ratios': [1, 0.78]})

    # --- Panel 1: 注意力热力图 ---
    im = ax_heat.imshow(attn_grouped, cmap='magma', aspect='equal', origin='upper')
    ax_heat.set_xticks(range(17))
    ax_heat.set_xticklabels(joint_names_grouped, rotation=90, fontsize=8)
    ax_heat.set_yticks(range(17))
    ax_heat.set_yticklabels(joint_names_grouped, fontsize=8)
    ax_heat.set_title("Adjacency Matrix", fontsize=11, fontweight='bold', pad=10) # 英文标题
    
    # 绘制生理区块分隔线
    dividers = [4.5, 7.5, 10.5, 13.5]
    for div in dividers:
        ax_heat.axhline(div, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_heat.axvline(div, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
    
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    # --- Panel 2: 2D 骨架正投影叠绘 ---
    for i in range(len(I)):
        joint_a = I[i]
        joint_b = J[i]
        x = [joints_2d[joint_a, 0], joints_2d[joint_b, 0]]
        y = [joints_2d[joint_a, 1], joints_2d[joint_b, 1]]
        ax_skel.plot(x, y, color='#e5e5e5', linewidth=1.5, zorder=1)

    # 专门筛选出全图非局部（Non-Local）非物理连接中，自适应学习效果最强的 6 条协同边
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            is_physical = False
            for b_i, b_j in zip(I, J):
                if (i == b_i and j == b_j) or (i == b_j and j == b_i):
                    is_physical = True
                    break
            if not is_physical:
                edges.append((i, j, attn_sym[i, j]))
                
    top_k_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:6]

    for (i, j, weight) in top_k_edges:
        x = [joints_2d[i, 0], joints_2d[j, 0]]
        y = [joints_2d[i, 1], joints_2d[j, 1]]

        lw = np.interp(weight, [0.05, 0.4], [1.5, 5.5])
        alpha = np.interp(weight, [0.05, 0.4], [0.35, 0.95])
        ax_skel.plot(x, y, color='#FF5722', linewidth=lw, alpha=alpha, zorder=2)

    # 绘制深色节点
    ax_skel.scatter(joints_2d[:, 0], joints_2d[:, 1], color='#1f77b4', s=35, zorder=3)

    # 定焦与美化
    xroot, zroot = joints_2d[0, 0], joints_2d[0, 1]
    RADIUS = 0.8
    ax_skel.set_xlim(xroot - RADIUS, xroot + RADIUS)
    ax_skel.set_ylim(zroot - RADIUS, zroot + RADIUS)
    ax_skel.axis('off')
    ax_skel.set_aspect('equal')
    ax_skel.set_title("Projected Skeleton Topology", fontsize=11, fontweight='bold', pad=10) # 英文标题

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ==========================================
# 3. 结合 Hook 截获的 result_load
# ==========================================
def result_load(device, path_model, path_args):
  args = OmegaConf.load(path_args)

  if 'encoder_input' not in args.model.keys():
    args.model.encoder_input = 'multi'
  args.train.traintest_split = 'random'
  data_train, data_test, data_test_video, len_data = dataloader_multi.LoadDataset_Keypoint(args)
  model = mobileVit_test23.main_Net(args).to(device)

  keypoint_startpoint = [data_test_video.__getitem__(i)[2][1] for i in range(len(data_test_video))]
  list_episode = [data_test_video.__getitem__(i)[4] for i in range(len(data_test_video))]

  # Load model
  checkpoint = torch.load(path_model, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  model_size = count_parameters(model)

  # 注册先验时谱平滑 Hook
  visualizer = PhysicsVisualizer(model)
  visualizer.register()

  # =========================================================================
  # 注册捕获图自适应注意力矩阵的 Hook
  # =========================================================================
  def graph_attn_hook(module, inputs, output):
      attn_tensor = output.detach().cpu()
      T = attn_tensor.shape[1]
      if T == 16:
          captured_graph_attn['doppler'].append(attn_tensor)
      elif T == 4:
          captured_graph_attn['range'].append(attn_tensor)

  attn_hook_handle = model.graph_fusion.dynamic_graph.register_forward_hook(graph_attn_hook)
  # =========================================================================

  # 执行推理测试 (Hook 在此时自动采集所有样本的先验轨迹和自适应注意力矩阵)
  test_loss, des_test, (y_target, y_pred) = test_keypoint(data_test, device, model, output_pred=True, output_temporal=True)
  
  # 拼合并卸载 Hook 释放显存
  visualizer.finalize()
  visualizer.remove()
  
  attn_hook_handle.remove()
  if len(captured_graph_attn['doppler']) > 0:
      captured_graph_attn['doppler'] = torch.cat(captured_graph_attn['doppler'], dim=0).numpy()
  if len(captured_graph_attn['range']) > 0:
      captured_graph_attn['range'] = torch.cat(captured_graph_attn['range'], dim=0).numpy()

  print('test_MPJPE: {:.3f}. test_PCC: {:.3f}. test_PCK: {:.3f}%'.
                format(test_loss['MPJPE'].mean(), test_loss['PCC'][:,1:].mean(), test_loss['PCK'].mean()*100))
  
  class_idx = [i for i in range(len(des_test))]
  test_loss_sel = {}
  for key in test_loss.keys():
    test_loss_sel[key] = test_loss[key][class_idx]
  des_test_sel = [des_test[idx] for idx in class_idx]

  return test_loss_sel, des_test_sel, (y_target[class_idx], y_pred[class_idx]), model_size, args, keypoint_startpoint, list_episode, data_test, visualizer


# ==========================================
# 4. 修改后的 main 函数
# ==========================================
@hydra.main(version_base=None, config_path="conf", config_name="config_inference")
def main(args: DictConfig) -> None:
  config = OmegaConf.to_container(args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  path_save = config['path_save']
  FPS = 90
  
  list_keypoint_start = [16, 36, 56, 76, 96, 116, 136, 156, 176, 196]

  # 正确接收 9 个返回值
  test_loss, des_test, (y_target, y_pred), _, _, _, _, data_test, visualizer = result_load(device, config['path_model'], config['path_args'])

  num_sample, num_tstart = y_pred.shape[0], y_pred.shape[1]
 
  # --- 遍历主循环 ---
  for i_sample in tqdm(range(num_sample)):
    name_episode  = des_test[i_sample]['fname'].split('.')[0]
    name_subject  = des_test[i_sample]['subject']
    name_pattern  = des_test[i_sample]['pattern']
  
    if str(args.test_episode) in name_episode:

      # 创建指定的输出目录（仅物理对齐图和拓扑图，去除了无用目录）
      physics_img_dir = f'{path_save}/{name_episode}-{name_subject}-{name_pattern}-Physics_Alignment'  # 物理融合面板目录
      attn_img_dir = f'{path_save}/{name_episode}-{name_subject}-{name_pattern}-Attn_frames'          # 动态图注意力目录

      os.makedirs(physics_img_dir, exist_ok=True)
      os.makedirs(attn_img_dir, exist_ok=True)

      clip_idx = 5
      pose_pred_clip = y_pred[i_sample][clip_idx, :, :, :]    # 预测物理 3D 骨架 [T, 17, 3]

      # =========================================================================
      # 1. 对比物理先验输入特征 (x_doppler / x_range) 与输出特征 (doppler_dist / range_vel)
      # =========================================================================
      flat_idx = i_sample * num_tstart + clip_idx
      visualizer.plot_and_save(
          flat_idx=flat_idx, 
          output_dir=physics_img_dir,
          clip_idx=clip_idx
      )
      # =========================================================================

      num_frames = pose_pred_clip.shape[0]

      # --- 遍历帧，仅生成时空邻接协同图 ---
      for frame_idx in range(num_frames):

          # =========================================================================
          # 2. 绘制自适应时空邻接图（包含关系矩阵与投影骨骼图）
          # =========================================================================
          if isinstance(captured_graph_attn['doppler'], np.ndarray):
              current_attn_matrix = captured_graph_attn['doppler'][flat_idx, frame_idx]  # [17, 17]
              
              img_path_attn = os.path.join(attn_img_dir, f"frame_{frame_idx:03d}.png")
              plot_dynamic_graph_alignment(
                  joints_3d=pose_pred_clip[frame_idx], 
                  attn_matrix=current_attn_matrix, 
                  save_path=img_path_attn
              )
          # =========================================================================

if __name__ == '__main__':
  main()