"""
- refer to "https://github.com/chinhsuanwu/mobilevit-pytorch/blob/master/mobilevit.py"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xformers.components.attention.core import scaled_dot_product_attention

import numpy as np


# class CrossModalityFusion(nn.Module):
#     """双向跨模态融合（无采样，仅操作C维度）"""   
#     def __init__(self, dim, depth=2):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 ModalityCrossAttention(dim),
#                 ModalityCrossAttention(dim)
#             ]))

#     def forward(self, x_range, x_doppler):
#         """
#         输入：x_range[4,C], x_doppler[16,C] → 输出：[4,C], [16,C]
#         """
#         for range_attn, doppler_attn in self.layers:
#             x_range = range_attn(x_range, x_doppler)
#             x_doppler = doppler_attn(x_doppler, x_range)
#         return x_range, x_doppler



class DynamicGraphLearning(nn.Module):
    """
    动态图结构学习模块 - 高效版本
    """
    def __init__(self, num_joints=17, hidden_dim=16):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 分离的查询和键投影
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 相对位置编码
        self.pos_embedding = nn.Embedding(2 * num_joints - 1, 1)
        
        # 边特征MLP
        edge_input_dim = 1  # 只用位置编码
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化相对位置索引
        self._init_relative_positions()
        
    def _init_relative_positions(self):
        positions = torch.arange(self.num_joints)
        rel_pos = positions.view(-1, 1) - positions.view(1, -1)  # [J,J]
        # 将相对位置映射到 [0, 2*num_joints-2] 范围内
        rel_pos = rel_pos + self.num_joints - 1
        self.register_buffer('relative_pos_indices', rel_pos.long())
        
    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: 输入特征 [B,T,J,H]
        """
        B, T, J, H = x.shape
        
        # 1. 节点特征投影
        query = self.query_proj(x)  # [B,T,J,H]
        key = self.key_proj(x)      # [B,T,J,H]
        
        # 2. 重塑为批量计算
        query_flat = query.reshape(B * T, J, H)  # [B*T, J, H]
        key_flat = key.reshape(B * T, J, H)      # [B*T, J, H]
        
        # 3. 计算基础注意力分数
        attention_scores = torch.bmm(query_flat, key_flat.transpose(1, 2))  # [B*T, J, J]
        
        # 4. 添加相对位置偏置
        pos_bias = self.pos_embedding(self.relative_pos_indices)  # [J,J,1]
        pos_bias = pos_bias.squeeze(-1).unsqueeze(0).expand(B * T, J, J)  # [B*T, J, J]
        
        # 5. 应用位置MLP得到更复杂的位置偏置
        rel_pos_features = self.relative_pos_indices.float().unsqueeze(-1)  # [J,J,1]
        pos_mlp_bias = self.edge_mlp(rel_pos_features).squeeze(-1)  # [J,J]
        pos_mlp_bias = pos_mlp_bias.unsqueeze(0).expand(B * T, J, J)  # [B*T, J, J]
        
        # 6. 合并所有偏置
        attention_scores = attention_scores + pos_bias + pos_mlp_bias
        
        # 7. 应用softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B*T, J, J]
        attention_weights = attention_weights.reshape(B, T, J, J)
        
        if return_attention_weights:
            return attention_weights, attention_scores.reshape(B, T, J, J)
        return attention_weights
class GatelessCoherentFusion(nn.Module):
    def __init__(self, dim, dropout=0.1, lambda_val=0.1):
        super().__init__()
        # 1. 对应公式 (12) 中的独立变换矩阵 Wc 和 Wd (使用不带偏置的线性层)
        self.W_c = nn.Linear(dim, dim, bias=False)
        self.W_d = nn.Linear(dim, dim, bias=False)
        
        # 2. 对应提取交互表示 I 的协同投影
        self.synergetic_proj = nn.Linear(dim * 2, dim)
        
        # 3. 融合超参数 lambda
        self.lambda_val = lambda_val

    def forward(self, spatial_feat, temporal_feat):
        # 对应公式 (12) 计算 C 和 D
        # (根据公式，Wc 和 Wd 分别用于投影 spatial 和 temporal 特征)
        constructive = self.W_c(spatial_feat) + self.W_c(temporal_feat)
        destructive = self.W_d(spatial_feat) - self.W_d(temporal_feat)
        
        # 对应计算交互表示 I
        combined = torch.cat([spatial_feat, temporal_feat], dim=-1)
        interacted = self.synergetic_proj(combined)
        
        # 对应公式 (13) 计算能量偏差 E
        # 这里的 C 是特征通道数，即 spatial_feat.shape[-1]
        energy_bias = torch.norm(destructive, p=2, dim=-1, keepdim=True) / (spatial_feat.shape[-1] ** 0.5)
        
        # 对应计算 Z = C + I - \lambda * (D \odot E)
        fused = constructive + interacted - self.lambda_val * (destructive * energy_bias)
        
        return fused
class CrossAttentionFusion(nn.Module):
    """使用 Cross‑Attention 融合两个特征序列（spatial ↔ temporal）"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, y):
        """
        x: [B, T, J, H]  (spatial)
        y: [B, T, J, H]  (temporal)
        return: [B, T, J, H]  (与 x 形状相同)
        """
        B, T, J, H = x.shape
        # 合并 batch 和 time 维度，将关节视为序列长度
        x_flat = x.reshape(B * T, J, H)
        y_flat = y.reshape(B * T, J, H)
        # Cross‑Attention: query = x, key/value = y
        attn_out, _ = self.attn(x_flat, y_flat, y_flat)
        # 残差 + LayerNorm
        out = self.norm(x_flat + attn_out)
        # FFN + 残差
        out = self.norm2(out + self.ffn(out))
        # 恢复形状
        return out.reshape(B, T, J, H)
class SpatioTemporalGraphFusion(nn.Module):
    def __init__(self, num_joints=17, feature_dim=3, hidden_dim=16, 
                 doppler_frames=16, range_frames=4, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.doppler_T = doppler_frames
        self.range_T = range_frames

        # 共享特征投影
        self.shared_proj = nn.Linear(feature_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 改进的动态图学习
        self.dynamic_graph = DynamicGraphLearning(num_joints, hidden_dim)
        self.cat_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        # 空间图传播
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)

        # 时序图传播 - 直接前后帧融合
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # 输入是两帧特征的拼接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # 时序融合权重
        self.temporal_weights = nn.Parameter(torch.ones(2) * 0.5)  # 前后帧权重

        
        self.coherent_fusion = GatelessCoherentFusion(hidden_dim, dropout)
        # self.cross_attn_fusion = CrossAttentionFusion(hidden_dim, num_heads=4, dropout=dropout)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        # 输出投影
        self.doppler_proj_out = nn.Linear(hidden_dim, feature_dim)
        self.range_proj_out = nn.Linear(hidden_dim, feature_dim)

        # 门控残差连接
        self.doppler_gate = nn.Parameter(torch.tensor(0.7))
        self.range_gate = nn.Parameter(torch.tensor(0.7))

        self._init_parameters()

    def _init_parameters(self):
        # 初始化动态图参数
        for m in self.dynamic_graph.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # 初始化其他参数
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in self.dynamic_graph.modules():
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # 初始化位置编码
        nn.init.uniform_(self.dynamic_graph.pos_embedding.weight, -0.1, 0.1)
        
        nn.init.constant_(self.doppler_gate, 0.7)
        nn.init.constant_(self.range_gate, 0.7)

    def _direct_temporal_fusion(self, x):
        """
        直接前后帧时序融合
        Args:
            x: 输入特征 [B,T,J,H]
        Returns:
            时序融合后的特征 [B,T,J,H]
        """
        B, T, J, H = x.shape
        
        if T == 1:
            # 单帧情况下直接返回
            return x
            
        # 初始化输出
        temporal_output = torch.zeros_like(x)
        
        # 处理第一帧（只与后一帧融合）
        if T > 1:
            # 第一帧与第二帧融合
            frame0 = x[:, 0]  # [B,J,H]
            frame1 = x[:, 1]  # [B,J,H]
            combined_01 = torch.cat([frame0, frame1], dim=-1)  # [B,J,2*H]
            fused_0 = self.temporal_mlp(combined_01)  # [B,J,H]
            temporal_output[:, 0] = fused_0
        
        # 处理中间帧（与前后的帧融合）
        for t in range(1, T-1):
            prev_frame = x[:, t-1]  # [B,J,H]
            curr_frame = x[:, t]    # [B,J,H]
            next_frame = x[:, t+1]  # [B,J,H]
            
            # 与前帧融合
            combined_prev = torch.cat([curr_frame, prev_frame], dim=-1)
            fused_prev = self.temporal_mlp(combined_prev)
            
            # 与后帧融合
            combined_next = torch.cat([curr_frame, next_frame], dim=-1)
            fused_next = self.temporal_mlp(combined_next)
            
            # 加权融合
            weights = F.softmax(self.temporal_weights, dim=0)
            fused_curr = weights[0] * fused_prev + weights[1] * fused_next
            temporal_output[:, t] = fused_curr
        
        # 处理最后一帧（只与前一帧融合）
        if T > 1:
            last_frame = x[:, -1]      # [B,J,H]
            prev_last_frame = x[:, -2] # [B,J,H]
            combined_last = torch.cat([last_frame, prev_last_frame], dim=-1)
            fused_last = self.temporal_mlp(combined_last)
            temporal_output[:, -1] = fused_last
        
        # 残差连接 + 层归一化
        temporal_output = temporal_output + x
        temporal_output = self.temporal_norm(temporal_output)
        
        return temporal_output

    def _enhanced_fusion(self, spatial_feat, temporal_feat):
        # """增强的时空融合"""
        # combined = torch.cat([spatial_feat, temporal_feat], dim=-1)  # [B, T, J, 2*H]
        # return self.cat_proj(combined)
        return self.coherent_fusion(spatial_feat, temporal_feat)
    def spatial_graph_propagation(self, x, return_attention=False):
        """增强的空间图传播"""
        B, T, J, H = x.shape
        
        # 动态生成关系矩阵
        if return_attention:
            dynamic_R, attention_weights = self.dynamic_graph(x, return_attention_weights=True)
        else:
            dynamic_R = self.dynamic_graph(x)  # [B,T,J,J]
        
        # 图传播
        spatial_msg = torch.matmul(dynamic_R, x)  # [B,T,J,H]
        spatial_feat = self.spatial_mlp(spatial_msg)
        
        # 残差连接 + 层归一化
        spatial_feat = spatial_feat + x
        spatial_feat = self.spatial_norm(spatial_feat)
        
        if return_attention:
            return spatial_feat, dynamic_R, attention_weights
        return spatial_feat

    def forward(self, joints_doppler, joints_range):
        B, T_d, J, C = joints_doppler.shape
        T_r = joints_range.size(1)

        # 特征投影和归一化
        dop_feat = self.shared_proj(joints_doppler)  # [B,16,J,H]
        dop_feat = self.input_norm(dop_feat)
        rng_feat = self.shared_proj(joints_range)    # [B,4,J,H]
        rng_feat = self.input_norm(rng_feat)

        # Doppler分支：时空融合
        dop_spatial = self.spatial_graph_propagation(dop_feat) 
        dop_temporal = self._direct_temporal_fusion(dop_feat)  # 使用直接前后帧融合
        
        # 增强的时空融合
        dop_fused = self._enhanced_fusion(dop_spatial, dop_temporal)
        dop_out = self.doppler_proj_out(dop_fused)

        # Range分支：仅空间传播
        rng_spatial = self.spatial_graph_propagation(rng_feat)
        rng_out = self.range_proj_out(rng_spatial)

        # 门控残差连接
        dop_gate = torch.sigmoid(self.doppler_gate)
        dop_out = dop_gate * dop_out + (1 - dop_gate) * joints_doppler
        
        rng_gate = torch.sigmoid(self.range_gate)
        rng_out = rng_gate * rng_out + (1 - rng_gate) * joints_range

        return dop_out, rng_out

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, mode='self'):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, C, N, D_3 = x.shape
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b c n (n_qkv h d) -> n_qkv b h n (c d)', n_qkv = 3, h = self.heads)
        qkv = qkv.flatten(1, 2)
        q, k, v = qkv.unbind()

        mask = (torch.rand((k.shape[1], k.shape[1])) <= 1).to(k.device)
        out = scaled_dot_product_attention(q, k, v, att_mask=mask)

        out = rearrange(out, '(b h) n (c d) -> b c n (h d)', b = B, c = C)
        return self.to_out(out)
  
class PhysicsPriorCalculator(nn.Module):
    """物理先验计算：Doppler积分出距离 / Range求导出速度"""
    def __init__(self, dim, hidden_dim=128, dt=0.01, dropout=0.1):
        super().__init__()
        self.vis_done=False
        self.dt = dt
        # Doppler→速度→积分→距离
        self.doppler_to_vel = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Range→距离→求导→速度
        self.range_to_dist = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 物理量投影回原特征维度（匹配KV维度）
        self.phys_to_dim = nn.Linear(hidden_dim, dim)
   
    def _temporal_integration(self, vel):
        """速度→距离（梯形积分，保留原始时序长度）"""
        B, T, H = vel.shape
        dist = torch.zeros_like(vel)
        dist[:, 0] = vel[:, 0] * self.dt
        if T == 1:
            return dist * self.dt  
        for t in range(1, T):
            dist[:, t] = dist[:, t-1] + (vel[:, t] + vel[:, t-1]) * self.dt / 2
        return dist

    def _temporal_derivative(self, dist):
        """距离→速度（中心差分，保留原始时序长度���"""
        B, T, H = dist.shape
        vel = torch.zeros_like(dist)

        if T == 1:
            # 单帧：速度定义为 0
            vel[:, 0] = 0.
            return vel 

        vel[:, 1:-1] = (dist[:, 2:] - dist[:, :-2]) / (2 * self.dt)
        vel[:, 0] = (dist[:, 1] - dist[:, 0]) / self.dt
        vel[:, -1] = (dist[:, -1] - dist[:, -2]) / self.dt
        return vel

    def get_doppler_dist(self, doppler_feat):
        """Doppler积分→距离特征（给Range做先验）"""
        vel = self.doppler_to_vel(doppler_feat) 
        dist = self._temporal_integration(vel) 
        return self.phys_to_dim(dist)                 # [B, Td, C]

    def get_range_vel(self, range_feat):
        """Range求导→速度特征（给Doppler做先验）"""
        dist = self.range_to_dist(range_feat)  
        vel = self._temporal_derivative(dist)         # [B, Tr, H]
        return self.phys_to_dim(vel)  


class ComplexHolographicFusion(nn.Module):
  
    def __init__(self, dim, dropout=0.1):
        super().__init__()
       
        self.norm_orig = nn.LayerNorm(dim)
        self.norm_prior = nn.LayerNorm(dim)
        
      
        self.W_real = nn.Linear(dim, dim, bias=False)
        self.W_imag = nn.Linear(dim, dim, bias=False)
        
       
        self.phase_shift = nn.Parameter(torch.zeros(1, 1, dim))

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x_orig, x_prior):
        x_orig = self.norm_orig(x_orig)
        x_prior = self.norm_prior(x_prior)
        
        B, T_orig, C = x_orig.shape
        _, T_prior, _ = x_prior.shape

        
        if T_orig < T_prior:
           
            ratio = T_prior // T_orig
            x_prior = x_prior.view(B, T_orig, ratio, C).mean(dim=2)
        elif T_orig > T_prior:
            
            ratio = T_orig // T_prior
            x_prior = x_prior.unsqueeze(2).expand(B, T_prior, ratio, C).reshape(B, T_orig, C)

        
        real_part = self.W_real(x_orig) - self.W_imag(x_prior)
        imag_part = self.W_imag(x_orig) + self.W_real(x_prior)

      
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)

        
        holo_feat = magnitude * torch.cos(phase + self.phase_shift)

      
        out = self.norm_out(x_orig + self.out_proj(holo_feat))
        return out


class CrossModalityFusion(nn.Module):
  
    def __init__(self, dim, depth=2, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ComplexHolographicFusion(dim),  # Range的全息融合
                ComplexHolographicFusion(dim)   
            ]))
            
        # 保留您的物理引擎核心！
        self.phys_calc = PhysicsPriorCalculator(dim, hidden_dim)

    def forward(self, x_range, x_doppler):
        # 1. 物理引擎生成完美的运动学先验
        doppler_dist = self.phys_calc.get_doppler_dist(x_doppler)  
        range_vel = self.phys_calc.get_range_vel(x_range)          

        # 2. 复平面交叉融合
        for range_holo, doppler_holo in self.layers:
            # Range 作为实部，Doppler 推导的伪 Range 作为虚部
            x_range = range_holo(x_orig=x_range, x_prior=doppler_dist)
            # Doppler 作为实部，Range 推导的伪 Doppler 作为虚部
            x_doppler = doppler_holo(x_orig=x_doppler, x_prior=range_vel)
        
        return x_range, x_doppler


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        B, C, N_2, D = x.shape
        q = self.to_q(x[:,:,0:N_2//2,:])
        q = rearrange(q, 'b c n (h d) -> (b h c) n d', h = self.heads)
        k = self.to_k(x[:,:,N_2//2:,:])
        k = rearrange(k, 'b c n (h d) -> (b h c) n d', h = self.heads)
        v = self.to_v(x[:,:,N_2//2:,:])
        v = rearrange(v, 'b c n (h d) -> (b h c) n d', h = self.heads)
        

        mask = (torch.rand((k.shape[1], k.shape[1])) <= 1).to(k.device)
        out = scaled_dot_product_attention(q, k, v, att_mask=mask)

        out = rearrange(out, '(b h c) n d -> b c n (h d)', b = B, c = C)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViTBlock_Cross(nn.Module):
    """
    V0: Indap: cross_attention + followed Cross_FF
    """
    def __init__(self, dim, depth, cross_attn_depth, channel, kernel_size, patch_size_single, patch_size_cross, mlp_dim, dropout=0.):
        super().__init__()
        self.ph_single, self.pw_single = patch_size_single
        self.ph, self.pw = patch_size_cross

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, 4, 8, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]))

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x1, x2):
        y1 = x1.clone()
        y2 = x2.clone()

        # Local representations
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        
        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph_single, pw=self.pw_single)
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph_single, pw=self.pw_single)

        x1 = self.transformer(x1)  # (TODO): test bet. weight share / non-share
        x2 = self.transformer(x2)

        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph_single, w=w//self.pw_single, ph=self.ph_single, pw=self.pw_single)
        x2 = rearrange(x2, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph_single, w=w//self.pw_single, ph=self.ph_single, pw=self.pw_single)
        
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        for cross_attn_1, f_1, f_2 in self.cross_attn_layers:
            cal_qkv = torch.cat((x1,x2), dim=2)
            cal_out = x1 + cross_attn_1(cal_qkv)
            x1_out = f_1(cal_out)  # (TODO): test cal_out = f_1(cal_out)+cal_out or cal_out = f_1(norm(cal_out))+cal_out
            
            cal_qkv = torch.cat((x2,x1), dim=2) 
            cal_out =  x2 + cross_attn_1(cal_qkv)
            x2_out = f_2(cal_out)

        x1 = rearrange(x1_out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x2 = rearrange(x2_out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x1= self.conv3(x1)
        x1 = torch.cat((x1, y1), 1)
        x1 = self.conv4(x1)
        x2 = self.conv3(x2)
        x2 = torch.cat((x2, y2), 1)
        x2 = self.conv4(x2)
        return (x1, x2)

class MobileViT(nn.Module):
    def __init__(self, args, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2), fusion_path_embed='time_centric'):
        super().__init__()
        self.fusion_level = args.fusion.fusion_level
        self.fusion_mode = args.fusion.fusion_mode
        self.project = args.project
        ih, iw = image_size
        if fusion_path_embed=='time_centric':
            patch_single = [(ih//8,1),(ih//16,1),(ih//32,1)]
            patch_cross = [(ih//8,1),(ih//16,1),(ih//32,1)]

        L = [3, 3, 3]
        L_C = [3, 3, 3]

        self.conv1 = conv_nxn_bn(1, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        if self.fusion_mode=='cross':
            self.mvit_cross = nn.ModuleList([])
            self.mvit_cross.append(MobileViTBlock_Cross(dims[0], L[0], L_C[0], channels[5], kernel_size, patch_single[0], patch_cross[0], int(dims[0]*2)))
            self.mvit_cross.append(MobileViTBlock_Cross(dims[1], L[1], L_C[1], channels[7], kernel_size, patch_single[1], patch_cross[1], int(dims[1]*4)))
            self.mvit_cross.append(MobileViTBlock_Cross(dims[2], L[2], L_C[2], channels[9], kernel_size, patch_single[2], patch_cross[2], int(dims[2]*4)))
        else:
            self.mvit = nn.ModuleList([])
            self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, (16, 1), int(dims[0]*2)))
            self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, (8, 1), int(dims[1]*4)))
            self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, (4, 1), int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AdaptiveAvgPool2d((1,None))
        # self.pool = nn.AdaptiveAvgPool2d((1,16))

    def encoding_single(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        return x

    def encoding_cross(self, x1, x2):
        # x1 processing before transformer
        x1 = self.conv1(x1)
        x1 = self.mv2[0](x1)

        x1 = self.mv2[1](x1)
        x1 = self.mv2[2](x1)
        x1 = self.mv2[3](x1)      # Repeat
        # x2 processing before transformer
        x2 = self.conv1(x2)
        x2 = self.mv2[0](x2)

        x2 = self.mv2[1](x2)
        x2 = self.mv2[2](x2)
        x2 = self.mv2[3](x2)      # Repeat

        # cross-attention transformer for multi-view fusion
        x1 = self.mv2[4](x1)
        x2 = self.mv2[4](x2)
        (x1, x2) = self.mvit_cross[0](x1, x2)

        x1 = self.mv2[5](x1)
        x2 = self.mv2[5](x2)
        (x1, x2) = self.mvit_cross[1](x1, x2)

        x1 = self.mv2[6](x1)
        x2 = self.mv2[6](x2)
        (x1, x2) = self.mvit_cross[2](x1, x2)
                
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)

        return (x1, x2)

    def forward(self, x):
        if 'single' in self.fusion_level:
            if '1' in self.fusion_level:
                x = x[:,0,:,:].unsqueeze(dim=1)
            elif '2' in self.fusion_level:
                x = x[:,1,:,:].unsqueeze(dim=1)
            x = self.encoding_single(x)
            x = self.pool(x).squeeze(dim=(2))
        else:
            x1 = x[:,0,:,:].unsqueeze(dim=1)
            x2 = x[:,1,:,:].unsqueeze(dim=1)
            if 'cross' in self.fusion_mode:
                x1, x2 = self.encoding_cross(x1,x2)
                x1 = self.pool(x1).squeeze(dim=(2))
                x2 = self.pool(x2).squeeze(dim=(2))
                x = (x1+x2)/2
            elif 'late' in self.fusion_level:
                if 'average' in self.fusion_mode:
                    x1 = self.encoding_single(x1)
                    x2 = self.encoding_single(x2)
                    x1 = self.pool(x1).squeeze(dim=(2))
                    x2 = self.pool(x2).squeeze(dim=(2))
                    x = (x1+x2)/2
        return x

class main_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_type = args.train.model
        self.decoder_input = args.model.decoder_input
       
        if 'mobileVit' in model_type:
            fusion_patch_embedding = 'time_centric'
            if 'xxs' in model_type:
                dims = [64, 80, 96]
                channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
                expansion = 2
            elif 'xs' in model_type:
                dims = [96, 120, 144]
                channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
                expansion = 4
            elif 's' in model_type:
                dims = [144, 192, 240]
                channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
                expansion = 4
            self.radar_mD_encoder = MobileViT(args, image_size=(args.transforms.Dop_size, args.transforms.win_size), 
                                                    dims=dims, 
                                                    channels=channels, 
                                                    expansion=expansion, 
                                                    fusion_path_embed=fusion_patch_embedding)
            self.radar_Rng_encoder = MobileViT(args, image_size=(args.transforms.R_size_rng, args.transforms.win_size_rng), 
                                                    dims=dims, 
                                                    channels=channels, 
                                                    expansion=expansion, 
                                                    fusion_path_embed=fusion_patch_embedding)
            # self.radar_mD_encoder = MobileViT(args, image_size=(64,600), 
            #                                         dims=dims, 
            #                                         channels=channels, 
            #                                         expansion=expansion, 
            #                                         fusion_path_embed=fusion_patch_embedding)
            # self.radar_Rng_encoder = MobileViT(args, image_size=(256,600), 
            #                                         dims=dims, 
            #                                         channels=channels, 
            #                                         expansion=expansion, 
            #                                         fusion_path_embed=fusion_patch_embedding)
        self.regress_mD = nn.Sequential(
                        nn.Conv1d(channels[-1], 3*17, kernel_size=1),
                        nn.BatchNorm1d(3*17),
                        nn.Tanh()
                        )
        self.regress_Rng = nn.Sequential(
                        nn.Conv1d(channels[-1], 3*17, kernel_size=1),
                        nn.BatchNorm1d(3*17),
                        nn.Tanh()
                        )
        # self.regress_mD = nn.Sequential(
        #                 nn.Conv1d(channels[-1], 2*14, kernel_size=1),
        #                 nn.BatchNorm1d(2*14),
        #                 nn.Tanh()
        #                 )
        # self.regress_Rng = nn.Sequential(
        #                 nn.Conv1d(channels[-1], 2*14, kernel_size=1),
        #                 nn.BatchNorm1d(2*14),
        #                 nn.Tanh()
        #                 )
        self.cross_modal_fusion = CrossModalityFusion(
            dim=channels[-1],  # 与编码器输出维度一致
            depth=2  # 融合层数
        )
        self.graph_fusion = SpatioTemporalGraphFusion(
            num_joints=17,
            feature_dim=3,
            hidden_dim=64,
            doppler_frames=16,  # Doppler帧数
            range_frames=4,     # Range帧数
            dropout=0.1
        )
        # self.graph_fusion = SpatioTemporalGraphFusion(
        #     num_joints=14,
        #     feature_dim=2,
        #     hidden_dim=64,
        #     doppler_frames=16,  # Doppler帧数
        #     range_frames=4,     # Range帧数
        #     dropout=0.1
        # )
        if self.decoder_input=='all':
            decoder_dim = 20
        elif self.decoder_input=='vel':
            decoder_dim = 16
        elif self.decoder_input=='rng':
            decoder_dim = 4
        self.fc = nn.Sequential(
                        nn.Linear(decoder_dim*3*17, 16*3*17),
                        nn.Tanh(),
                        nn.Dropout(p=0.5),
                        nn.Linear(16*3*17, 16*3*17)
                        )
        # self.fc = nn.Sequential(
        #                 nn.Linear(896, 448),
        #                 nn.Tanh(),
        #                 nn.Dropout(p=0.5),
        #                 nn.Linear(448, 448)
        #                 )
        # self.fc_init = nn.Sequential(
        #                 nn.Linear(20*3*17, 1*3*17),
        #                 nn.Tanh(),
        #                 nn.Dropout(p=0.5),
        #                 nn.Linear(1*3*17, 1*3*17)
        #                 )
        # self.fc_vel = nn.Sequential(
        #                 nn.Linear(20*3*17, 16*3*17),
        #                 nn.Tanh(),
        #                 nn.Dropout(p=0.5),
        #                 nn.Linear(16*3*17, 16*3*17)
        #                 )
        #initialize
       
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_mD, x_R):
        # x_mD = self.slope_optimizer(x_mD)  # 对Doppler数据去噪
        # x_R = self.slope_optimizer(x_R) 
        # Encoder
        
        x_mD = self.radar_mD_encoder(x_mD)
        x_R = self.radar_Rng_encoder(x_R)
        
        x_R = rearrange(x_R, 'b c t -> b t c')  # [B, T, C]
        x_mD = rearrange(x_mD, 'b c t -> b t c')
        x_R, x_mD = self.cross_modal_fusion(x_R, x_mD)  # 特征层面融合

        # # 3. 姿态回归与晚期融合
        x_R = rearrange(x_R, 'b t c -> b c t')  # [B, C, T]
        x_mD = rearrange(x_mD, 'b t c -> b c t')
        # Decoder
        B,_,T_mD = x_mD.size()
        B,_,T_R = x_R.size()
        # print(f"x_mR:{x_mD.size()}")
        # print(f"x_R:{x_R.size()}")
        # print("x_mD shape before regress:", x_mD.shape)
        # x_mD = self.regress_mD(x_mD).view(-1,T_mD*17*3)     # 17x3xT_mD
        # x_R = self.regress_Rng(x_R).view(-1,T_R*17*3)       # 17x3xT_R
        x_mD = self.regress_mD(x_mD)  # [B, 51, T_mD]
        x_R  = self.regress_Rng(x_R)  # [B, 51, T_R]

        # reshape 成 [B, T, J, 3]
        x_mD = x_mD.permute(0, 2, 1).contiguous().view(B, T_mD, 17, 3)
        x_R  = x_R.permute(0, 2, 1).contiguous().view(B, T_R, 17, 3)
        # x_mD = x_mD.permute(0, 2, 1).contiguous().view(B, T_mD, 14, 2)
        # x_R  = x_R.permute(0, 2, 1).contiguous().view(B, T_R, 14, 2)
        x_mD, x_R = self.graph_fusion(x_mD, x_R)
        x_mD = x_mD.reshape(B,-1)    # 17x3xT_mD
        x_R = x_R.reshape(B,-1) 
        
        if self.decoder_input=='all':
              
            x = self.fc(torch.cat((x_mD,x_R),dim=-1))
        elif self.decoder_input=='vel':
            x = self.fc(x_mD)
        elif self.decoder_input=='rng':
            x = self.fc(x_R)
        x = rearrange(x, 'b (t j c) -> b t j c', j=17, t=16, c=3).contiguous()
        # x = rearrange(x, 'b (t j c) -> b t j c', j=14, t=16, c=2).contiguous()
        return x
        # return torch.sigmoid(x)
        




def mobilevit_xxs(args):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels, expansion=2)


def mobilevit_xs(args):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels)


def mobilevit_s(args):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 1, 256, 256)
    
    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))