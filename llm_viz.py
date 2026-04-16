import argparse
import copy
import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
import pickle
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from app import load_model, generate_with_attention
from app import state
from app import visualize_token_attention, visualize_prompt_token_attention
from visualization import AttentionVisualizer
import config
from config import CACHE_DIR


def detect_sink_tokens_by_quantile(sink_scores: torch.Tensor, quantile_threshold: float = 0.85) -> tuple:
    """
    基于分位数阈值检测 Sink Tokens
    
    根据 LeCun 团队 2026 年研究 "The Spike, the Sparse and the Sink":
    Sink Tokens 是那些具有异常高 sink scores 的 token，它们会过度吸引注意力，
    导致模型忽略其他重要信息。本函数通过设定分位数阈值（默认 Top 15%）来识别这些 token。
    
    算法逻辑:
    1. 计算 sink scores 的指定分位数作为阈值
    2. 将 score 高于阈值的 token 标记为 Sink Token
    3. 返回 Sink Token 的布尔掩码和索引列表
    
    Args:
        sink_scores: Sink scores 张量，形状为 [seq_len]
        quantile_threshold: 分位数阈值，默认 0.85（即选择 Top 15% 作为 Sink Tokens）
    
    Returns:
        tuple: (is_sink_token, sink_indices, threshold_value)
            - is_sink_token: 布尔掩码张量，形状 [seq_len]，True 表示是 Sink Token
            - sink_indices: Sink Token 的索引列表
            - threshold_value: 实际使用的分位数阈值
    """
    threshold_value = torch.quantile(sink_scores.float(), quantile_threshold)
    is_sink_token = sink_scores >= threshold_value
    sink_indices = torch.where(is_sink_token)[0].tolist()
    
    return is_sink_token, sink_indices, threshold_value.item()


def detect_sink_tokens_by_std(sink_scores: torch.Tensor, k_sigma: float = 3.0) -> tuple:
    """
    基于均值+K倍标准差检测 Sink Tokens
    
    使用统计学中的异常值检测方法：将阈值设为均值加上 K 倍标准差，
    高于此阈值的 token 被视为具有异常高 sink scores 的 Sink Tokens。
    
    算法逻辑:
    1. 计算 sink scores 的均值和标准差
    2. 阈值 = 均值 + k_sigma * 标准差
    3. 将 score 高于阈值的 token 标记为 Sink Token
    
    Args:
        sink_scores: Sink scores 张量，形状为 [seq_len]
        k_sigma: 标准差倍数，默认 3.0（统计学常用的异常值检测阈值）
    
    Returns:
        tuple: (is_sink_token, sink_indices, threshold_value, mean_score, std_score)
            - is_sink_token: 布尔掩码张量，形状 [seq_len]，True 表示是 Sink Token
            - sink_indices: Sink Token 的索引列表
            - threshold_value: 计算得到的动态阈值 (mean + k_sigma * std)
            - mean_score: sink scores 的均值
            - std_score: sink scores 的标准差
    """
    mean_score = sink_scores.mean().float().item()
    std_score = sink_scores.std().float().item()
    threshold_value = mean_score + k_sigma * std_score
    
    is_sink_token = sink_scores >= threshold_value
    sink_indices = torch.where(is_sink_token)[0].tolist()
    
    return is_sink_token, sink_indices, threshold_value, mean_score, std_score


def detect_sink_tokens_by_tau(sink_scores: torch.Tensor, tau: float = 20.0) -> tuple:
    """
    基于绝对阈值检测 Sink Tokens
    
    直接使用预设的绝对阈值 tau 来判定 Sink Tokens，
    适用于 sink scores 具有明确物理意义或经验阈值已知的情况。
    
    算法逻辑:
    1. 设定绝对阈值 tau
    2. 将 score 直接大于 tau 的 token 标记为 Sink Token
    
    Args:
        sink_scores: Sink scores 张量，形状为 [seq_len]
        tau: 绝对阈值，默认 20.0（经验值，可根据实际数据调整）
    
    Returns:
        tuple: (is_sink_token, sink_indices, threshold_value)
            - is_sink_token: 布尔掩码张量，形状 [seq_len]，True 表示是 Sink Token
            - sink_indices: Sink Token 的索引列表
            - threshold_value: 使用的绝对阈值 tau
    """
    threshold_value = tau
    is_sink_token = sink_scores >= threshold_value
    sink_indices = torch.where(is_sink_token)[0].tolist()
    
    return is_sink_token, sink_indices, threshold_value


def visualize_distribution(data, title, xlabel, ylabel="Frequency", color="steelblue", show_stats=True, **kwargs):
    """
    可视化数据分布的辅助函数
    
    Args:
        data: 数据数组 (numpy array)
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        color: 柱状图颜色
        show_stats: 是否显示统计信息线 (mean/min/max)
        **kwargs: 其他matplotlib参数
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(data, bins=50, color=color, edgecolor='black', alpha=0.7, **kwargs)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if show_stats:
        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(min_val, color='green', linestyle=':', linewidth=2, label=f'Min: {min_val:.4f}')
        ax.axvline(max_val, color='orange', linestyle=':', linewidth=2, label=f'Max: {max_val:.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_line_plot(data, title, xlabel, ylabel, color="blue", marker='o', **kwargs):
    """
    绘制折线图的辅助函数
    
    Args:
        data: 数据数组 (numpy array)
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        color: 线条颜色
        marker: 标记样式
        **kwargs: 其他matplotlib参数
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(data, marker=marker, markersize=3, linestyle='-', linewidth=1, color=color, **kwargs)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_cumulative_distribution(data, title, xlabel, color="blue", threshold=None, **kwargs):
    """
    绘制累积分布图的辅助函数
    
    Args:
        data: 数据数组 (numpy array)
        title: 图表标题
        xlabel: x轴标签
        color: 线条颜色
        threshold: 可选的阈值线
        **kwargs: 其他matplotlib参数
    """
    sorted_data = np.sort(data)
    cumprobs = np.arange(len(sorted_data)) / len(sorted_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(sorted_data, cumprobs, color=color, linewidth=2, **kwargs)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_sink_token_analysis(sink_scores, k_sigma, current_token=None, save_dir=None):
    """
    可视化Sink Token分析的完整流程
    
    包括:
    1. sink_scores分布直方图 + 累积分布
    
    Args:
        sink_scores: 已经计算好的sink scores张量 [seq_len]
        k_sigma: 动态阈值的sigma倍数
        current_token: 当前token位置 (可选)
        save_dir: 保存图片的文件夹路径，如果为None则显示图片
    """
    print(f"\n[Clean] 步骤1: 可视化 Sink Scores")
    print(f"  - sink_scores shape: {sink_scores.shape}")
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"  - 图片将保存到: {save_dir}")
    
    print(f"  - sink_scores 统计: min={sink_scores.min().float().item():.4f}, max={sink_scores.max().float().item():.4f}, mean={sink_scores.mean().float().item():.4f}")
    
    # mean_score = sink_scores.mean().float().item()
    # std_score = sink_scores.std().float().item()
    # dynamic_threshold = mean_score + k_sigma * std_score
    
    mean_score = sink_scores.mean().float().item()
    std_score = sink_scores.std().float().item()
    dynamic_threshold = torch.quantile(sink_scores.float(), 0.85).item()
    
    sink_scores_np = sink_scores.cpu().float().numpy()
    fig_scores, axes_scores = plt.subplots(1, 2, figsize=(14, 5))
    
    axes_scores[0].hist(sink_scores_np, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes_scores[0].set_xlabel('Sink Scores', fontsize=12)
    axes_scores[0].set_ylabel('Frequency', fontsize=12)
    axes_scores[0].set_title('Distribution of Sink Scores', fontsize=14, fontweight='bold')
    axes_scores[0].grid(True, alpha=0.3)
    axes_scores[0].axvline(sink_scores_np.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sink_scores_np.mean():.4f}')
    axes_scores[0].axvline(sink_scores_np.min(), color='green', linestyle=':', linewidth=2, label=f'Min: {sink_scores_np.min():.4f}')
    axes_scores[0].axvline(sink_scores_np.max(), color='orange', linestyle=':', linewidth=2, label=f'Max: {sink_scores_np.max():.4f}')
    axes_scores[0].legend()
    
    sorted_scores = np.sort(sink_scores_np)
    cumprobs = np.arange(len(sorted_scores)) / len(sorted_scores)
    axes_scores[1].plot(sorted_scores, cumprobs, color='blue', linewidth=2)
    axes_scores[1].set_xlabel('Sink Scores', fontsize=12)
    axes_scores[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes_scores[1].set_title('Cumulative Distribution of Sink Scores', fontsize=14, fontweight='bold')
    axes_scores[1].grid(True, alpha=0.3)
    axes_scores[1].axvline(dynamic_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {dynamic_threshold:.4f}')
    axes_scores[1].legend()
    
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f"3_sink_scores_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  - 保存: {save_path}")
        plt.close(fig_scores)
    else:
        plt.show()
    
    print(f"  - sink_scores 统计: min={sink_scores.min().float().item():.4f}, max={sink_scores.max().float().item():.4f}, mean={sink_scores.mean().float().item():.4f}")
    print(f"  - 动态阈值计算: mean={mean_score:.4f}, std={std_score:.4f}, threshold={dynamic_threshold:.4f}")
    
    return dynamic_threshold, mean_score, std_score


def extract_vision_attention(attention_dict, vision_token_ranges, current_token):
    """
    从 attention_dict 中提取 vision token 相关的注意力部分
    
    Args:
        attention_dict: 完整的注意力字典 {step_key: {layer_idx: {head_idx: attn_matrix}}}
        vision_token_ranges: Dictionary with 'image' and 'video' token ranges
        current_token: 当前 token 索引
        
    Returns:
        vision_attention_dict: 只包含 vision token 行的注意力字典（深拷贝，保留原始结构）
        other_attention_dict: 包含非 vision token 行的注意力字典（用于后续拼接）
        vision_indices: vision token 的索引列表
        original_info: 原始信息，用于后续恢复（数据类型、其他step等）
    """
    import copy
    
    vision_attention_dict = {}
    other_attention_dict = {}
    
    if current_token not in attention_dict:
        print(f"[Warning] current_token {current_token} 不在 attention_dict 中")
        return attention_dict, {}, [], {}
    
    # 深拷贝原始字典，保留所有 step 和原始数据
    vision_attention_dict = copy.deepcopy(attention_dict)
    other_attention_dict[current_token] = {}
    
    layers = attention_dict[current_token]
    
    vision_token_positions = set()
    if vision_token_ranges:
        for start, end in vision_token_ranges.get('image', []):
            vision_token_positions.update(range(start, end))
    
    vision_indices = sorted(list(vision_token_positions))
    
    if not vision_indices:
        print("[Warning] 未找到 vision token 位置，返回原始 attention_dict")
        return attention_dict, {}, [], {}
    
    # 记录原始信息用于后续恢复
    original_info = {
        'current_token': current_token,
        'vision_indices': vision_indices,
        'tensor_types': {}  # 记录每个 head 的原始数据类型
    }
    
    for layer_idx, heads_dict in layers.items():
        other_attention_dict[current_token][layer_idx] = {}
        original_info['tensor_types'][layer_idx] = {}
        
        for head_idx, attn_matrix in heads_dict.items():
            # 记录原始数据类型
            is_tensor = isinstance(attn_matrix, torch.Tensor)
            original_info['tensor_types'][layer_idx][head_idx] = is_tensor
            
            # 转换为 numpy 进行处理
            if is_tensor:
                attn_matrix_np = attn_matrix.cpu().float().numpy()
            else:
                attn_matrix_np = attn_matrix
            
            if attn_matrix_np.shape[0] == 1:
                vision_attn = attn_matrix_np.copy()
                other_attn = None
            else:
                seq_len = attn_matrix_np.shape[0]
                valid_vision_indices = [i for i in vision_indices if i < seq_len]
                
                if not valid_vision_indices:
                    vision_attn = attn_matrix_np.copy()
                    other_attn = None
                else:
                    all_indices = set(range(seq_len))
                    other_indices = sorted(list(all_indices - set(valid_vision_indices)))
                    
                    vision_attn = attn_matrix_np[valid_vision_indices, :]
                    
                    if other_indices:
                        other_attn = attn_matrix_np[other_indices, :]
                    else:
                        other_attn = None
            
            # 更新 vision_attention_dict 中的值为提取后的 vision 部分
            vision_attention_dict[current_token][layer_idx][head_idx] = vision_attn
            
            if other_attn is not None:
                other_attention_dict[current_token][layer_idx][head_idx] = {
                    'attn': other_attn,
                    'indices': other_indices,
                    'original_shape': attn_matrix_np.shape
                }
    
    return vision_attention_dict, other_attention_dict, vision_indices, original_info


def merge_attention_back(vision_attention_dict, other_attention_dict, vision_indices, current_token, original_info):
    """
    将清洗后的 vision token 注意力与其他 token 注意力拼接回去
    
    Args:
        vision_attention_dict: 清洗后的 vision token 注意力字典（包含所有 step 的深拷贝）
        other_attention_dict: 其他 token 的注意力字典
        vision_indices: vision token 的索引列表
        current_token: 当前 token 索引
        original_info: 原始信息，包含数据类型等
        
    Returns:
        merged_attention_dict: 合并后的完整注意力字典，结构和数据类型与原始一致
    """
    import copy
    import torch
    
    # 从 vision_attention_dict 开始（已经是深拷贝，包含所有 step）
    merged_attention_dict = vision_attention_dict
    
    if current_token not in merged_attention_dict:
        return merged_attention_dict
    
    other_layers = other_attention_dict.get(current_token, {})
    tensor_types = original_info.get('tensor_types', {})
    
    for layer_idx, vision_heads in merged_attention_dict[current_token].items():
        other_heads = other_layers.get(layer_idx, {})
        layer_tensor_types = tensor_types.get(layer_idx, {})
        
        for head_idx, vision_attn in vision_heads.items():
            if head_idx in other_heads:
                other_data = other_heads[head_idx]
                other_attn = other_data['attn']
                other_indices = other_data['indices']
                original_shape = other_data.get('original_shape', None)
                
                # 确保都是 numpy 数组进行拼接
                if isinstance(vision_attn, torch.Tensor):
                    vision_attn_np = vision_attn.cpu().float().numpy()
                else:
                    vision_attn_np = vision_attn
                
                if isinstance(other_attn, torch.Tensor):
                    other_attn_np = other_attn.cpu().float().numpy()
                else:
                    other_attn_np = other_attn
                
                # 计算总序列长度
                if original_shape is not None:
                    total_seq_len = original_shape[0]
                else:
                    total_seq_len = len(vision_indices) + len(other_indices)
                
                # 创建合并后的矩阵
                merged_attn = np.zeros((total_seq_len, vision_attn_np.shape[1]), dtype=vision_attn_np.dtype)
                
                # 将 vision 部分放回原位置
                for i, idx in enumerate(vision_indices):
                    if idx < total_seq_len and i < vision_attn_np.shape[0]:
                        merged_attn[idx, :] = vision_attn_np[i, :]
                
                # 将其他部分放回原位置
                for i, idx in enumerate(other_indices):
                    if idx < total_seq_len and i < other_attn_np.shape[0]:
                        merged_attn[idx, :] = other_attn_np[i, :]
                
                # 恢复原始数据类型
                was_tensor = layer_tensor_types.get(head_idx, False)
                if was_tensor:
                    merged_attn = torch.from_numpy(merged_attn)
                
                merged_attention_dict[current_token][layer_idx][head_idx] = merged_attn
            else:
                # 没有 other 部分，但需要恢复原始数据类型
                was_tensor = layer_tensor_types.get(head_idx, False)
                if was_tensor and not isinstance(vision_attn, torch.Tensor):
                    merged_attention_dict[current_token][layer_idx][head_idx] = torch.from_numpy(vision_attn)
    
    return merged_attention_dict


def _vision_key_positions(vision_token_ranges, seq_len: int):
    """所有图像/视频 vision token 的 key 位置（列索引），限制在 [0, seq_len)。"""
    positions = set()
    if not vision_token_ranges:
        return sorted(positions)
    for start, end in vision_token_ranges.get("image", []):
        for i in range(int(start), int(end)):
            if 0 <= i < seq_len:
                positions.add(i)
    for start, end in vision_token_ranges.get("video", []):
        for i in range(int(start), int(end)):
            if 0 <= i < seq_len:
                positions.add(i)
    return sorted(positions)


def build_mean_vision_attention_1d(processor, step_attention, absolute_token_position, aggregation_method):
    """
    对每层先聚合 head，得到该层 query->vision 的 1D 向量，再对层做平均。
    用于与 2D patch map 互补的数值差分（未经过 create_attention_map 的归一化）。
    """
    if processor is None or step_attention is None:
        return None
    available_layers = sorted(step_attention.keys())
    vecs = []
    for layer_idx in available_layers:
        heads = sorted(step_attention[layer_idx].keys())
        if not heads:
            continue
        v = processor.get_attention_to_vision_tokens(
            step_attention,
            token_position=absolute_token_position,
            layer_indices=[layer_idx],
            head_indices=heads,
            aggregation_method=str(aggregation_method).lower(),
        )
        if v is None:
            continue
        vecs.append(v.detach().cpu().float().numpy().ravel())
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)


def build_mean_all_layers_heatmap(processor, step_attention, absolute_token_position, aggregation_method, normalize: bool):
    """
    与 visualize_token_attention 一致：逐层用该层实际存在的 head 列表聚合，再对层求平均。
    normalize=False 时得到未做 min-max 的 2D map，便于清洗前后数值差分。
    """
    if processor is None or step_attention is None:
        return None
    available_layers = sorted(step_attention.keys())
    layer_maps = []
    for layer_idx in available_layers:
        heads = sorted(step_attention[layer_idx].keys())
        if not heads:
            continue
        m = processor.get_attention_heatmap_for_token(
            step_attention,
            token_position=absolute_token_position,
            layer_indices=[layer_idx],
            head_indices=heads,
            aggregation_method=str(aggregation_method).lower(),
            normalize=normalize,
        )
        if m is not None:
            layer_maps.append(np.asarray(m, dtype=np.float64))
    if not layer_maps:
        return None
    return np.mean(layer_maps, axis=0)


def report_heatmap_diff_stats(
    map_before: np.ndarray,
    map_after: np.ndarray,
    label: str,
    save_path: str = None,
) -> str:
    """计算两张热力图（同 shape）的差分统计，打印并可写入文件。"""
    lines = [f"\n[Diff] {label}"]
    if map_before is None or map_after is None:
        lines.append("  (skip: one of maps is None)")
        text = "\n".join(lines)
        print(text)
        return text
    a = np.asarray(map_before, dtype=np.float64)
    b = np.asarray(map_after, dtype=np.float64)
    if a.shape != b.shape:
        lines.append(f"  shape mismatch: before {a.shape} vs after {b.shape}")
        text = "\n".join(lines)
        print(text)
        return text
    diff = b - a
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    sum_a, sum_b = float(np.sum(a)), float(np.sum(b))
    lines.append(f"  max_abs_diff: {max_abs:.6e}")
    lines.append(f"  mean_abs_diff: {mean_abs:.6e}")
    lines.append(f"  rmse: {rmse:.6e}")
    lines.append(f"  sum(before): {sum_a:.6e}  sum(after): {sum_b:.6e}")
    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        lines.append(f"  pearson_r: {corr:.6f}")
    else:
        lines.append("  pearson_r: (skip: near-constant map)")
    text = "\n".join(lines)
    print(text)
    if save_path:
        try:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError as e:
            print(f"[Diff] 写入报告失败: {save_path} ({e})")
    return text


def clean_attention_dict(
    attention_dict,
    hidden_states,
    vision_token_ranges=None,
    sink_dims=None,
    bad_head_threshold=0.5,
    current_token=None,
    k_sigma=3.0,
    save_dir=None,
    sink_ratio_denominator="all_keys",
    rho=0.8,
    use_paper_method=True,
    vision_attn_prefilter=0.2,
):
    """
    通过hidden_state计算得到sink token的索引
    处理注意力字典，移除被sink token影响的注意头
    
    注意：此函数现在期望接收已经过 extract_vision_attention 处理的 attention_dict，
    即只包含 vision token 行的注意力矩阵
    
    Args:
        attention_dict: Dictionary containing attention weights (vision token only)
        hidden_states: Hidden states of the model
        vision_token_ranges: Dictionary with 'image' and 'video' token ranges
        sink_dims: Sink token dimensions
        bad_head_threshold: Threshold for bad heads
        current_token: Current token index
        k_sigma: 标准差倍数，用于动态计算阈值 (mean + k_sigma * std)
        save_dir: 保存可视化图片的文件夹路径，如果为None则显示图片
        sink_ratio_denominator: 坏头 sink 占比的分母。
            - 'all_keys': 与原先一致，分母为 query 行对所有 key 的注意力总和。
            - 'vision_keys_only': 分母仅为「图像/视频 vision token 列」上的注意力总和，
              更贴近热力图（只看对 vision 列的分配），避免全序列 sink 占比与可视化目标不一致。
        rho: 论文方法中的 visual non-sink ratio 阈值
        use_paper_method: True 使用论文方法；False 使用兼容的旧删坏头逻辑
        vision_attn_prefilter: 论文 prefilter，视觉注意力总和小于该值时跳过删头
    Returns:
        Cleaned attention dictionary
    """
    if sink_ratio_denominator not in ("all_keys", "vision_keys_only"):
        raise ValueError(
            f"sink_ratio_denominator must be 'all_keys' or 'vision_keys_only', got {sink_ratio_denominator!r}"
        )

    if sink_dims is None:
        sink_dims = [458, 2570]

    print("\n" + "="*60)
    print("[Clean] 开始注意力清洗过程")
    print("="*60)
    print(f"[Clean] 参数配置:")
    print(f"  - sink_dims: {sink_dims}")
    print(f"  - k_sigma (动态阈值倍数): {k_sigma}")
    print(f"  - bad_head_threshold (坏头阈值): {bad_head_threshold}")
    print(f"  - sink_ratio_denominator: {sink_ratio_denominator}")
    print(f"  - rho: {rho}")
    print(f"  - use_paper_method: {use_paper_method}")
    print(f"  - vision_attn_prefilter: {vision_attn_prefilter}")
    print(f"  - vision_token_ranges: {vision_token_ranges}")
    print(f"  - current_token: {current_token} (type={type(current_token).__name__})")

    # 检查 current_token 是否对应 attention_dict 的最后一个 step
    step_keys = list(attention_dict.keys())
    print(f"  - attention step keys 数量: {len(step_keys)}")
    if step_keys:
        try:
            step_keys_as_int = sorted(int(k) for k in step_keys)
            last_step_key = step_keys_as_int[-1]
            current_token_as_int = int(current_token) if current_token is not None else None
            is_last_step = (current_token_as_int == last_step_key)
            print(f"  - attention 最后 step key: {last_step_key}")
            print(f"  - current_token 是否最后 step: {is_last_step}")
        except (TypeError, ValueError):
            # 如果 key 不是纯数字，回退到字符串比较
            last_step_key = str(step_keys[-1])
            is_last_step = (str(current_token) == last_step_key)
            print(f"  - attention 最后 step key(字符串): {last_step_key}")
            print(f"  - current_token 是否最后 step(字符串比较): {is_last_step}")
    
    print(f"\n[Clean] 步骤1: 检测 Sink Tokens")
    print(f"  - hidden_states shape: {hidden_states.shape}")
    hs_len = int(hidden_states.shape[0])
    sink_scores = torch.max(
        torch.abs(hidden_states[:, sink_dims])
        / torch.sqrt(torch.mean(hidden_states ** 2, dim=-1, keepdim=True)),
        dim=-1,
    ).values
    # 调用封装的可视化函数，传入已经计算好的 sink_scores
    dynamic_threshold, mean_score, std_score = visualize_sink_token_analysis(sink_scores, k_sigma, current_token, save_dir)

    is_sink_token, sink_indices, threshold_85_percentile = detect_sink_tokens_by_quantile(sink_scores, quantile_threshold=0.85)

    print(f"\n[Clean] Sink Token 检测结果:")
    if len(sink_indices) > 0:
        print(f"  - Sink Token 位置: {sink_indices[:20]}{'...' if len(sink_indices) > 20 else ''}")
        # 打印每个 sink token 的 score
        for idx in sink_indices[:10]:
            print(f"    Token {idx}: sink_score = {sink_scores[idx]:.4f}")

    full_seq_len = int(hidden_states.shape[0])
    vkeys_all = _vision_key_positions(vision_token_ranges, full_seq_len)
    is_visual_full = torch.zeros(full_seq_len, dtype=torch.bool, device=is_sink_token.device)
    if vkeys_all:
        vkeys_tensor = torch.as_tensor(vkeys_all, dtype=torch.long, device=is_sink_token.device)
        is_visual_full[vkeys_tensor] = True

    visual_sink_mask_bool = is_sink_token & is_visual_full
    print(
        f"  - visual_sink_mask: {int(visual_sink_mask_bool.sum().item())} visual sink / "
        f"{int(is_sink_token.sum().item())} total sink"
    )

    # --- 2. 遍历字典进行清洗 ---
    print(f"\n[Clean] 步骤2: 遍历注意力字典进行清洗")

    total_removed = 0
    total_heads_checked = 0

    layers = None
    for candidate in (current_token, str(current_token)):
        if candidate in attention_dict:
            layers = attention_dict[candidate]
            break
    if layers is None and current_token is not None:
        try:
            current_token_as_int = int(current_token)
            if current_token_as_int in attention_dict:
                layers = attention_dict[current_token_as_int]
        except (TypeError, ValueError):
            pass
    if layers is None:
        raise KeyError(f"current_token={current_token!r} not found in attention_dict keys={list(attention_dict.keys())[:10]}")

    layer_order = list(layers.keys())
    last_layer_idx = layer_order[-1] if layer_order else None

    for layer_idx, heads_dict in layers.items():
        print(f"\n  [Layer {layer_idx}]:")
        print(f"    - 包含 {len(heads_dict)} 个注意力头")

        # 我们需要知道序列长度，取任意一个 head 的形状即可
        sample_head_key = next(iter(heads_dict))
        sample_attn = heads_dict[sample_head_key]

        # 统一处理：获取输入序列长度
        seq_len = int(sample_attn.shape[-1])

        print(f"    - 序列长度: {seq_len}")
        print(f"    - 注意力矩阵形状: {sample_attn.shape}")

        # 与 query_attn 对齐：sink 仅对 hidden_states 覆盖的前 hs_len 位有效，其余 key 位视为非 sink；
        # 视觉位按 vision_token_ranges 在完整 seq_len 上展开（避免 hidden 较短时与注意力列数不一致）。
        n_sink_src = min(seq_len, hs_len)
        is_sink_np = np.zeros(seq_len, dtype=np.float64)
        if n_sink_src > 0:
            is_sink_np[:n_sink_src] = is_sink_token[:n_sink_src].detach().cpu().float().numpy()
        current_sink_mask = is_sink_np

        vkeys_here = _vision_key_positions(vision_token_ranges, seq_len)
        current_is_visual_np = np.zeros(seq_len, dtype=np.float64)
        for j in vkeys_here:
            if 0 <= j < seq_len:
                current_is_visual_np[j] = 1.0
        current_vis_non_sink_np = current_is_visual_np * (1.0 - is_sink_np)

        print(f"    - Sink mask 长度: {len(current_sink_mask)}, True数量: {current_sink_mask.sum()}")
        print(
            f"    - visual_sink 列数: {int(np.sum(current_is_visual_np * is_sink_np))}, "
            f"visual_non_sink 列数: {int(current_vis_non_sink_np.sum())}"
        )

        bad_heads = []

        print(f"    - 注意力矩阵已经是 vision token 行（预处理后）")

        for head_idx, attn_matrix in heads_dict.items():
            total_heads_checked += 1

            if isinstance(attn_matrix, torch.Tensor):
                attn_matrix_np = attn_matrix.cpu().float().numpy()
            else:
                attn_matrix_np = attn_matrix

            if attn_matrix_np.shape[0] == 1:
                query_attn = attn_matrix_np[0, :seq_len]
            else:
                query_attn = attn_matrix_np[:, :seq_len]

            print(f"    - Query 注意力形状: {query_attn.shape}")

            if use_paper_method:
                attn_to_visual = float((query_attn * current_is_visual_np).sum())
                if attn_to_visual < vision_attn_prefilter:
                    print(
                        f"      Head {head_idx}: [prefilter skip] "
                        f"attn_to_visual={attn_to_visual:.4f} < {vision_attn_prefilter}"
                    )
                    continue

                attn_to_vis_non_sink = float((query_attn * current_vis_non_sink_np).sum())
                r_non_sink = attn_to_vis_non_sink / attn_to_visual
                print(
                    f"      Head {head_idx}: r_non_sink={r_non_sink:.4f} ({r_non_sink:.2%}) "
                    f"[rho={rho}]"
                )

                if r_non_sink < rho:
                    bad_heads.append((head_idx, float(r_non_sink)))
                    print(f"      [!] Head {head_idx} 被标记为坏头: r_non_sink={r_non_sink:.4f} < {rho}")
            else:
                attn_to_sink = float((query_attn * current_sink_mask).sum())

                if sink_ratio_denominator == "vision_keys_only":
                    vkeys = _vision_key_positions(vision_token_ranges, seq_len)
                    if vkeys:
                        vk = np.asarray(vkeys, dtype=np.int64)
                        if query_attn.ndim == 1:
                            total_attn = float(query_attn[vk].sum())
                        else:
                            total_attn = float(query_attn[:, vk].sum())
                    else:
                        total_attn = float(query_attn.sum())
                else:
                    total_attn = float(query_attn.sum())

                if total_attn > 1e-8:
                    sink_ratio = attn_to_sink / total_attn
                    print(
                        f"      Head {head_idx}: sink_ratio={sink_ratio:.4f} ({sink_ratio:.2%}) "
                        f"[denom={sink_ratio_denominator}]"
                    )

                    # --- 核心判断 ---
                    # 如果这个头超过阈值的精力都在看 Sink，它就是坏头
                    if sink_ratio > bad_head_threshold:
                        bad_heads.append((head_idx, float(sink_ratio)))
                        print(
                            f"      [!] Head {head_idx} 被标记为坏头: "
                            f"sink_ratio={sink_ratio:.4f} > {bad_head_threshold}"
                        )

        # --- 3. 执行删除 ---
        is_last_layer = layer_idx == last_layer_idx
        if is_last_layer and bad_heads:
            print(
                f"\n    [保护] 最后一层 Layer {layer_idx} 跳过删头，"
                f"候选坏头数量: {len(bad_heads)}"
            )
            continue

        if bad_heads:
            # 最小改动策略：避免某层 head 被清空，至少保留 1 个 sink_ratio/r_non_sink 最优的头
            if len(bad_heads) >= len(heads_dict) and len(heads_dict) > 0:
                if use_paper_method:
                    keep_head, keep_metric = max(bad_heads, key=lambda x: x[1])
                    print(
                        f"\n    [保护] 本层坏头数={len(bad_heads)} 与总头数={len(heads_dict)}，"
                        f"为避免空层，保留 Head {keep_head} (r_non_sink={keep_metric:.2%})"
                    )
                else:
                    keep_head, keep_metric = min(bad_heads, key=lambda x: x[1])
                    print(
                        f"\n    [保护] 本层坏头数={len(bad_heads)} 与总头数={len(heads_dict)}，"
                        f"为避免空层，保留 Head {keep_head} (sink_ratio={keep_metric:.2%})"
                    )
                bad_heads = [(h, r) for h, r in bad_heads if h != keep_head]

            print(f"\n    执行删除: 剔除 {len(bad_heads)} 个坏头")
            for bad_head, ratio in bad_heads:
                del heads_dict[bad_head]
                if use_paper_method:
                    print(f"      - 删除 Head {bad_head}: r_non_sink={ratio:.2%}")
                else:
                    print(f"      - 删除 Head {bad_head}: sink_ratio={ratio:.2%}")

            total_removed += len(bad_heads)
        else:
            print(f"\n    无需删除: 本层没有坏头")
    
    print(f"\n" + "="*60)
    print(f"[Clean] 清洗完成统计:")
    print(f"  - 检查的注意力头总数: {total_heads_checked}")
    print(f"  - 剔除的坏头总数: {total_removed}")
    print(f"  - 剔除比例: {total_removed/total_heads_checked*100:.2f}%" if total_heads_checked > 0 else "  - 剔除比例: 0%")
    print("="*60 + "\n")

    return attention_dict

def save_inference_data(image_path: str, prompt: str, generated_text: str, state):
    """
    Save inference data including image path, prompt, generated text, and attention weights.
    
    Args:
        image_path: Path to input image
        prompt: Input prompt
        generated_text: Generated text
        state: Global state containing attention weights and other data
    """
    try:
        # Create save directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), "inference_data")
        os.makedirs(save_dir, exist_ok=True)
        
        # Use fixed directory name for overwrite behavior
        save_path = os.path.join(save_dir, "latest_inference")
        # Remove existing directory if it exists
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "generated_text": generated_text,
            "image_path": image_path
        }
        
        # Save input_ids if available
        if state.current_input_ids is not None:
            input_ids_path = os.path.join(save_path, "input_ids.pkl")
            with open(input_ids_path, "wb") as f:
                pickle.dump(state.current_input_ids, f)
            metadata["input_ids_path"] = "input_ids.pkl"
        
        # Save image_grid_thw if available
        if state.current_image_grid_thw is not None:
            image_grid_thw_path = os.path.join(save_path, "image_grid_thw.pkl")
            with open(image_grid_thw_path, "wb") as f:
                pickle.dump(state.current_image_grid_thw, f)
            metadata["image_grid_thw_path"] = "image_grid_thw.pkl"
        
        # Save attention weights if available
        if state.current_attention is not None:
            # Convert attention weights to CPU and numpy for storage
            attention_data = {}
            for step_key, step_data in state.current_attention.items():
                attention_data[step_key] = {}
                for layer_idx, layer_data in step_data.items():
                    attention_data[step_key][layer_idx] = {}
                    for head_idx, head_data in layer_data.items():
                        attention_data[step_key][layer_idx][head_idx] = head_data.cpu().float().numpy()
            
            attention_path = os.path.join(save_path, "attention_weights.pkl")
            with open(attention_path, "wb") as f:
                pickle.dump(attention_data, f)
            metadata["attention_weights_path"] = "attention_weights.pkl"
        
        # Save hidden states if available (only the last generation step)
        if state.hidden_state is not None:
            hidden_state_data = {}
            for layer_idx, hidden_state in state.hidden_state.items():
                if hidden_state is not None:
                    hidden_state_data[layer_idx] = hidden_state.cpu().float().numpy()
            
            hidden_state_path = os.path.join(save_path, "hidden_states.pkl")
            with open(hidden_state_path, "wb") as f:
                pickle.dump(hidden_state_data, f)
            metadata["hidden_states_path"] = "hidden_states.pkl"
        
        # Save metadata
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        # print(f"[Save] Metadata saved to: {metadata_path}")
        
        # Save token information
        if state.current_tokens is not None:
            tokens_path = os.path.join(save_path, "tokens.json")
            with open(tokens_path, "w", encoding="utf-8") as f:
                json.dump(state.current_tokens, f, ensure_ascii=False, indent=2)
            metadata["tokens_path"] = "tokens.json"
        
        # Save prompt token information if available
        if state.current_prompt_tokens is not None:
            prompt_tokens_path = os.path.join(save_path, "prompt_tokens.json")
            with open(prompt_tokens_path, "w", encoding="utf-8") as f:
                json.dump(state.current_prompt_tokens, f, ensure_ascii=False, indent=2)
            metadata["prompt_tokens_path"] = "prompt_tokens.json"
        
        # print(f"[Save] Inference data saved to: {save_path}")
        
    except Exception as e:
        # error_msg = f"Error saving inference data: {str(e)}"
        # print(error_msg)
        import traceback
        traceback.print_exc()


def visualize_prompt_attention(token_start_idx: int, token_end_idx: int, aggregation_method: str, colormap: str, alpha: float):
    """
    Visualize attention from prompt text tokens to image regions.
    
    Args:
        token_start_idx: Start token index (inclusive)
        token_end_idx: End token index (inclusive)
        aggregation_method: How to aggregate attention heads (mean/max/min)
        colormap: Colormap name
        alpha: Overlay transparency
        
    Returns:
        Dictionary mapping layer names to attention heatmaps, or None if visualization failed
    """
    try:
        # Check if attention data is available
        if state.current_attention is None or state.current_processor is None:
            # print("Error: No attention data available. Please generate text first.")
            return None
        
        if state.current_prompt_tokens is None or len(state.current_prompt_tokens) == 0:
            # print("Error: No prompt tokens available.")
            return None
        
        # Get prompt token info
        prompt_token_info = state.current_processor.get_prompt_text_token_info()
        if not prompt_token_info:
            # print("Error: No prompt token info available.")
            return None
        
        # print(f"Visualizing prompt token range: {token_start_idx} to {token_end_idx}")
        # print(f"Available prompt tokens: {len(state.current_prompt_tokens)}")
        
        # Call the original visualize_prompt_token_attention function
        attention_maps = visualize_prompt_token_attention(
            token_start_idx=token_start_idx,
            token_end_idx=token_end_idx,
            aggregation_method=aggregation_method,
            colormap=colormap,
            alpha=alpha
        )
        
        if attention_maps is None:
            # print("Error: Failed to generate attention maps for prompt tokens")
            return None
        
        # print(f"Successfully generated attention maps for {len(attention_maps)} layers")
        return attention_maps
        
    except Exception as e:
        # print(f"Error in visualize_prompt_attention: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def verify_hidden_state_extraction(state):
    """
    Verify if hidden states are correctly extracted and print their shapes.
    
    Args:
        state: Global state containing attention weights and hidden states
    """
    print("\n" + "="*60)
    print("验证 Hidden State 提取")
    print("="*60)
    
    if state.current_attention is not None:
        print(f"找到 {len(state.current_attention)} 个生成步骤的注意力数据")
        
        for step_key, step_data in state.current_attention.items():
            print(f"\n生成步骤 {step_key}:")
            
            for layer_idx, head_data in step_data.items():
                if isinstance(head_data, dict):
                    print(f"  Layer {layer_idx}: attention heads = {len(head_data)}")
                else:
                    print(f"  Layer {layer_idx}: 数据格式不符合预期 (type: {type(head_data)})")
    else:
        print("没有找到注意力数据")
    
    if state.hidden_state is not None:
        print(f"\n找到最后一个生成步的 hidden state 数据（共 {len(state.hidden_state)} 层）")
        
        for layer_idx, hidden_state in state.hidden_state.items():
            if hidden_state is not None:
                print(f"  Layer {layer_idx}: hidden_state shape = {hidden_state.shape}")
            else:
                print(f"  Layer {layer_idx}: hidden_state = None")
    else:
        print("没有找到 hidden state 数据")
    
    print("="*60 + "\n")


def find_massive_activation_dims(hidden_states_batch, top_k=3):
    """
    从隐藏状态中识别 Massive Activation Dimensions (大值激活维度/Spikes)。
    
    根据 LeCun 团队 2026 年研究 "The Spike, the Sparse and the Sink":
    - Massive Activations 是某些维度激活值异常大 (Spikes)
    - 通常由 FFN 的 SwiGLU 在前几个位置产生
    - 经过 RMSNorm 后主导整个表示，是 Attention Sink 形成的前提
    
    Args:
        hidden_states_batch: 形状 [Seq_Len, Hidden_Size] 的张量
                            例如: 一批 token 的隐藏状态
        top_k: 要返回的维度数量
    
    Returns:
        top_dims: Top-K 大值激活维度的索引列表
        stats: 统计信息字典
    """
    if not isinstance(hidden_states_batch, torch.Tensor):
        hidden_states_batch = torch.tensor(hidden_states_batch)
    
    # 计算 RMS 归一化
    rms = torch.sqrt(torch.mean(hidden_states_batch ** 2, dim=-1, keepdim=True))
    rms = torch.clamp(rms, min=1e-8)
    normalized = torch.abs(hidden_states_batch) / rms
    
    # 沿序列维度取最大值，得到每个维度的峰值激活
    max_per_dim = torch.max(normalized, dim=0).values
    
    # 找激活值最大的维度 (Massive Activations / Spikes)
    top_dims = torch.topk(max_per_dim, top_k, largest=True).indices
    
    print(f"发现的 Massive Activation Dimensions (大值激活维度): {top_dims.tolist()}")
    
    stats = {
        'max_per_dim': max_per_dim,
        'mean_activation': torch.mean(max_per_dim).item(),
        'std_activation': torch.std(max_per_dim).item(),
        'max_activation': torch.max(max_per_dim).item(),
        'min_activation': torch.min(max_per_dim).item()
    }
    
    return top_dims.tolist(), stats


def analyze_all_layers_massive_activations(state, top_k=5):
    """
    分析所有层的 Massive Activations (大值激活维度)。
    
    根据 LeCun 团队 2026 年研究 "The Spike, the Sparse and the Sink":
    Massive Activations 是由 FFN 的 SwiGLU 在前几个位置产生的极端异常值，
    经过 RMSNorm 后主导整个表示，是 Attention Sink 形成的前提条件。
    
    Args:
        state: 全局状态，包含 hidden_state 数据
        top_k: 每层要返回的维度数量
    
    Returns:
        all_massive_dims: 字典，键为层索引，值为该层的大值激活维度信息
    """
    print("\n" + "="*60)
    print("分析所有层的 Massive Activations (大值激活维度)")
    print("="*60)
    
    if state.hidden_state is None:
        print("错误: 没有 hidden state 数据")
        return None
    
    all_massive_dims = {}
    
    for layer_idx, hidden_state in state.hidden_state.items():
        if hidden_state is not None:
            print(f"\n--- Layer {layer_idx} ---")
            massive_dims, stats = find_massive_activation_dims(hidden_state, top_k=top_k)
            all_massive_dims[layer_idx] = {
                'dims': massive_dims,
                'stats': stats
            }
            print(f"  平均激活: {stats['mean_activation']:.4f}, "
                  f"标准差: {stats['std_activation']:.4f}, "
                  f"最大激活: {stats['max_activation']:.4f}")
    
    print("\n" + "="*60)
    
    return all_massive_dims


# 设置参数解析器
parser = argparse.ArgumentParser(description='Qwen2.5-VL Attention Visualization with OpenCV')
parser.add_argument('--image', type=str, required=True, help='Path to input image')
parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image description')
parser.add_argument('--model_path', type=str, default=config.MODEL_NAME, help='Path to model')
parser.add_argument('--content_order', type=str, default='Image → Text', choices=['Image → Text', 'Text → Image'], help='Order of image and text in prompt')
parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens to generate')
parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
parser.add_argument('--colormap', type=str, default='jet', help='Colormap for heatmap visualization')
parser.add_argument('--alpha', type=float, default=0.6, help='Alpha value for heatmap overlay')
parser.add_argument('--aggregation_method', type=str, default=config.DEFAULT_AGGREGATION, choices=['mean', 'max', 'min'], help='Method to aggregate attention heads')
parser.add_argument(
    '--clean_sink_ratio_denominator',
    type=str,
    default='all_keys',
    choices=['all_keys', 'vision_keys_only'],
    help="Sink 坏头判定中 sink_ratio 的分母：all_keys=全序列 key；vision_keys_only=仅图像/视频 vision 列（更贴近热力图）",
)
parser.add_argument('--layer_name', type=str, default=None, help='Specific layer to visualize (e.g., "Layer 0" or "Mean (All Layers)")')
args = parser.parse_args()

def main():
    # 清空cache文件夹
    # print(f"Clearing cache directory: {CACHE_DIR}")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # print("Cache directory cleared successfully")
    
    # 清空debug log文件
    from attention_processor import AttentionProcessor
    from pathlib import Path
    log_file = Path(CACHE_DIR) / "attention_debug.log"
    if log_file.exists():
        log_file.unlink()
    # print("Debug log file cleared successfully")
    
    # # 创建专门的目录来保存所有注意力可视化结果
    # visualize_dir = os.path.join(os.getcwd(), "attention_visualizations")
    # os.makedirs(visualize_dir, exist_ok=True)
    # # print(f"Created visualization directory: {visualize_dir}")
    
    # 1. 加载模型
    # print(f"Loading model from: {args.model_path}")
    model_status = load_model(args.model_path)
    
    # 检查模型是否加载成功
    if "✗" in model_status or "Error" in model_status:
        # print(f"Model loading failed: {model_status}")
        # print("Please check the model path and try again.")
        return
    
    # print(f"Model loaded successfully: {model_status}")
    
    # 2. 加载图像
    # print(f"Loading image from: {args.image}")
    try:
        image = Image.open(args.image)
        print(f"Image loaded successfully: {image.size} pixels")
        
        # Resize image: limit the longest edge to 512
        max_edge = 512
        width, height = image.size
        if max(width, height) > max_edge:
            if width >= height:
                new_width = max_edge
                new_height = int(height * (max_edge / width))
            else:
                new_height = max_edge
                new_width = int(width * (max_edge / height))
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"Image resized to: {image.size} pixels (max edge: {max_edge})")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return
    
    # 3. 生成文本并提取注意力权重
    print(f"Generating text with attention extraction...")
    try:
        result = generate_with_attention(
            image=image,
            prompt=args.prompt,
            content_order=args.content_order,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 处理返回值
        generated_text, status, display_html = result[0], result[1], result[2]
        
        # print(f"Generated text: {generated_text}")
        # print(f"Status: {status}")
        
        # 检查生成是否成功
        if "✗" in status or "Error" in status:
            print(f"Text generation failed: {status}")
            return
        
        # 保存推理数据
        save_inference_data(args.image, args.prompt, generated_text, state)
        # print("[Save] Inference data saved successfully!")
        
        # 验证hidden_state是否被正确提取
        #verify_hidden_state_extraction(state)
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 检查是否有生成的token
    if state.current_tokens is None or len(state.current_tokens) == 0:
        print("No tokens generated, cannot visualize attention")
        return
    
    # 5. 保存注意力热力图到./save目录
    # print(f"Visualizing attention for {len(state.current_tokens)} tokens...")
    
    # 创建保存目录
    save_dir = os.path.join(os.getcwd(), "save")
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建注意力可视化器
    visualizer = AttentionVisualizer(colormap=args.colormap, alpha=args.alpha)
    print("生成的token range"+str(min(state.current_attention.keys()))+"-"+str(max(state.current_attention.keys())))
    # 收集所有token的注意力图
    all_attention_maps = []
    input_len = min(state.current_attention.keys())
    # 确定要可视化的token索引
    num_tokens = len(state.current_tokens)
    # token_indices_to_visualize = [0, 1]  # prefill阶段(0)和第一个生成步(1) - 已注释
    token_indices_to_visualize = []  # 只可视化最后一个token
    
    # 添加最后一个token（如果存在）
    last_token_idx = len(state.current_tokens) - 1
    if last_token_idx >= 0:
        token_indices_to_visualize.append(last_token_idx)
    
    # 遍历指定的token
    for token_idx in token_indices_to_visualize:
        # print(f"Visualizing attention for token {token_idx}: {state.current_tokens[token_idx]}")
        token_selector = f"Token {token_idx}: '{state.current_tokens[token_idx]}'"
        did_dual_heatmap = False

        # 4.5 应用 Sink Token 注意力清洗，并同时导出「未清洗 / 已清洗」两张热力图
        if state.current_attention and state.hidden_state:
            try:
                print(f"[Clean-Call] token_idx={token_idx+input_len}, last_token_idx={last_token_idx+input_len}, is_last_generated_token={token_idx == last_token_idx}")
                print(f"[Clean-Call] token_text='{state.current_tokens[token_idx]}'")
                first_layer_idx = min(state.hidden_state.keys())
                hidden_state_for_cleaning = state.hidden_state[first_layer_idx]

                print("\n" + "="*60)
                print("应用 Sink Token 注意力清洗")
                ranges = state.current_processor.vision_token_ranges['image']
                print("vision_token_ranges 数量:", len(ranges))
                print("vision_token_ranges 前3个:", ranges[:3])
                print("="*60)

                current_token_key = token_idx + input_len
                vision_token_ranges = state.current_processor.vision_token_ranges if state.current_processor else None

                attention_before_split = copy.deepcopy(state.current_attention)

                print("\n[Pre-process] 提取 vision token 注意力部分...")
                vision_attention_dict, other_attention_dict, vision_indices, original_info = extract_vision_attention(
                    attention_dict=state.current_attention,
                    vision_token_ranges=vision_token_ranges,
                    current_token=current_token_key
                )
                print(f"[Pre-process] 提取完成: vision_indices 数量 = {len(vision_indices)}")

                if not vision_indices:
                    print("[Warning] 未找到 vision token，跳过清洗")
                    continue

                vision_clean = copy.deepcopy(vision_attention_dict)

                print("\n[Clean] 开始清洗 vision token 注意力...")
                vision_clean = clean_attention_dict(
                    current_token=current_token_key,
                    attention_dict=vision_clean,
                    hidden_states=hidden_state_for_cleaning,
                    vision_token_ranges=vision_token_ranges,
                    sink_dims=[458, 2570],
                    k_sigma=3.0,
                    bad_head_threshold=0.5,
                    save_dir=r'./save',
                    sink_ratio_denominator=args.clean_sink_ratio_denominator,
                )

                print("\n[Post-process] 拼接清洗后的 vision 注意力回完整矩阵...")
                merged_clean = merge_attention_back(
                    vision_attention_dict=vision_clean,
                    other_attention_dict=other_attention_dict,
                    vision_indices=vision_indices,
                    current_token=current_token_key,
                    original_info=original_info
                )
                print("[Post-process] 拼接完成")

                layer_name = "Mean (All Layers)"
                absolute_token_position = input_len + token_idx
                diff_report_path = os.path.join(
                    save_dir, f"token_{token_idx}_heatmap_diff_report.txt"
                )
                try:
                    with open(diff_report_path, "w", encoding="utf-8") as df:
                        df.write(
                            f"token_idx={token_idx} absolute_pos={absolute_token_position}\n"
                            f"cli_aggregation={args.aggregation_method} "
                            f"clean_sink_ratio_denominator={args.clean_sink_ratio_denominator}\n\n"
                        )
                except OSError as e:
                    print(f"[Diff] 无法创建报告文件: {diff_report_path} ({e})")
                    diff_report_path = None

                # 未归一化 Mean(All Layers) 图：用于数值差分（不受 per-map min-max 影响）
                proc = state.current_processor
                step_before = attention_before_split.get(current_token_key)
                step_after = merged_clean.get(current_token_key)
                for _, agg_m in (
                    ("mean", "mean"),
                    ("max", "max"),
                ):
                    unnorm_before = build_mean_all_layers_heatmap(
                        proc, step_before, absolute_token_position, agg_m, normalize=False
                    )
                    unnorm_after = build_mean_all_layers_heatmap(
                        proc, step_after, absolute_token_position, agg_m, normalize=False
                    )
                    report_heatmap_diff_stats(
                        unnorm_before,
                        unnorm_after,
                        label=f"unnormalized Mean(All Layers), head_agg={agg_m}",
                        save_path=diff_report_path,
                    )

                # 主流程：CLI 指定的聚合方式（常为 max）导出 not_clean / clean_sink
                for merged_attn, headline, suffix in (
                    (attention_before_split, "has not clean", "not_clean"),
                    (merged_clean, "has clean sink token", "clean_sink"),
                ):
                    state.current_attention = merged_attn
                    attention_maps = visualize_token_attention(
                        token_selector=token_selector,
                        aggregation_method=args.aggregation_method,
                        colormap=args.colormap,
                        alpha=args.alpha
                    )
                    if attention_maps is None or layer_name not in attention_maps:
                        print(f"No attention maps for token {token_idx} ({suffix})")
                        continue
                    attention_map = attention_maps[layer_name]
                    if suffix == "clean_sink":
                        all_attention_maps.append(attention_map)

                    heatmap_overlay = visualizer.create_heatmap_overlay(
                        state.current_image,
                        attention_map
                    )
                    cv_img = cv2.cvtColor(np.array(heatmap_overlay), cv2.COLOR_RGB2BGR)
                    cv2.putText(cv_img, headline, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    subtitle = (
                        f"Token {token_idx}: {state.current_tokens[token_idx]} - {layer_name} "
                        f"(agg={args.aggregation_method})"
                    )
                    cv2.putText(cv_img, subtitle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    save_path = os.path.join(save_dir, f"token_{token_idx}_heatmap_{suffix}.png")
                    try:
                        pil_img.save(save_path)
                        print(f"热力图已保存: {save_path}")
                    except Exception as e:
                        print(f"保存热力图失败: {save_path}")
                        print(f"Error: {str(e)}")

                # 验证：固定用 mean 聚合再导出一对图，避免 max 掩盖删头效果
                for merged_attn, headline, suffix in (
                    (attention_before_split, "has not clean (agg=mean)", "not_clean_aggmean"),
                    (merged_clean, "has clean sink (agg=mean)", "clean_sink_aggmean"),
                ):
                    state.current_attention = merged_attn
                    attention_maps = visualize_token_attention(
                        token_selector=token_selector,
                        aggregation_method="mean",
                        colormap=args.colormap,
                        alpha=args.alpha
                    )
                    if attention_maps is None or layer_name not in attention_maps:
                        print(f"No attention maps for token {token_idx} ({suffix})")
                        continue
                    attention_map = attention_maps[layer_name]
                    heatmap_overlay = visualizer.create_heatmap_overlay(
                        state.current_image,
                        attention_map
                    )
                    cv_img = cv2.cvtColor(np.array(heatmap_overlay), cv2.COLOR_RGB2BGR)
                    cv2.putText(cv_img, headline, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    subtitle = f"Token {token_idx}: {state.current_tokens[token_idx]} - {layer_name} (agg=mean)"
                    cv2.putText(cv_img, subtitle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    save_path = os.path.join(save_dir, f"token_{token_idx}_heatmap_{suffix}.png")
                    try:
                        pil_img.save(save_path)
                        print(f"热力图已保存: {save_path}")
                    except Exception as e:
                        print(f"保存热力图失败: {save_path}")
                        print(f"Error: {str(e)}")

                # 归一化后的 Mean(All Layers) 差分（与叠图观感更一致）
                for _, agg_m in (
                    ("mean", "mean"),
                    ("max", "max"),
                ):
                    norm_before = build_mean_all_layers_heatmap(
                        proc, step_before, absolute_token_position, agg_m, normalize=True
                    )
                    norm_after = build_mean_all_layers_heatmap(
                        proc, step_after, absolute_token_position, agg_m, normalize=True
                    )
                    report_heatmap_diff_stats(
                        norm_before,
                        norm_after,
                        label=f"normalized Mean(All Layers), head_agg={agg_m}",
                        save_path=diff_report_path,
                    )

                # 1D：对 vision token 列的聚合注意力向量（跨层平均），head 聚合用 mean
                vec_b = build_mean_vision_attention_1d(
                    proc, step_before, absolute_token_position, "mean"
                )
                vec_a = build_mean_vision_attention_1d(
                    proc, step_after, absolute_token_position, "mean"
                )
                report_heatmap_diff_stats(
                    vec_b,
                    vec_a,
                    label="1D mean vision attention (mean over layers, head_agg=mean)",
                    save_path=diff_report_path,
                )

                state.current_attention = merged_clean
                did_dual_heatmap = True
                all_massive_dims = analyze_all_layers_massive_activations(state, top_k=5)
                print("="*60 + "\n")

            except Exception as e:
                print(f"[Warning] Sink Token 清洗失败: {e}")
                import traceback
                traceback.print_exc()

        if did_dual_heatmap:
            continue

        attention_maps = visualize_token_attention(
            token_selector=token_selector,
            aggregation_method=args.aggregation_method,
            colormap=args.colormap,
            alpha=args.alpha
        )

        if attention_maps is None:
            print(f"No attention maps found for token {token_idx}")
            continue
        all_massive_dims = analyze_all_layers_massive_activations(state, top_k=5)
        if "Mean (All Layers)" in attention_maps:
            layer_name = "Mean (All Layers)"
            attention_map = attention_maps[layer_name]
            all_attention_maps.append(attention_map)

            heatmap_overlay = visualizer.create_heatmap_overlay(
                state.current_image,
                attention_map
            )
            cv_img = cv2.cvtColor(np.array(heatmap_overlay), cv2.COLOR_RGB2BGR)
            cv2.putText(cv_img, "has not clean", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            subtitle = f"Token {token_idx}: {state.current_tokens[token_idx]} - {layer_name}"
            cv2.putText(cv_img, subtitle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            save_path = os.path.join(save_dir, f"token_{token_idx}_heatmap.png")
            try:
                pil_img.save(save_path)
                print(f"热力图已保存: {save_path}")
            except Exception as e:
                print(f"保存热力图失败: {save_path}")
                print(f"Error: {str(e)}")
        else:
            pass
    
    # 保存所有token的平均注意力热力图
    if len(all_attention_maps) > 0:
        # print("Generating average attention heatmap for all tokens...")
        
        # 计算所有token的平均注意力
        avg_attention_map = np.mean(all_attention_maps, axis=0)
        
        # 创建热力图叠加
        avg_heatmap_overlay = visualizer.create_heatmap_overlay(
            state.current_image,
            avg_attention_map
        )
        
        # 转换PIL图像为OpenCV格式
        avg_cv_img = cv2.cvtColor(np.array(avg_heatmap_overlay), cv2.COLOR_RGB2BGR)
        
        # 添加标题
        avg_title = f"Average Attention for All {len(all_attention_maps)} Tokens - Mean (All Layers)"
        cv2.putText(avg_cv_img, avg_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        pil_avg_img = Image.fromarray(cv2.cvtColor(avg_cv_img, cv2.COLOR_BGR2RGB))
        
        # 保存热力图到./save目录
        avg_save_path = os.path.join(save_dir, "average_attention_heatmap.png")
        try:
            pil_avg_img.save(avg_save_path)
            print(f"平均注意力热力图已保存: {avg_save_path}")
        except Exception as e:
            print(f"保存平均注意力热力图失败: {avg_save_path}")
            print(f"Error: {str(e)}")
        
        # # 显示图像
        # cv2.imshow("Attention Heatmap", avg_cv_img)
        
        # # 等待按键
        # key = cv2.waitKey(0)
        # if key == 27 or key == ord('q'):  # ESC或'q'键退出
        #     cv2.destroyAllWindows()
        #     return
    
    # 6. 保存prompt tokens的注意力热力图
    # print("\nVisualizing prompt tokens attention...")
    # if state.current_prompt_tokens is not None and len(state.current_prompt_tokens) > 0:
    #     # print(f"Found {len(state.current_prompt_tokens)} prompt tokens")
    #     # print(f"Prompt tokens: {state.current_prompt_tokens}")
        
    #     # 可视化前几个prompt tokens的注意力
    #     prompt_token_start = 13
    #     prompt_token_end = 18
        
    #     prompt_attention_maps = visualize_prompt_attention(
    #         token_start_idx=prompt_token_start,
    #         token_end_idx=prompt_token_end,
    #         aggregation_method=args.aggregation_method,
    #         colormap=args.colormap,
    #         alpha=args.alpha
    #     )
        
    #     if prompt_attention_maps is not None:
    #         # 保存prompt tokens的平均注意力热力图
    #         if "Mean (All Layers)" in prompt_attention_maps:
    #             layer_name = "Mean (All Layers)"
    #             attention_map = prompt_attention_maps[layer_name]
                
    #             # 创建热力图叠加
    #             heatmap_overlay = visualizer.create_heatmap_overlay(
    #                 state.current_image,
    #                 attention_map
    #             )
                
    #             # 转换PIL图像为OpenCV格式
    #             cv_img = cv2.cvtColor(np.array(heatmap_overlay), cv2.COLOR_RGB2BGR)
                
    #             # 添加标题
    #             title = f"Prompt Tokens {prompt_token_start}-{prompt_token_end} - {layer_name}"
    #             cv2.putText(cv_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
    #             # 将OpenCV格式转换回PIL图像
    #             pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                
    #             # 保存热力图到./save目录
    #             prompt_save_path = os.path.join(save_dir, f"prompt_tokens_{prompt_token_start}_{prompt_token_end}_heatmap.png")
    #             try:
    #                 pil_img.save(prompt_save_path)
    #                 print(f"Prompt tokens热力图已保存: {prompt_save_path}")
    #             except Exception as e:
    #                 print(f"保存Prompt tokens热力图失败: {prompt_save_path}")
    #                 print(f"Error: {str(e)}")
                
    #             # # 显示图像
    #             # cv2.imshow("Prompt Tokens Attention Heatmap", cv_img)
                
    #             # # 等待按键
    #             # key = cv2.waitKey(0)
    #             # if key == 27 or key == ord('q'):  # ESC或'q'键退出
    #             #     cv2.destroyAllWindows()
    #             #     return
    #         else:
    #             print(f"No attention maps found for prompt Mean (All Layers)")
    #             pass
    #     else:
    #         print(f"No attention maps found for prompt tokens {prompt_token_start}-{prompt_token_end}")
    #         pass
    # else:
    #     print("No prompt tokens available for visualization")
    #     pass
    
    # cv2.destroyAllWindows()
    print("generate_text: ", state.generated_text)

if __name__ == "__main__":
    main()



