import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
import pickle
import shutil
from datetime import datetime
from app import load_model, generate_with_attention
from app import state
from app import visualize_token_attention, visualize_prompt_token_attention
from visualization import AttentionVisualizer
import config
from config import CACHE_DIR


def clean_attention_dict(attention_dict, hidden_states, vision_token_ranges=None, sink_dims=[1874, 1819], tau=20, bad_head_threshold=0.5,current_token=None):
    """
    通过hidden_state计算得到sink token的索引
    处理注意力字典，移除被sink token影响的注意头
    
    Args:
        attention_dict: Dictionary containing attention weights
        hidden_states: Hidden states of the model
        vision_token_ranges: Dictionary with 'image' and 'video' token ranges
        sink_dims: Sink token dimensions
        tau: Temperature parameter
        bad_head_threshold: Threshold for bad heads
        current_token: Current token index
    Returns:
        Cleaned attention dictionary
    """
    """
    原地修改 attention_dict，剔除被 Sink Token 吸引的坏头
    """
    
    print("\n" + "="*60)
    print("[Clean] 开始注意力清洗过程")
    print("="*60)
    print(f"[Clean] 参数配置:")
    print(f"  - sink_dims: {sink_dims}")
    print(f"  - tau (sink检测阈值): {tau}")
    print(f"  - bad_head_threshold (坏头阈值): {bad_head_threshold}")
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
    
    # --- 1. 先找出哪些 Token 是 Sink Token ---
    print(f"\n[Clean] 步骤1: 检测 Sink Tokens")
    print(f"  - hidden_states shape: {hidden_states.shape}")
    
    # hidden_states shape: [seq_len, 2048]
    sink_vals = hidden_states[:, sink_dims]
    print(f"  - sink_vals shape: {sink_vals.shape}")
    print(f"  - sink_vals 统计: min={sink_vals.min():.4f}, max={sink_vals.max():.4f}, mean={sink_vals.mean():.4f}")
    
    rms = torch.sqrt(torch.mean(hidden_states ** 2, dim=-1, keepdim=True))
    rms = torch.clamp(rms, min=1e-8)
    print(f"  - rms shape: {rms.shape}")
    print(f"  - rms 统计: min={rms.min():.4f}, max={rms.max():.4f}, mean={rms.mean():.4f}")
    
    sink_scores = torch.max(torch.abs(sink_vals) / rms, dim=-1).values
    print(f"  - sink_scores shape: {sink_scores.shape}")
    print(f"  - sink_scores 统计: min={sink_scores.min():.4f}, max={sink_scores.max():.4f}, mean={sink_scores.mean():.4f}")
    
    # 得到一个布尔列表，True 代表是 Sink Token
    is_sink_token = sink_scores >= tau  # shape: [seq_len]
    sink_indices = torch.where(is_sink_token)[0].tolist()
    
    print(f"\n[Clean] Sink Token 检测结果:")
    print(f"  - 检测到 {len(sink_indices)} 个 Sink Tokens (阈值: {tau})")
    if len(sink_indices) > 0:
        print(f"  - Sink Token 位置: {sink_indices[:20]}{'...' if len(sink_indices) > 20 else ''}")
        # 打印每个 sink token 的 score
        for idx in sink_indices[:10]:
            print(f"    Token {idx}: sink_score = {sink_scores[idx]:.4f}")

    # --- 2. 遍历字典进行清洗 ---
    print(f"\n[Clean] 步骤2: 遍历注意力字典进行清洗")
    
    total_removed = 0
    total_heads_checked = 0
    
    # for step_key, layers in attention_dict.items():
    #     print(f"\n[Clean] 处理 Step {step_key}:")
    #     print(f"  - 包含 {len(layers)} 个层")
    layers = attention_dict[current_token]
        
    for layer_idx, heads_dict in layers.items():
        print(f"\n  [Layer {layer_idx}]:")
        print(f"    - 包含 {len(heads_dict)} 个注意力头")
        
        # 我们需要知道序列长度，取任意一个 head 的形状即可
        sample_head_key = next(iter(heads_dict))
        sample_attn = heads_dict[sample_head_key]
        
        # 统一处理：获取序列长度
        if isinstance(sample_attn, torch.Tensor):
            seq_len = sample_attn.shape[0]
        else:
            seq_len = sample_attn.shape[0]
        
        print(f"    - 序列长度: {seq_len}")
        print(f"    - 注意力矩阵形状: {sample_attn.shape}")
        
        # 确保 is_sink_token 长度匹配（防止 padding 差异）
        # 转换为 numpy 数组以便与注意力矩阵相乘
        current_sink_mask = is_sink_token[:seq_len].cpu().numpy()
        print(f"    - Sink mask 长度: {len(current_sink_mask)}, True数量: {current_sink_mask.sum()}")
        
        bad_heads = []
        
        # 构建所有 vision token 位置的集合
        vision_token_positions = set()
        if vision_token_ranges:
            for start, end in vision_token_ranges.get('image', []):
                vision_token_positions.update(range(start, end))
        
        # 如果没有提供 vision_token_ranges，使用默认值（前256个）
        if not vision_token_positions:
            print("    [Warning] 未提供 vision_token_ranges，使用默认值（前256个tokens）")
            vision_token_positions = set(range(min(256, seq_len)))
        
        print(f"    - Vision Token 位置数量: {len(vision_token_positions)}")
        if vision_token_positions:
            vision_list = sorted(list(vision_token_positions))[:10]
            print(f"    - Vision Token 位置示例: {vision_list}{'...' if len(vision_token_positions) > 10 else ''}")
        
        # 使用实际的 vision token 位置作为 Query
        vision_query_indices = sorted([i for i in vision_token_positions if i < seq_len])
        
        if not vision_query_indices:
            print("    [Warning] 没有有效的 vision query indices，跳过此层")
            continue
        
        print(f"    - 有效的 Vision Query 数量: {len(vision_query_indices)}")
        
        for head_idx, attn_matrix in heads_dict.items():
            total_heads_checked += 1
            
            # 统一转换为 numpy 数组处理
            if isinstance(attn_matrix, torch.Tensor):
                attn_matrix_np = attn_matrix.cpu().numpy()
            else:
                attn_matrix_np = attn_matrix
            
            # attn_matrix shape: [seq_len, seq_len]
            # 我们关注的是：作为 Query 的图像 Token，是否把注意力给了 Sink
            
            # 提取 vision tokens 作为 Query 的注意力
            query_attn = attn_matrix_np[vision_query_indices, :] # [num_vision, seq_len]
            
            # 计算给 Sink 的总注意力
            # 修复：使用 numpy 广播机制
            attn_to_sink = (query_attn * current_sink_mask).sum()
            # 计算总注意力
            total_attn = query_attn.sum()
            
            if total_attn > 1e-8:
                sink_ratio = attn_to_sink / total_attn
                
                # 每10个head打印一次统计信息
                if head_idx % 10 == 0:
                    print(f"      Head {head_idx}: sink_ratio={sink_ratio:.4f} ({sink_ratio:.2%})")
                
                # --- 核心判断 ---
                # 如果这个头超过阈值的精力都在看 Sink，它就是坏头
                if sink_ratio > bad_head_threshold:
                    bad_heads.append((head_idx, float(sink_ratio)))
                    print(f"      [!] Head {head_idx} 被标记为坏头: sink_ratio={sink_ratio:.4f} > {bad_head_threshold}")
        
        # --- 3. 执行删除 ---
        if bad_heads:
            print(f"\n    执行删除: 剔除 {len(bad_heads)} 个坏头")
            for bad_head, ratio in bad_heads:
                del heads_dict[bad_head]
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
                        attention_data[step_key][layer_idx][head_idx] = head_data.cpu().numpy()
            
            attention_path = os.path.join(save_path, "attention_weights.pkl")
            with open(attention_path, "wb") as f:
                pickle.dump(attention_data, f)
            metadata["attention_weights_path"] = "attention_weights.pkl"
        
        # Save hidden states if available (only the last generation step)
        if state.hidden_state is not None:
            hidden_state_data = {}
            for layer_idx, hidden_state in state.hidden_state.items():
                if hidden_state is not None:
                    hidden_state_data[layer_idx] = hidden_state.cpu().numpy()
            
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
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', help='Path to model')
parser.add_argument('--content_order', type=str, default='Image → Text', choices=['Image → Text', 'Text → Image'], help='Order of image and text in prompt')
parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens to generate')
parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
parser.add_argument('--colormap', type=str, default='jet', help='Colormap for heatmap visualization')
parser.add_argument('--alpha', type=float, default=0.6, help='Alpha value for heatmap overlay')
parser.add_argument('--aggregation_method', type=str, default=config.DEFAULT_AGGREGATION, choices=['mean', 'max', 'min'], help='Method to aggregate attention heads')
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
    
    # 收集所有token的注意力图
    all_attention_maps = []
    
    # 确定要可视化的token索引
    num_tokens = len(state.current_tokens)
    # token_indices_to_visualize = [0, 1]  # prefill阶段(0)和第一个生成步(1) - 已注释
    token_indices_to_visualize = []  # 只可视化最后一个token
    
    # 添加最后一个token（如果存在）
    last_token_idx = num_tokens - 1
    if last_token_idx >= 0:
        token_indices_to_visualize.append(last_token_idx)
    
    # 遍历指定的token
    for token_idx in token_indices_to_visualize:
        # print(f"Visualizing attention for token {token_idx}: {state.current_tokens[token_idx]}")
         # 4.5 应用 Sink Token 注意力清洗
        if state.current_attention and state.hidden_state:
            try:
                print(f"[Clean-Call] token_idx={token_idx}, last_token_idx={last_token_idx}, is_last_generated_token={token_idx == last_token_idx}")
                print(f"[Clean-Call] token_text='{state.current_tokens[token_idx]}'")
                # 使用最后一个层的 hidden state 来检测 sink tokens
                last_layer_idx = max(state.hidden_state.keys())
                hidden_state_for_cleaning = state.hidden_state[last_layer_idx]
                
                print("\n" + "="*60)
                print("应用 Sink Token 注意力清洗")
                print("="*60)
                
                # 清洗注意力数据
                state.current_attention = clean_attention_dict(
                    current_token = token_idx,
                    attention_dict=state.current_attention,
                    hidden_states=hidden_state_for_cleaning,
                    vision_token_ranges=state.current_processor.vision_token_ranges if state.current_processor else None,
                    sink_dims=[1874, 1819],
                    tau=20,
                    bad_head_threshold=0.5
                )
                
                print("="*60 + "\n")
                
            except Exception as e:
                print(f"[Warning] Sink Token 清洗失败: {e}")
                import traceback
                traceback.print_exc()
        # 创建token选择器字符串
        token_selector = f"Token {token_idx}: '{state.current_tokens[token_idx]}'"
        
        # 获取注意力图
        attention_maps = visualize_token_attention(
            token_selector=token_selector,
            aggregation_method=args.aggregation_method,
            colormap=args.colormap,
            alpha=args.alpha
        )
        
        if attention_maps is None:
            print(f"No attention maps found for token {token_idx}")
            continue
        # 分析隐藏状态中的 Massive Activations
        all_massive_dims = analyze_all_layers_massive_activations(state, top_k=5)
        # 只显示模型整体对图片的注意力（Mean (All Layers)）
        if "Mean (All Layers)" in attention_maps:
            layer_name = "Mean (All Layers)"
            attention_map = attention_maps[layer_name]
            
            # 收集注意力图
            all_attention_maps.append(attention_map)
            
            # 创建热力图叠加
            heatmap_overlay = visualizer.create_heatmap_overlay(
                state.current_image,
                attention_map
            )
            
            # 转换PIL图像为OpenCV格式
            cv_img = cv2.cvtColor(np.array(heatmap_overlay), cv2.COLOR_RGB2BGR)
            
            # 添加标题
            title = f"Token {token_idx}: {state.current_tokens[token_idx]} - {layer_name}"
            cv2.putText(cv_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 将OpenCV格式转换回PIL图像
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            
            # 保存热力图到./save目录
            save_path = os.path.join(save_dir, f"token_{token_idx}_heatmap.png")
            try:
                pil_img.save(save_path)
                print(f"热力图已保存: {save_path}")
            except Exception as e:
                print(f"保存热力图失败: {save_path}")
                print(f"Error: {str(e)}")
            
            # # 显示图像
            # cv2.imshow("Attention Heatmap", cv_img)
            
            # # 等待按键
            # key = cv2.waitKey(0)
            # if key == 27:  # ESC键退出
            #     cv2.destroyAllWindows()
            #     return
            # elif key == ord('q'):  # 'q'键退出
            #     cv2.destroyAllWindows()
            #     return
        else:
            # print("No Mean (All Layers) attention map found")
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

if __name__ == "__main__":
    main()
