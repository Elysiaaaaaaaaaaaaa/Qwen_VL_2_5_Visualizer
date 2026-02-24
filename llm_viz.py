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
from app import visualize_token_attention
from visualization import AttentionVisualizer
from config import CACHE_DIR


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
        
        # Save metadata
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[Save] Metadata saved to: {metadata_path}")
        
        # Save token information
        if state.current_tokens is not None:
            tokens_path = os.path.join(save_path, "tokens.json")
            with open(tokens_path, "w", encoding="utf-8") as f:
                json.dump(state.current_tokens, f, ensure_ascii=False, indent=2)
            metadata["tokens_path"] = "tokens.json"
        
        print(f"[Save] Inference data saved to: {save_path}")
        
    except Exception as e:
        error_msg = f"Error saving inference data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()


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
parser.add_argument('--aggregation_method', type=str, default='mean', choices=['mean', 'max', 'min'], help='Method to aggregate attention heads')
parser.add_argument('--layer_name', type=str, default=None, help='Specific layer to visualize (e.g., "Layer 0" or "Mean (All Layers)")')
args = parser.parse_args()

def main():
    # 清空cache文件夹
    print(f"Clearing cache directory: {CACHE_DIR}")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Cache directory cleared successfully")
    
    # 1. 加载模型
    print(f"Loading model from: {args.model_path}")
    model_status = load_model(args.model_path)
    
    # 检查模型是否加载成功
    if "✗" in model_status or "Error" in model_status:
        print(f"Model loading failed: {model_status}")
        print("Please check the model path and try again.")
        return
    
    print(f"Model loaded successfully: {model_status}")
    
    # 2. 加载图像
    print(f"Loading image from: {args.image}")
    try:
        image = Image.open(args.image)
        print(f"Image loaded successfully: {image.size} pixels")
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
        
        print(f"Generated text: {generated_text}")
        print(f"Status: {status}")
        
        # 检查生成是否成功
        if "✗" in status or "Error" in status:
            print(f"Text generation failed: {status}")
            return
        
        # 保存推理数据
        save_inference_data(args.image, args.prompt, generated_text, state)
        print("[Save] Inference data saved successfully!")
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 检查是否有生成的token
    if state.current_tokens is None or len(state.current_tokens) == 0:
        print("No tokens generated, cannot visualize attention")
        return
    
    # 5. 使用OpenCV展示注意力热力图
    print(f"Visualizing attention for {len(state.current_tokens)} tokens...")
    
    # 创建注意力可视化器
    visualizer = AttentionVisualizer(colormap=args.colormap, alpha=args.alpha)
    
    # 收集所有token的注意力图
    all_attention_maps = []
    
    # 遍历所有生成的token
    for token_idx in range(len(state.current_tokens)):
        # print(f"Visualizing attention for token {token_idx}: {state.current_tokens[token_idx]}")
        
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
            
            # 将添加了标题的热力图保存到cache文件夹
            cache_save_path = os.path.join(CACHE_DIR, f"token_{token_idx}_heatmap.png")
            # 将OpenCV格式转换回PIL图像并保存
            cv2.imwrite(cache_save_path, cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            print(f"Heatmap with title saved to cache: {cache_save_path}")
            
            # 显示图像
            cv2.imshow("Attention Heatmap", cv_img)
            
            # 等待按键
            key = cv2.waitKey(0)
            if key == 27:  # ESC键退出
                cv2.destroyAllWindows()
                return
            elif key == ord('q'):  # 'q'键退出
                cv2.destroyAllWindows()
                return
        else:
            print("No Mean (All Layers) attention map found")
    
    # 显示所有token的平均注意力热力图
    if len(all_attention_maps) > 0:
        print("Generating average attention heatmap for all tokens...")
        
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
        
        # 将添加了标题的平均热力图保存到cache文件夹
        avg_cache_save_path = os.path.join(CACHE_DIR, "average_attention_heatmap.png")
        # 将OpenCV格式转换回RGB并保存
        cv2.imwrite(avg_cache_save_path, cv2.cvtColor(avg_cv_img, cv2.COLOR_BGR2RGB))
        print(f"Average heatmap with title saved to cache: {avg_cache_save_path}")
        
        # 显示图像
        cv2.imshow("Attention Heatmap", avg_cv_img)
        
        # 等待按键
        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):  # ESC或'q'键退出
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()