#!/usr/bin/env python3
"""
Visualization script for saved inference data

This script loads saved inference data (including image, prompt, attention weights)
and generates visualizations without re-running the model.
"""

import os
import json
import pickle
import argparse
from PIL import Image
import numpy as np
import torch
import cv2

import config
import utils
from attention_processor import AttentionProcessor
from visualization import AttentionVisualizer


def load_inference_data(data_dir):
    """
    Load saved inference data from directory.
    
    Args:
        data_dir: Directory containing saved inference data
        
    Returns:
        Dictionary with loaded data
    """
    print(f"Loading inference data from: {data_dir}")
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata: {metadata}")
    
    # Load image
    image_path = metadata.get("image_path", "")
    if os.path.exists(image_path):
        # If image path is absolute or relative and exists
        image = Image.open(image_path)
        print(f"Loaded image from: {image_path}")
    else:
        # If image path doesn't exist, create a placeholder
        print(f"Image path not found: {image_path}, creating placeholder")
        image = Image.new('RGB', (256, 256), color='gray')
    
    # Load input_ids
    input_ids = None
    if "input_ids_path" in metadata:
        input_ids_path = os.path.join(data_dir, metadata["input_ids_path"])
        if os.path.exists(input_ids_path):
            with open(input_ids_path, "rb") as f:
                input_ids = pickle.load(f)
            print(f"Loaded input_ids: {input_ids.shape}")
    
    # Load image_grid_thw
    image_grid_thw = None
    if "image_grid_thw_path" in metadata:
        image_grid_thw_path = os.path.join(data_dir, metadata["image_grid_thw_path"])
        if os.path.exists(image_grid_thw_path):
            with open(image_grid_thw_path, "rb") as f:
                image_grid_thw = pickle.load(f)
            print(f"Loaded image_grid_thw: {image_grid_thw.shape}")
    
    # Load attention weights
    attention_weights = None
    if "attention_weights_path" in metadata:
        attention_weights_path = os.path.join(data_dir, metadata["attention_weights_path"])
        if os.path.exists(attention_weights_path):
            with open(attention_weights_path, "rb") as f:
                attention_data = pickle.load(f)
            
            # Convert numpy arrays back to torch tensors
            attention_weights = {}
            for step_key, step_data in attention_data.items():
                attention_weights[step_key] = {}
                for layer_idx, layer_data in step_data.items():
                    attention_weights[step_key][layer_idx] = {}
                    for head_idx, head_data in layer_data.items():
                        attention_weights[step_key][layer_idx][head_idx] = torch.tensor(head_data)
            print(f"Loaded attention weights for {len(attention_weights)} steps")
    
    # Load tokens
    tokens = None
    if "tokens_path" in metadata:
        tokens_path = os.path.join(data_dir, metadata["tokens_path"])
        if os.path.exists(tokens_path):
            with open(tokens_path, "r", encoding="utf-8") as f:
                tokens = json.load(f)
            print(f"Loaded {len(tokens)} tokens")
    
    return {
        "metadata": metadata,
        "image": image,
        "input_ids": input_ids,
        "image_grid_thw": image_grid_thw,
        "attention_weights": attention_weights,
        "tokens": tokens
    }


def visualize_attention(data, output_dir, token_index=0, layer_indices=None, head_indices=None, 
                       aggregation_method="mean", colormap="viridis", alpha=0.6):
    """
    Visualize attention for saved data.
    
    Args:
        data: Loaded inference data
        output_dir: Directory to save visualizations
        token_index: Index of token to visualize
        layer_indices: Specific layers to use
        head_indices: Specific heads to use
        aggregation_method: How to aggregate attention
        colormap: Colormap name
        alpha: Overlay transparency
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create attention processor
    processor = AttentionProcessor(
        input_ids=data["input_ids"],
        image_grid_thw=data["image_grid_thw"],
        original_image_size=(data["image"].width, data["image"].height)
    )
    
    # Get input length
    input_length = data["input_ids"].shape[1]
    
    # Get available steps
    available_steps = sorted(data["attention_weights"].keys())
    print(f"Input length: {input_length}")
    print(f"Available generation steps: {available_steps}")
    
    # Create visualizer
    visualizer = AttentionVisualizer(colormap=colormap, alpha=alpha)
    
    # Collect all attention maps for mean calculation across tokens
    all_token_attention_maps = []
    
    # Get tokens from metadata if available
    tokens = data.get("tokens") or []
    num_tokens = len(tokens)
    print(f"Number of generated tokens: {num_tokens}")
    
    # If no tokens available, use all generation steps
    if num_tokens == 0:
        # Use all available steps (excluding the first one which is prefill)
        token_indices = range(1, len(available_steps)) if len(available_steps) > 1 else [0]
    else:
        # Use all generated tokens
        token_indices = range(num_tokens)
    
    print(f"Visualizing {len(token_indices)} tokens...")
    
    # Process each token
    for token_idx in token_indices:
        # Calculate absolute token position
        absolute_token_position = input_length + token_idx
        step_key = absolute_token_position
        
        print(f"\nProcessing token {token_idx}: position={absolute_token_position}")
        
        # Find closest step if exact match not found
        if step_key not in data["attention_weights"]:
            print(f"Step key {step_key} not found, using closest available")
            step_key = min(available_steps, key=lambda x: abs(x - step_key))
            print(f"Using step key: {step_key}")
        
        step_attention = data["attention_weights"][step_key]
        
        # Get available layers and heads
        available_layers = sorted(step_attention.keys())
        if not available_layers:
            print("No layers available in attention data")
            continue
        
        sample_layer = available_layers[0]
        available_heads = sorted(step_attention[sample_layer].keys())
        
        # Get mean attention across all layers for this token
        mean_attention_map = processor.get_attention_heatmap_for_token(
            step_attention,
            token_position=absolute_token_position,
            layer_indices=available_layers,
            head_indices=available_heads,
            aggregation_method=aggregation_method,
            normalize=True
        )
        
        if mean_attention_map is not None:
            all_token_attention_maps.append(mean_attention_map)
            
            # Create visualization
            overlay = visualizer.create_heatmap_overlay(
                data["image"],
                mean_attention_map
            )
            
            # Save visualization
            output_path = os.path.join(output_dir, f"token_{token_idx}_mean_all_layers.png")
            overlay.save(output_path)
            print(f"Saved visualization for token {token_idx} to: {output_path}")
            
            # 转换PIL图像为OpenCV格式
            cv_img = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
            
            # 添加标题
            token_text = tokens[token_idx] if token_idx < len(tokens) else f"Token {token_idx}"
            title = f"Token {token_idx}: {token_text} - Mean (All Layers)"
            cv2.putText(cv_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
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
            print(f"No attention map found for token {token_idx}")
    
    # Calculate average attention across all tokens
    if len(all_token_attention_maps) > 1:
        avg_attention_map = np.mean(all_token_attention_maps, axis=0)
        
        # Create average visualization
        avg_overlay = visualizer.create_heatmap_overlay(
            data["image"],
            avg_attention_map
        )
        
        # Save average visualization
        avg_output_path = os.path.join(output_dir, "average_all_tokens.png")
        avg_overlay.save(avg_output_path)
        print(f"Saved average visualization across all tokens to: {avg_output_path}")
        
        # 转换PIL图像为OpenCV格式
        avg_cv_img = cv2.cvtColor(np.array(avg_overlay), cv2.COLOR_RGB2BGR)
        
        # 添加标题
        avg_title = f"Average Attention - All {len(all_token_attention_maps)} Tokens"
        cv2.putText(avg_cv_img, avg_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示图像
        cv2.imshow("Attention Heatmap", avg_cv_img)
        
        # 等待按键
        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):  # ESC或'q'键退出
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()
    print(f"All visualizations saved to: {output_dir}")


def main():
    """
    Main function for visualization script.
    """
    parser = argparse.ArgumentParser(description="Visualize saved inference data")
    parser.add_argument("data_dir", help="Directory containing saved inference data")
    parser.add_argument("--output-dir", default=None, help="Directory to save visualizations")
    parser.add_argument("--token-index", type=int, default=0, help="Index of token to visualize")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "max", "min"],
                      help="Attention aggregation method")
    parser.add_argument("--colormap", default="viridis", help="Colormap name")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay transparency")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "visualizations")
    
    try:
        # Load data
        data = load_inference_data(args.data_dir)
        
        # Print metadata
        print("\n" + "="*60)
        print("INFERENCE METADATA")
        print("="*60)
        print(f"Prompt: {data['metadata']['prompt']}")
        print(f"Generated text: {data['metadata']['generated_text']}")
        print(f"Tokens available: {len(data['tokens']) if data['tokens'] else 0}")
        print("="*60 + "\n")
        
        # Visualize attention
        visualize_attention(
            data=data,
            output_dir=args.output_dir,
            token_index=args.token_index,
            aggregation_method=args.aggregation,
            colormap=args.colormap,
            alpha=args.alpha
        )
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
