#!/usr/bin/env python3
"""
Script to print the structure of the Qwen2.5-VL model

This script loads the model and prints its detailed structure,
including layers, heads, and parameter counts.
"""

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import config

def print_model_summary(model):
    """
    Print a detailed summary of the model structure
    """
    print("=" * 80)
    print("MODEL STRUCTURE SUMMARY")
    print("=" * 80)
    
    # Print model type and basic info
    print(f"Model type: {type(model).__name__}")
    print(f"Model device: {next(model.parameters()).device}")
    print()
    
    # Print language model structure
    print("LANGUAGE MODEL STRUCTURE:")
    print("-" * 80)
    
    # Access the language model
    language_model = model.model.language_model
    print(f"Language model type: {type(language_model).__name__}")
    
    # Print encoder (if exists) or decoder layers
    if hasattr(language_model, 'layers'):
        num_layers = len(language_model.layers)
        print(f"Number of layers: {num_layers}")
        
        # Print details of the first layer as example
        if num_layers > 0:
            first_layer = language_model.layers[0]
            print(f"First layer type: {type(first_layer).__name__}")
            
            # Check for self-attention
            if hasattr(first_layer, 'self_attn'):
                self_attn = first_layer.self_attn
                print(f"Self-attention type: {type(self_attn).__name__}")
                # Check number of heads
                if hasattr(self_attn, 'num_heads'):
                    print(f"Number of attention heads: {self_attn.num_heads}")
                elif hasattr(self_attn, 'head_dim'):
                    print(f"Attention head dimension: {self_attn.head_dim}")
            
            # Check for MLP
            if hasattr(first_layer, 'mlp'):
                mlp = first_layer.mlp
                print(f"MLP type: {type(mlp).__name__}")
    
    # Print vision model structure if exists
    print()
    print("VISION MODEL STRUCTURE:")
    print("-" * 80)
    
    if hasattr(model.model, 'vision_model'):
        vision_model = model.model.vision_model
        print(f"Vision model type: {type(vision_model).__name__}")
        
        # Print vision encoder layers
        if hasattr(vision_model, 'encoder'):
            vision_encoder = vision_model.encoder
            print(f"Vision encoder type: {type(vision_encoder).__name__}")
            
            if hasattr(vision_encoder, 'layers'):
                num_vision_layers = len(vision_encoder.layers)
                print(f"Number of vision layers: {num_vision_layers}")
                
                # Print details of the first vision layer
                if num_vision_layers > 0:
                    first_vision_layer = vision_encoder.layers[0]
                    print(f"First vision layer type: {type(first_vision_layer).__name__}")
    else:
        print("No separate vision model found (integrated into language model)")
    
    # Print parameter counts
    print()
    print("PARAMETER COUNTS:")
    print("-" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {(total_params - trainable_params):,}")
    
    # Print parameter sizes by component
    if hasattr(model.model, 'language_model'):
        lm_params = sum(p.numel() for p in model.model.language_model.parameters())
        print(f"Language model parameters: {lm_params:,} ({lm_params/total_params*100:.1f}%)")
    
    if hasattr(model.model, 'vision_model'):
        vision_params = sum(p.numel() for p in model.model.vision_model.parameters())
        print(f"Vision model parameters: {vision_params:,} ({vision_params/total_params*100:.1f}%)")
    
    print()
    print("MODEL CONFIGURATION:")
    print("-" * 80)
    print(model.config)
    print("=" * 80)

def main():
    """
    Main function to load model and print structure
    """
    parser = argparse.ArgumentParser(description='Print Qwen2.5-VL model structure')
    parser.add_argument('--model_path', type=str, 
                        default=config.MODEL_NAME, 
                        help='Path to model (local or Hugging Face hub)')
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    print("This may take a few minutes...")
    
    try:
        # Load model with eager attention implementation for compatibility
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=config.MODEL_TORCH_DTYPE,
            device_map=config.MODEL_DEVICE_MAP,
            attn_implementation="eager"  # Required for output_attentions support
        )
        
        print("Model loaded successfully!")
        print()
        
        # Print model structure
        print_model_summary(model)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
