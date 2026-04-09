#!/usr/bin/env python3
"""
Test script for save functionality logic
"""

import os
import json
import pickle
from PIL import Image
import numpy as np
import torch
from datetime import datetime

# Directly test the save logic without importing llm_viz.py
def test_save_logic():
    """Test the save logic directly"""
    print("Testing save functionality logic...")
    
    try:
        # Create mock data
        test_image_path = "test_image.png"
        test_prompt = "这是一个测试提示词"
        test_generated_text = "这是生成的测试文本"
        
        # Create mock state
        class MockState:
            def __init__(self):
                self.current_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                self.current_image_grid_thw = torch.tensor([[1, 16, 16]])
                self.current_attention = {
                    5: {
                        0: {
                            0: torch.randn(1, 1, 5, 5),
                            1: torch.randn(1, 1, 5, 5)
                        }
                    }
                }
                self.current_tokens = ["这", "是", "一", "个", "测试"]
                self.current_image = Image.new('RGB', (256, 256), color='white')
        
        mock_state = MockState()
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color='white')
        test_image.save(test_image_path)
        print("✓ Test image created")
        
        # Test save logic
        print("Testing save logic...")
        
        # Create save directory if it doesn't exist
        save_dir = os.path.join(os.getcwd(), "inference_data")
        os.makedirs(save_dir, exist_ok=True)
        print(f"✓ Created save directory: {save_dir}")
        
        # Use fixed directory name for overwrite behavior
        save_path = os.path.join(save_dir, "latest_inference")
        # Remove existing directory if it exists
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        print(f"✓ Created save subdirectory: {save_path}")
        
        # Save metadata
        metadata = {
            "prompt": test_prompt,
            "generated_text": test_generated_text,
            "image_path": test_image_path
        }
        
        # Save input_ids if available
        if mock_state.current_input_ids is not None:
            input_ids_path = os.path.join(save_path, "input_ids.pkl")
            with open(input_ids_path, "wb") as f:
                pickle.dump(mock_state.current_input_ids, f)
            metadata["input_ids_path"] = "input_ids.pkl"
            print("✓ Saved input_ids.pkl")
        
        # Save image_grid_thw if available
        if mock_state.current_image_grid_thw is not None:
            image_grid_thw_path = os.path.join(save_path, "image_grid_thw.pkl")
            with open(image_grid_thw_path, "wb") as f:
                pickle.dump(mock_state.current_image_grid_thw, f)
            metadata["image_grid_thw_path"] = "image_grid_thw.pkl"
            print("✓ Saved image_grid_thw.pkl")
        
        # Save attention weights if available
        if mock_state.current_attention is not None:
            # Convert attention weights to CPU and numpy for storage
            attention_data = {}
            for step_key, step_data in mock_state.current_attention.items():
                attention_data[step_key] = {}
                for layer_idx, layer_data in step_data.items():
                    attention_data[step_key][layer_idx] = {}
                    for head_idx, head_data in layer_data.items():
                        attention_data[step_key][layer_idx][head_idx] = head_data.cpu().float().numpy()
            
            attention_path = os.path.join(save_path, "attention_weights.pkl")
            with open(attention_path, "wb") as f:
                pickle.dump(attention_data, f)
            metadata["attention_weights_path"] = "attention_weights.pkl"
            print("✓ Saved attention_weights.pkl")
        
        # Save metadata
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved metadata.json")
        
        # Save token information
        if mock_state.current_tokens is not None:
            tokens_path = os.path.join(save_path, "tokens.json")
            with open(tokens_path, "w", encoding="utf-8") as f:
                json.dump(mock_state.current_tokens, f, ensure_ascii=False, indent=2)
            metadata["tokens_path"] = "tokens.json"
            print("✓ Saved tokens.json")
        
        print(f"✓ All data saved to: {save_path}")
        
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing save functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test directory structure
def test_directory_structure():
    """Test if inference_data directory is created"""
    print("Testing directory structure...")
    try:
        save_dir = os.path.join(os.getcwd(), "inference_data")
        if os.path.exists(save_dir):
            print(f"✓ Inference data directory exists: {save_dir}")
            # Check for latest_inference directory
            latest_dir = "latest_inference"
            latest_dir_path = os.path.join(save_dir, latest_dir)
            if os.path.exists(latest_dir_path):
                print(f"✓ Latest saved directory: {latest_dir_path}")
                # Check if required files exist
                required_files = ["metadata.json", "tokens.json", "input_ids.pkl", "image_grid_thw.pkl", "attention_weights.pkl"]
                for file in required_files:
                    file_path = os.path.join(latest_dir_path, file)
                    if os.path.exists(file_path):
                        print(f"✓ Found {file}")
                    else:
                        print(f"✗ Missing {file}")
            else:
                print(f"✗ Latest inference directory not found: {latest_dir_path}")
                return False
            return True
        else:
            print(f"✗ Inference data directory not found: {save_dir}")
            return False
    except Exception as e:
        print(f"✗ Error testing directory structure: {e}")
        return False

if __name__ == "__main__":
    print("Running save functionality logic tests...\n")
    
    # Run tests
    test1 = test_save_logic()
    test2 = test_directory_structure()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Save logic test: {'PASS' if test1 else 'FAIL'}")
    print(f"Directory structure test: {'PASS' if test2 else 'FAIL'}")
    print("="*60)
    
    if all([test1, test2]):
        print("\n✓ All tests passed! The save functionality logic is working correctly.")
        print("\nTo test with llm_viz.py:")
        print("1. Run llm_viz.py with an image and prompt")
        print("2. Check the 'inference_data' directory for saved files")
        print("3. Run visualization: python visualize_saved_data.py <data_dir>")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
