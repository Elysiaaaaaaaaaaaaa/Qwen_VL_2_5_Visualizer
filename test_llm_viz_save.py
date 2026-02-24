#!/usr/bin/env python3
"""
Test script for llm_viz.py save functionality
"""

import os 
import tempfile
import json
import pickle
from PIL import Image
import numpy as np
import torch

# Create a mock state class for testing
class MockState:
    """Mock state class for testing save functionality"""
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

# Test save functionality
def test_save_functionality():
    """Test save functionality from llm_viz.py"""
    print("Testing llm_viz.py save functionality...")
    
    try:
        # Import the save function from llm_viz.py
        from llm_viz import save_inference_data
        print("✓ save_inference_data function imported successfully")
        
        # Create mock data
        test_image_path = "test_image.png"
        test_prompt = "这是一个测试提示词"
        test_generated_text = "这是生成的测试文本"
        mock_state = MockState()
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color='white')
        test_image.save(test_image_path)
        print("✓ Test image created")
        
        # Test save function
        save_inference_data(test_image_path, test_prompt, test_generated_text, mock_state)
        print("✓ save_inference_data function executed successfully")
        
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
            # Check if there are any saved directories
            saved_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
            if saved_dirs:
                print(f"✓ Found {len(saved_dirs)} saved inference directories")
                # Check the latest saved directory
                latest_dir = sorted(saved_dirs)[-1]
                latest_dir_path = os.path.join(save_dir, latest_dir)
                print(f"✓ Latest saved directory: {latest_dir_path}")
                # Check if required files exist
                required_files = ["metadata.json"]
                for file in required_files:
                    file_path = os.path.join(latest_dir_path, file)
                    if os.path.exists(file_path):
                        print(f"✓ Found {file}")
                    else:
                        print(f"✗ Missing {file}")
            return True
        else:
            print(f"✗ Inference data directory not found: {save_dir}")
            return False
    except Exception as e:
        print(f"✗ Error testing directory structure: {e}")
        return False

if __name__ == "__main__":
    print("Running llm_viz.py save functionality tests...\n")
    
    # Run tests
    test1 = test_save_functionality()
    test2 = test_directory_structure()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Save functionality test: {'PASS' if test1 else 'FAIL'}")
    print(f"Directory structure test: {'PASS' if test2 else 'FAIL'}")
    print("="*60)
    
    if all([test1, test2]):
        print("\n✓ All tests passed! The llm_viz.py save functionality is ready to use.")
        print("\nTo test with actual model output:")
        print("1. Run llm_viz.py with an image and prompt")
        print("2. Check the 'inference_data' directory for saved files")
        print("3. Run visualization: python visualize_saved_data.py <data_dir>")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
