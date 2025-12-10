#!/usr/bin/env python3
"""
Test script to verify model loading and inference pipeline with real stereo images.
Loads stereo image pairs from disk and saves visualization output.
"""

import numpy as np
import cv2
import torch
from pathlib import Path

from model import DualStreamCNN, EfficientNetDualStream
from model_inference import VisualServoingPredictor, create_side_by_side_visualization


def load_stereo_pair(left_path: str, right_path: str):
    """Load a stereo image pair from disk."""
    print(f"   Loading left image: {left_path}")
    left = cv2.imread(left_path)
    if left is None:
        raise ValueError(f"Failed to load left image: {left_path}")
    
    print(f"   Loading right image: {right_path}")
    right = cv2.imread(right_path)
    if right is None:
        raise ValueError(f"Failed to load right image: {right_path}")
    
    # Convert BGR to RGB
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    
    return left_rgb, right_rgb


def test_model_architecture():
    """Test that model architectures can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)
    
    try:
        # Test DualStreamCNN
        print("\n1. Testing DualStreamCNN...")
        model1 = DualStreamCNN(input_channels=3, output_dim=3)
        print(f"   ✓ DualStreamCNN instantiated")
        print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
        
        # Test EfficientNetDualStream
        print("\n2. Testing EfficientNetDualStream...")
        model2 = EfficientNetDualStream(output_dim=3)
        print(f"   ✓ EfficientNetDualStream instantiated")
        print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_inference_pipeline(
    checkpoint_path: str,
    left_image: str,
    right_image: str,
    output_path: str = "test_inference_output.jpg",
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225),
    display_scale: float = 100.0
):
    """Test the complete inference pipeline with real stereo images."""
    print("\n" + "=" * 60)
    print("Testing Inference Pipeline")
    print("=" * 60)
    
    # Load stereo images
    print("\n1. Loading stereo image pair from files...")
    try:
        left_img, right_img = load_stereo_pair(left_image, right_image)
        print(f"   ✓ Left image shape: {left_img.shape}")
        print(f"   ✓ Right image shape: {right_img.shape}")
    except Exception as e:
        print(f"   ✗ Failed to load images: {e}")
        return False
    
    # Load model
    print(f"\n2. Loading model from checkpoint: {checkpoint_path}")
    try:
        predictor = VisualServoingPredictor(
            checkpoint_path=checkpoint_path,
            model_class=DualStreamCNN,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_mean=normalize_mean,
            normalize_std=normalize_std
        )
        print(f"   ✓ Model loaded successfully")
        print(f"   Using normalization: mean={normalize_mean}, std={normalize_std}")
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run inference
    print("\n3. Running inference...")
    try:
        offset = predictor.predict_from_numpy(left_img, right_img)
        print(f"   ✓ Inference successful")
        print(f"   Predicted offset: x={offset[0]:.4f}, y={offset[1]:.4f}, z={offset[2]:.4f}")
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create and save visualization
    print("\n4. Creating visualization...")
    try:
        vis = create_side_by_side_visualization(
            left_img,
            right_img,
            offset,
            scale=display_scale
        )
        print(f"   ✓ Visualization created: {vis.shape}")
        
        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"   ✓ Saved visualization to: {output_path}")
        
        # Try to display (skip if no display available)
        try:
            print("\n   Press any key to close the visualization...")
            cv2.imshow("Test Inference", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"   ⓘ Display not available (running headless): {type(e).__name__}")
            print(f"   ⓘ Visualization saved to {output_path} - view it there!")
        
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test inference pipeline with real stereo image pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with stereo pair
  python test_inference.py --checkpoint best.pth \\
      --left left_image.jpg \\
      --right right_image.jpg \\
      -o result.jpg
  
  # Custom visualization scale
  python test_inference.py --checkpoint best.pth \\
      --left left.jpg --right right.jpg \\
      --display-scale 200 -o output.jpg
  
  # With custom normalization
  python test_inference.py --checkpoint best.pth \\
      --left left.jpg --right right.jpg \\
      --normalize-mean 0.5 0.5 0.5 \\
      --normalize-std 0.25 0.25 0.25
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--left",
        type=str,
        required=True,
        help="Path to left image"
    )
    parser.add_argument(
        "--right",
        type=str,
        required=True,
        help="Path to right image"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="test_inference_output.jpg",
        help="Output path for visualization (default: test_inference_output.jpg)"
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=100.0,
        help="Scale factor for visualization arrows (default: 100.0)"
    )
    parser.add_argument(
        "--normalize-mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean values (R G B). Must match training!"
    )
    parser.add_argument(
        "--normalize-std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Normalization std values (R G B). Must match training!"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return False
    
    if not Path(args.left).exists():
        print(f"Error: Left image not found: {args.left}")
        return False
    
    if not Path(args.right).exists():
        print(f"Error: Right image not found: {args.right}")
        return False
    
    print("\n" + "=" * 60)
    print("VISUAL SERVOING INFERENCE TEST")
    print("=" * 60)
    
    # Test 1: Model architectures
    arch_ok = test_model_architecture()
    
    # Test 2: Inference pipeline
    if arch_ok:
        pipeline_ok = test_inference_pipeline(
            checkpoint_path=args.checkpoint,
            left_image=args.left,
            right_image=args.right,
            output_path=args.output,
            normalize_mean=tuple(args.normalize_mean),
            normalize_std=tuple(args.normalize_std),
            display_scale=args.display_scale
        )
    else:
        pipeline_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Model Architecture:   {'✓ PASS' if arch_ok else '✗ FAIL'}")
    print(f"Inference Pipeline:   {'✓ PASS' if pipeline_ok else '✗ FAIL'}")
    print("=" * 60)
    
    if arch_ok and pipeline_ok:
        print("\n✓ All tests passed!")
        print(f"✓ Processed stereo pair: {args.left}, {args.right}")
        print(f"✓ Visualization saved to: {args.output}")
        print("✓ Ready for live_inference.py with your camera.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return arch_ok and pipeline_ok


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)