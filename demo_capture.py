#!/usr/bin/env python3
"""
Demo script for ZED camera image capture using ZedWrapper.

This script demonstrates:
1. Opening the camera with context management
2. Running pre-flight validation
3. Capturing single images
4. Capturing a sequence of images
5. Saving images to disk
"""

import os
import logging
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from zed_wrapper import ZedWrapper, CameraError, UnrecoverableCameraError

logger = logging.getLogger(__name__)


def save_capture_result(result, output_dir: Path, prefix: str = "capture"):
    """
    Save a capture result to disk.
    
    Args:
        result: CaptureResult from ZedWrapper
        output_dir: Directory to save images
        prefix: Filename prefix
    
    Returns:
        Dict of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.fromtimestamp(result.timestamp).strftime("%Y%m%d_%H%M%S_%f")
    saved_files = {}
    
    # Save left image (RGBA -> BGR for OpenCV)
    if result.image_left is not None:
        left_path = output_dir / f"{prefix}_{timestamp_str}_left.png"
        left_bgr = cv2.cvtColor(result.image_left, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(left_path), left_bgr)
        saved_files['left'] = left_path
        logger.info(f"Saved left image: {left_path}")
    
    # Save right image (RGBA -> BGR for OpenCV)
    if result.image_right is not None:
        right_path = output_dir / f"{prefix}_{timestamp_str}_right.png"
        right_bgr = cv2.cvtColor(result.image_right, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(right_path), right_bgr)
        saved_files['right'] = right_path
        logger.info(f"Saved right image: {right_path}")
    
    # Save depth visualization
    if result.depth_image is not None:
        depth_vis_path = output_dir / f"{prefix}_{timestamp_str}_depth_vis.png"
        depth_bgr = cv2.cvtColor(result.depth_image, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(depth_vis_path), depth_bgr)
        saved_files['depth_vis'] = depth_vis_path
        logger.info(f"Saved depth visualization: {depth_vis_path}")
    
    # Save raw depth map as numpy file
    if result.depth_map is not None:
        depth_npy_path = output_dir / f"{prefix}_{timestamp_str}_depth.npy"
        np.save(str(depth_npy_path), result.depth_map)
        saved_files['depth_npy'] = depth_npy_path
        logger.info(f"Saved depth map: {depth_npy_path}")
    
    return saved_files


def demo_single_capture(output_dir: Path, config: dict = None):
    """
    Demonstrate capturing a single image.
    
    Args:
        output_dir: Directory to save output
        config: Optional camera configuration
    """
    print("\n" + "=" * 60)
    print("DEMO: Single Image Capture")
    print("=" * 60)
    
    try:
        with ZedWrapper(config=config) as zed:
            # Run pre-flight checks
            print("\nRunning pre-flight validation...")
            zed.validate_camera_ready()
            print("✓ Camera validated and ready")
            
            # Get camera info
            intrinsics = zed.get_intrinsics()
            print(f"\nCamera intrinsics:")
            print(f"  fx: {intrinsics.fx:.2f}")
            print(f"  fy: {intrinsics.fy:.2f}")
            print(f"  cx: {intrinsics.cx:.2f}")
            print(f"  cy: {intrinsics.cy:.2f}")
            
            # Capture single image
            print("\nCapturing image...")
            result = zed.capture_image()
            
            print(f"✓ Captured at {datetime.fromtimestamp(result.timestamp)}")
            print(f"  Left image shape: {result.image_left.shape}")
            print(f"  Right image shape: {result.image_right.shape}")
            print(f"  Depth map shape: {result.depth_map.shape}")
            
            # Save images
            saved = save_capture_result(result, output_dir / "single", prefix="single")
            print(f"\n✓ Saved {len(saved)} files to {output_dir / 'single'}")
            
    except CameraError as e:
        logger.error(f"Camera error: {e}")
        raise


def demo_sequence_capture(output_dir: Path, num_captures: int = 5, delay: float = 1.0, config: dict = None):
    """
    Demonstrate capturing a sequence of images.
    
    Args:
        output_dir: Directory to save output
        num_captures: Number of images to capture
        delay: Delay between captures in seconds
        config: Optional camera configuration
    """
    print("\n" + "=" * 60)
    print(f"DEMO: Sequence Capture ({num_captures} images)")
    print("=" * 60)
    
    try:
        with ZedWrapper(config=config) as zed:
            # Run pre-flight checks
            print("\nRunning pre-flight validation...")
            zed.validate_camera_ready()
            print("✓ Camera validated and ready")
            
            # Capture sequence
            print(f"\nStarting capture sequence ({num_captures} images, {delay}s delay)...")
            sequence = zed.capture_sequence(
                num_captures=num_captures,
                skip_on_failure=True,
                delay_between_captures=delay
            )
            
            # Report results
            completed, total = sequence.get_progress()
            print(f"\n✓ Sequence complete: {completed}/{total} successful")
            
            if sequence.failed_captures:
                print(f"  Failed captures: {sequence.failed_captures}")
            
            # Save all captures
            seq_output_dir = output_dir / "sequence"
            for idx, result in sequence.completed_captures.items():
                save_capture_result(result, seq_output_dir, prefix=f"seq_{idx:03d}")
            
            print(f"\n✓ Saved captures to {seq_output_dir}")
            
    except UnrecoverableCameraError as e:
        logger.error(f"Unrecoverable camera error: {e}")
        raise
    except CameraError as e:
        logger.error(f"Camera error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for ZED camera capture using ZedWrapper"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./captures",
        help="Output directory for captured images (default: ./captures)"
    )
    parser.add_argument(
        "-n", "--num-captures",
        type=int,
        default=5,
        help="Number of images to capture in sequence mode (default: 5)"
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=1.0,
        help="Delay between captures in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only run single capture demo"
    )
    parser.add_argument(
        "--sequence-only",
        action="store_true",
        help="Only run sequence capture demo"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Camera configuration
    config = {
        'resolution': sl.RESOLUTION.HD720,
        'depth_mode': sl.DEPTH_MODE.NEURAL,
        'units': sl.UNIT.MILLIMETER,
        'min_depth': 0.05,
    }
    
    output_dir = Path(args.output)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    try:
        if args.sequence_only:
            demo_sequence_capture(output_dir, args.num_captures, args.delay, config)
        elif args.single_only:
            demo_single_capture(output_dir, config)
        else:
            # Run both demos
            demo_single_capture(output_dir, config)
            demo_sequence_capture(output_dir, args.num_captures, args.delay, config)
        
        print("\n" + "=" * 60)
        print("✓ Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
