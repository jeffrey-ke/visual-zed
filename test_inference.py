#!/usr/bin/env python3
"""
Test script to verify model loading and inference pipeline with stereo videos.
Processes stereo video (side-by-side or separate left/right) and saves visualization output.
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

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


def process_video(
    checkpoint_path: str,
    video_path: str,
    output_path: str,
    side_by_side: bool = True,
    left_video: str = None,
    right_video: str = None,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225),
    baseline: float = 0.063,
    apply_transform: bool = True,
    codec: str = 'mp4v',
    fps: float = None
):
    """
    Process stereo video and save output with predictions.
    
    Args:
        checkpoint_path: Path to model checkpoint
        video_path: Path to input video (if side_by_side=True)
        output_path: Path to save output video
        side_by_side: Whether input is side-by-side stereo (True) or separate videos (False)
        left_video: Path to left video (if side_by_side=False)
        right_video: Path to right video (if side_by_side=False)
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        baseline: Stereo baseline in meters
        apply_transform: Whether to apply grasp-to-camera transformation
        codec: Video codec (e.g., 'mp4v', 'avc1', 'h264')
        fps: Output framerate (None = use input framerate)
    """
    
    # Load model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        predictor = VisualServoingPredictor(
            checkpoint_path=checkpoint_path,
            model_class=DualStreamCNN,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            baseline=baseline
        )
        print(f"Model loaded successfully")
        print(f"Using normalization: mean={normalize_mean}, std={normalize_std}")
        print(f"Stereo baseline: {baseline}m")
        print(f"Apply coordinate transform: {apply_transform}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Open video(s)
    if side_by_side:
        print(f"\nOpening side-by-side video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Side-by-side: split width in half
        left_width = frame_width // 2
        output_width = frame_width  # Keep same width for visualization
        output_height = frame_height
        
        print(f"Video properties:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {input_fps}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  Left/Right split at: x={left_width}")
        
    else:
        print(f"\nOpening separate stereo videos:")
        print(f"  Left: {left_video}")
        print(f"  Right: {right_video}")
        
        cap_left = cv2.VideoCapture(left_video)
        cap_right = cv2.VideoCapture(right_video)
        
        if not cap_left.isOpened():
            print(f"Error: Could not open left video: {left_video}")
            return False
        if not cap_right.isOpened():
            print(f"Error: Could not open right video: {right_video}")
            return False
        
        # Get video properties (assume both videos have same properties)
        total_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap_left.get(cv2.CAP_PROP_FPS)
        left_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_width = left_width * 2  # Side-by-side output
        output_height = frame_height
        
        print(f"Video properties:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {input_fps}")
        print(f"  Resolution per camera: {left_width}x{frame_height}")
    
    # Set output FPS
    output_fps = fps if fps is not None else input_fps
    print(f"Output FPS: {output_fps}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        output_fps,
        (output_width, output_height)
    )
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        return False
    
    print(f"\nProcessing video...")
    print(f"Output will be saved to: {output_path}")
    
    # Process frames
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    try:
        while True:
            if side_by_side:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Split frame into left and right
                left_bgr = frame[:, :left_width, :]
                right_bgr = frame[:, left_width:, :]
                
            else:
                ret_left, left_bgr = cap_left.read()
                ret_right, right_bgr = cap_right.read()
                
                if not ret_left or not ret_right:
                    break
            
            # Convert BGR to RGB
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
            
            # Run inference
            try:
                offset = predictor.predict_from_numpy(
                    left_rgb,
                    right_rgb,
                    apply_transform=apply_transform
                )
                
                # Create visualization
                vis_rgb = create_side_by_side_visualization(
                    left_rgb,
                    right_rgb,
                    offset
                )
                
                # Convert back to BGR for video writer
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(vis_bgr)
                
                frame_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
    finally:
        pbar.close()
        
        # Release resources
        if side_by_side:
            cap.release()
        else:
            cap_left.release()
            cap_right.release()
        out.release()
    
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")
    
    return True
        

def test_inference_pipeline(
    checkpoint_path: str,
    left_image: str,
    right_image: str,
    output_path: str = "test_inference_output.jpg",
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225),
    baseline: float = 0.063,
    apply_transform: bool = True
):
    """
    Test the inference pipeline with a stereo image pair.
    
    Args:
        checkpoint_path: Path to model checkpoint
        left_image: Path to left image
        right_image: Path to right image
        output_path: Path to save visualization
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        baseline: Stereo baseline in meters
        apply_transform: Whether to apply grasp-to-camera transformation
    """
    
    try:
        left_img, right_img = load_stereo_pair(left_image, right_image)
        print(f"Left image shape: {left_img.shape}")
        print(f"Right image shape: {right_img.shape}")
    except Exception as e:
        print(f"Failed to load images: {e}")
        return False
    
    # Load model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        predictor = VisualServoingPredictor(
            checkpoint_path=checkpoint_path,
            model_class=DualStreamCNN,
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            baseline=baseline
        )
        print(f"Model loaded successfully")
        print(f"Using normalization: mean={normalize_mean}, std={normalize_std}")
        print(f"Stereo baseline: {baseline}m")
        print(f"Apply coordinate transform: {apply_transform}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run inference
    print("\nRunning inference...")
    try:
        offset = predictor.predict_from_numpy(
            left_img, 
            right_img, 
            apply_transform=apply_transform
        )
        print(f"Inference successful")
        
        if apply_transform:
            print(f"Predicted offset (camera frame):")
        else:
            print(f"Predicted offset (grasp frame):")
        print(f"  x={offset[0]:7.4f} m")
        print(f"  y={offset[1]:7.4f} m")
        print(f"  z={offset[2]:7.4f} m")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create and save visualization
    print("\nCreating visualization...")
    try: 
        vis = create_side_by_side_visualization(
            left_img,
            right_img,
            offset
        )
        print(f"Visualization created: {vis.shape}")
        
        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to: {output_path}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test inference pipeline with stereo images or videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with stereo image pair
  python test_inference.py --checkpoint best.pth \\
      --left left_image.jpg \\
      --right right_image.jpg \\
      -o result.jpg
  
  # Process side-by-side stereo video
  python test_inference.py --checkpoint best.pth \\
      --video stereo_sbs.mp4 \\
      -o output.mp4
  
  # Process separate left/right videos
  python test_inference.py --checkpoint best.pth \\
      --left-video left.mp4 \\
      --right-video right.mp4 \\
      -o output.mp4
  
  # Video with custom codec and FPS
  python test_inference.py --checkpoint best.pth \\
      --video stereo.mp4 \\
      -o output.mp4 \\
      --codec avc1 \\
      --fps 30
  
  # Without coordinate transform
  python test_inference.py --checkpoint best.pth \\
      --video stereo.mp4 \\
      --no-transform -o output.mp4
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    
    # Image inputs
    parser.add_argument(
        "--left",
        type=str,
        help="Path to left image (for image mode)"
    )
    parser.add_argument(
        "--right",
        type=str,
        help="Path to right image (for image mode)"
    )
    
    # Video inputs
    parser.add_argument(
        "--video",
        type=str,
        help="Path to side-by-side stereo video (for video mode)"
    )
    parser.add_argument(
        "--left-video",
        type=str,
        help="Path to left video (for separate video mode)"
    )
    parser.add_argument(
        "--right-video",
        type=str,
        help="Path to right video (for separate video mode)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output path for visualization (.jpg for images, .mp4 for videos)"
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
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.063,
        help="Stereo baseline in meters (default: 0.063 for ZED)"
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Skip grasp-to-camera coordinate transformation (output in grasp frame)"
    )
    
    # Video-specific arguments
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec (default: mp4v). Options: mp4v, avc1, h264, xvid"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS (default: same as input)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return False
    
    # Determine mode: image or video
    is_video_mode = args.video or (args.left_video and args.right_video)
    is_image_mode = args.left and args.right
    
    if not is_video_mode and not is_image_mode:
        print("Error: Must specify either:")
        print("  - Image mode: --left and --right")
        print("  - Video mode: --video (side-by-side) OR --left-video and --right-video")
        return False
    
    if is_video_mode and is_image_mode:
        print("Error: Cannot mix image and video inputs")
        return False
    
    if is_video_mode:
        # VIDEO MODE
        if args.video:
            # Side-by-side video
            if not Path(args.video).exists():
                print(f"Error: Video not found: {args.video}")
                return False
            
            success = process_video(
                checkpoint_path=args.checkpoint,
                video_path=args.video,
                output_path=args.output,
                side_by_side=True,
                normalize_mean=tuple(args.normalize_mean),
                normalize_std=tuple(args.normalize_std),
                baseline=args.baseline,
                apply_transform=not args.no_transform,
                codec=args.codec,
                fps=args.fps
            )
        else:
            # Separate left/right videos
            if not Path(args.left_video).exists():
                print(f"Error: Left video not found: {args.left_video}")
                return False
            if not Path(args.right_video).exists():
                print(f"Error: Right video not found: {args.right_video}")
                return False
            
            success = process_video(
                checkpoint_path=args.checkpoint,
                video_path=None,
                output_path=args.output,
                side_by_side=False,
                left_video=args.left_video,
                right_video=args.right_video,
                normalize_mean=tuple(args.normalize_mean),
                normalize_std=tuple(args.normalize_std),
                baseline=args.baseline,
                apply_transform=not args.no_transform,
                codec=args.codec,
                fps=args.fps
            )
    
    else:
        # IMAGE MODE
        if not Path(args.left).exists():
            print(f"Error: Left image not found: {args.left}")
            return False
        
        if not Path(args.right).exists():
            print(f"Error: Right image not found: {args.right}")
            return False
        
        success = test_inference_pipeline(
            checkpoint_path=args.checkpoint,
            left_image=args.left,
            right_image=args.right,
            output_path=args.output,
            normalize_mean=tuple(args.normalize_mean),
            normalize_std=tuple(args.normalize_std),
            baseline=args.baseline,
            apply_transform=not args.no_transform
        )
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)