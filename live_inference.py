#!/usr/bin/env python3
"""
Live inference script for visual servoing with ZED camera.
Captures images, runs model predictions, and displays results in real-time.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import pyzed.sl as sl

# Import your model architecture
# Adjust this import based on where your model definition is
try:
    from model import DualStreamCNN, EfficientNetDualStream
except ImportError:
    print("WARNING: Could not import models from 'models.py'")
    print("Make sure your model definition file is in the same directory")
    print("or update the import statement")
    DualStreamCNN = None
    EfficientNetDualStream = None

from zed_wrapper import ZedWrapper, CameraError
from model_inference import VisualServoingPredictor, create_side_by_side_visualization

logger = logging.getLogger(__name__)


def run_live_inference(
    checkpoint_path: str,
    model_class,
    output_dir: Optional[Path] = None,
    save_interval: int = 10,
    display_scale: float = 100.0,
    config: dict = None,
    max_frames: Optional[int] = None,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    Run live inference on ZED camera stream.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_class: Model class to use
        output_dir: Optional directory to save visualizations
        save_interval: Save every Nth frame
        display_scale: Scale factor for visualization arrows
        config: Camera configuration
        max_frames: Maximum number of frames to process (None = infinite)
        normalize_mean: Mean values for normalization (must match training!)
        normalize_std: Std values for normalization (must match training!)
    """
    # Initialize predictor
    logger.info("Initializing model predictor...")
    predictor = VisualServoingPredictor(
        checkpoint_path=checkpoint_path,
        model_class=model_class,
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )
    
    # Setup output directory if saving
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving visualizations to: {output_dir}")
    
    frame_count = 0
    saved_count = 0
    
    try:
        with ZedWrapper(config=config) as zed:
            logger.info("Camera initialized successfully")
            
            # Validate camera
            zed.validate_camera_ready()
            logger.info("Camera validated and ready")
            
            # Get camera info
            intrinsics = zed.get_intrinsics()
            logger.info(f"Camera intrinsics: fx={intrinsics['fx']:.2f}, "
                       f"fy={intrinsics['fy']:.2f}, "
                       f"cx={intrinsics['cx']:.2f}, "
                       f"cy={intrinsics['cy']:.2f}")
            
            logger.info("\n" + "=" * 60)
            logger.info("Starting live inference...")
            logger.info("Press 'q' to quit, 's' to save current frame")
            logger.info("=" * 60 + "\n")
            
            while True:
                # Check frame limit
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached maximum frame count: {max_frames}")
                    break
                
                # Capture image
                try:
                    result = zed.capture_image()
                except CameraError as e:
                    logger.warning(f"Capture failed: {e}")
                    continue
                
                # Convert RGBA to RGB
                left_rgb = cv2.cvtColor(result.image_left, cv2.COLOR_RGBA2RGB)
                right_rgb = cv2.cvtColor(result.image_right, cv2.COLOR_RGBA2RGB)
                
                # Run prediction
                offset = predictor.predict_from_numpy(
                    left_rgb, 
                    right_rgb,
                    result.depth_map
                )
                
                # Create visualization
                vis = create_side_by_side_visualization(
                    left_rgb,
                    right_rgb,
                    offset,
                    scale=display_scale
                )
                
                # Add frame counter
                cv2.putText(
                    vis,
                    f"Frame: {frame_count}",
                    (vis.shape[1] - 150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Display
                cv2.imshow("Visual Servoing - Live Inference", vis)
                
                # Auto-save at intervals
                if output_dir and (frame_count % save_interval == 0):
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_path = output_dir / f"inference_{timestamp_str}.jpg"
                    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    saved_count += 1
                    logger.debug(f"Saved frame {frame_count} to {save_path}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s') and output_dir:
                    # Manual save
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_path = output_dir / f"inference_manual_{timestamp_str}.jpg"
                    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    logger.info(f"Manually saved frame to {save_path}")
                    saved_count += 1
                
                frame_count += 1
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames, "
                               f"latest prediction: x={offset[0]:.4f}, "
                               f"y={offset[1]:.4f}, z={offset[2]:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise
    finally:
        cv2.destroyAllWindows()
        logger.info(f"\nProcessed {frame_count} frames total")
        if output_dir:
            logger.info(f"Saved {saved_count} visualizations to {output_dir}")


def run_single_frame_inference(
    checkpoint_path: str,
    model_class,
    output_path: Optional[str] = None,
    display_scale: float = 100.0,
    config: dict = None,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    Run inference on a single captured frame.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_class: Model class to use
        output_path: Optional path to save visualization
        display_scale: Scale factor for visualization arrows
        config: Camera configuration
        normalize_mean: Mean values for normalization (must match training!)
        normalize_std: Std values for normalization (must match training!)
    """
    # Initialize predictor
    logger.info("Initializing model predictor...")
    predictor = VisualServoingPredictor(
        checkpoint_path=checkpoint_path,
        model_class=model_class,
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )
    
    try:
        with ZedWrapper(config=config) as zed:
            logger.info("Camera initialized successfully")
            zed.validate_camera_ready()
            
            # Capture single image
            logger.info("Capturing image...")
            result = zed.capture_image()
            logger.info(f"Captured at {datetime.fromtimestamp(result.timestamp)}")
            
            # Convert RGBA to RGB
            left_rgb = cv2.cvtColor(result.image_left, cv2.COLOR_RGBA2RGB)
            right_rgb = cv2.cvtColor(result.image_right, cv2.COLOR_RGBA2RGB)
            
            # Run prediction
            logger.info("Running inference...")
            offset = predictor.predict_from_numpy(
                left_rgb,
                right_rgb,
                result.depth_map
            )
            
            # Print prediction
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            print(f"  x: {offset[0]:7.4f} m")
            print(f"  y: {offset[1]:7.4f} m")
            print(f"  z: {offset[2]:7.4f} m")
            print("=" * 60 + "\n")
            
            # Create visualization
            vis = create_side_by_side_visualization(
                left_rgb,
                right_rgb,
                offset,
                scale=display_scale
            )
            
            # Save if requested
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved visualization to {output_path}")
            
            # Display
            cv2.imshow("Visual Servoing - Single Frame", vis)
            logger.info("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run visual servoing model inference on ZED camera stream"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint file (e.g., best.pth)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dualstream", "efficientnet"],
        default="dualstream",
        help="Model architecture to use (default: dualstream)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Capture and process single frame instead of live stream"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output directory for saved visualizations (live mode) or file path (single mode)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save every Nth frame in live mode (default: 10)"
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=100.0,
        help="Scale factor for arrow visualization (default: 100.0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process in live mode"
    )
    parser.add_argument(
        "--normalize-mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Normalization mean values (R G B). Must match training! Default: ImageNet values"
    )
    parser.add_argument(
        "--normalize-std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Normalization std values (R G B). Must match training! Default: ImageNet values"
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
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Select model class
    if args.model == "dualstream":
        if DualStreamCNN is None:
            logger.error("DualStreamCNN not available. Check model imports.")
            sys.exit(1)
        model_class = DualStreamCNN
    else:  # efficientnet
        if EfficientNetDualStream is None:
            logger.error("EfficientNetDualStream not available. Check model imports.")
            sys.exit(1)
        model_class = EfficientNetDualStream
    
    # Camera configuration
    config = {
        'resolution': sl.RESOLUTION.HD720,
        'depth_mode': sl.DEPTH_MODE.NEURAL,
        'units': sl.UNIT.METER,  # Use meters for offset prediction
        'min_depth': 0.05,
    }
    
    try:
        if args.single:
            # Single frame mode
            output_path = args.output if args.output else "inference_result.jpg"
            run_single_frame_inference(
                checkpoint_path=args.checkpoint,
                model_class=model_class,
                output_path=output_path,
                display_scale=args.display_scale,
                config=config,
                normalize_mean=tuple(args.normalize_mean),
                normalize_std=tuple(args.normalize_std)
            )
        else:
            # Live stream mode
            output_dir = Path(args.output) if args.output else None
            run_live_inference(
                checkpoint_path=args.checkpoint,
                model_class=model_class,
                output_dir=output_dir,
                save_interval=args.save_interval,
                display_scale=args.display_scale,
                config=config,
                max_frames=args.max_frames,
                normalize_mean=tuple(args.normalize_mean),
                normalize_std=tuple(args.normalize_std)
            )
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()