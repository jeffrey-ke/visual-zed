#!/usr/bin/env python3
"""
Model inference module for visual servoing with ZED camera.
Loads trained DualStreamCNN and runs predictions on stereo images.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import logging

from datastructs import StereoSample

logger = logging.getLogger(__name__)


class VisualServoingPredictor:
    """Wrapper for loading and running visual servoing model predictions."""
    
    def __init__(
        self, 
        checkpoint_path: str,
        model_class,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_size: Tuple[int, int] = (224, 224),  # (H, W) - must match training!
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet defaults
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)    # ImageNet defaults
    ):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pth file)
            model_class: The model class to instantiate (e.g., DualStreamCNN)
            device: Device to run inference on
            input_size: Expected input size (H, W) for the model
            normalize_mean: Mean values for normalization (should match training)
            normalize_std: Std values for normalization (should match training)
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path, model_class)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Using normalization - mean: {normalize_mean}, std: {normalize_std}")
        
    def _load_model(self, checkpoint_path: str, model_class) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        model = model_class(input_channels=3, output_dim=3)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def preprocess_images(
        self, 
        left_img: np.ndarray, 
        right_img: np.ndarray,
        left_depth: Optional[np.ndarray] = None
    ) -> StereoSample:
        """
        Preprocess images for model input.
        
        This preprocessing MUST match the training pipeline:
        1. Resize to expected input size
        2. Convert to float32 and scale to [0, 1]
        3. Convert from HWC to CHW format
        4. Normalize using mean and std (ImageNet or custom)
        5. Add batch dimension
        
        Args:
            left_img: Left RGB image (H, W, 3) in range [0, 255]
            right_img: Right RGB image (H, W, 3) in range [0, 255]
            left_depth: Optional left depth map (H, W)
            
        Returns:
            StereoSample ready for model input
        """
        # Store original images for visualization
        orig_left = left_img.copy()
        orig_right = right_img.copy()
        
        # Resize images to expected input size
        left_resized = cv2.resize(left_img, (self.input_size[1], self.input_size[0]))
        right_resized = cv2.resize(right_img, (self.input_size[1], self.input_size[0]))
        
        # Convert to float and scale to [0, 1]
        # This matches: v2.ToDtype(torch.float32, scale=True)
        left_float = left_resized.astype(np.float32) / 255.0
        right_float = right_resized.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format
        left_tensor = torch.from_numpy(left_float).permute(2, 0, 1)
        right_tensor = torch.from_numpy(right_float).permute(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        left_tensor = left_tensor.unsqueeze(0)
        right_tensor = right_tensor.unsqueeze(0)
        
        # Apply normalization (CRITICAL: must match training!)
        # This matches: v2.Normalize(mean=..., std=...)
        left_tensor = (left_tensor - self.normalize_mean.to(left_tensor.device)) / self.normalize_std.to(left_tensor.device)
        right_tensor = (right_tensor - self.normalize_mean.to(right_tensor.device)) / self.normalize_std.to(right_tensor.device)
        
        # Process depth if available
        if left_depth is not None:
            depth_resized = cv2.resize(left_depth, (self.input_size[1], self.input_size[0]))
            left_depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)  # (1, H, W)
            right_depth_tensor = left_depth_tensor.clone()  # Use same depth for both (stereo depth)
        else:
            # Create dummy depth tensors (required by StereoSample)
            left_depth_tensor = torch.zeros(1, self.input_size[0], self.input_size[1])
            right_depth_tensor = torch.zeros(1, self.input_size[0], self.input_size[1])
        
        # Create dummy offset (not used during inference, but required by dataclass)
        dummy_offset = torch.zeros(1, 3)
        
        # Create StereoSample with all required fields
        sample = StereoSample(
            left_img=left_tensor,
            right_img=right_tensor,
            left_depth=left_depth_tensor,
            right_depth=right_depth_tensor,
            offset=dummy_offset,
            orig_left_img=orig_left,
            orig_right_img=orig_right
        )
        
        # Move tensors to device
        sample.move_to(str(self.device))
        
        return sample
    
    @torch.no_grad()
    def predict(self, stereo_sample: StereoSample) -> np.ndarray:
        """
        Run model prediction.
        
        Args:
            stereo_sample: Preprocessed stereo sample
            
        Returns:
            Predicted offset as numpy array of shape (3,) with [x, y, z]
        """
        # Run inference
        output = self.model(stereo_sample)
        
        # Handle different output formats
        if hasattr(output, 'pred'):
            # ModelOutputs dataclass (from GeometricServoing)
            prediction = output.pred
        else:
            # Direct tensor output (from DualStreamCNN/EfficientNet)
            prediction = output
        
        # Convert to numpy and remove batch dimension
        # Output shape is (1, 3), we want (3,)
        offset = prediction.squeeze(0).cpu().numpy()
        
        return offset
    
    def predict_from_numpy(
        self, 
        left_img: np.ndarray, 
        right_img: np.ndarray,
        left_depth: Optional[np.ndarray] = None,
        save_output: Optional[str] = None,
        display_scale: float = 100.0
    ) -> np.ndarray:
        """
        Convenience method for prediction from numpy arrays.
        
        Args:
            left_img: Left RGB image (H, W, 3) in range [0, 255]
            right_img: Right RGB image (H, W, 3) in range [0, 255]
            left_depth: Optional left depth map (H, W)
            save_output: Optional path to save visualization (e.g., 'result.jpg')
            display_scale: Scale factor for visualization arrows
            
        Returns:
            Predicted offset as numpy array of shape (3,) with [x, y, z]
        """
        stereo_sample = self.preprocess_images(left_img, right_img, left_depth)
        offset = self.predict(stereo_sample)
        
        # Save visualization if requested
        if save_output is not None:
            vis = create_side_by_side_visualization(
                left_img,
                right_img,
                offset
            )
            # Convert RGB to BGR for cv2.imwrite
            cv2.imwrite(save_output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to {save_output}")
        
        return offset
    
    def predict_and_save_batch(
        self,
        stereo_pairs: list,
        output_dir: str,
        display_scale: float = 100.0,
        prefix: str = "inference"
    ) -> list:
        """
        Run predictions on a batch of stereo pairs and save visualizations.
        
        Args:
            stereo_pairs: List of tuples (left_img, right_img, left_depth_optional)
            output_dir: Directory to save visualization images
            display_scale: Scale factor for visualization arrows
            prefix: Prefix for output filenames
            
        Returns:
            List of predicted offsets
        """
        from datetime import datetime
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        offsets = []
        for i, pair in enumerate(stereo_pairs):
            if len(pair) == 2:
                left_img, right_img = pair
                left_depth = None
            else:
                left_img, right_img, left_depth = pair
            
            # Run prediction
            offset = self.predict_from_numpy(left_img, right_img, left_depth)
            offsets.append(offset)
            
            # Create visualization
            vis = create_side_by_side_visualization(
                left_img,
                right_img,
                offset
            )
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = output_path / f"{prefix}_{i:04d}_{timestamp}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(stereo_pairs)} images")
        
        logger.info(f"Saved {len(stereo_pairs)} visualizations to {output_dir}")
        return offsets


def create_camera_matrix(image_width: int, image_height: int, focal_length_mm: float = 2.8) -> np.ndarray:
    """
    Create camera intrinsic matrix with hardcoded focal length.
    
    Camera parameters:
        - Focal length: 2.8mm (hardcoded)
        - Principal point: Image center (cx = width/2, cy = height/2)
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels  
        focal_length_mm: Focal length in millimeters (default: 2.8mm)
        
    Returns:
        3x3 camera intrinsic matrix K
    """
    # For a camera with 2.8mm focal length, we need to convert to pixels
    # Assuming typical sensor size, fx and fy in pixels ≈ focal_length_mm * image_width
    # For ZED-like cameras with 2.8mm lens, typical fx ≈ 700 pixels for 1920x1080
    fx = fy = focal_length_mm * image_width / (2.8 * 1920 / 700)  # Scale appropriately
    
    # Principal point at image center
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    return K


def project_3d_to_2d(point_3d: np.ndarray, camera_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Project a 3D point in camera frame to 2D image coordinates.
    
    Camera frame:
        X: right
        Y: down
        Z: forward (depth)
    
    Args:
        point_3d: 3D point [x, y, z] in meters
        camera_matrix: 3x3 camera intrinsic matrix K
        
    Returns:
        (u, v): Pixel coordinates
    """
    x, y, z = point_3d
    
    # Ensure positive depth
    if z <= 0:
        z = 0.01
    
    # Project: [u, v, w]^T = K * [X, Y, Z]^T
    point_2d_h = camera_matrix @ np.array([x, y, z])
    
    # Convert to pixel coordinates
    u = int(point_2d_h[0] / point_2d_h[2])
    v = int(point_2d_h[1] / point_2d_h[2])
    
    return u, v


def draw_prediction_overlay(
    image: np.ndarray,
    offset: np.ndarray,
    origin: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Draw prediction overlay on image with projected 3D point.
    
    Projects the 3D offset to image plane and visualizes it.
    
    Args:
        image: Input image (H, W, 3)
        offset: Predicted 3D offset [x, y, z] in meters
        origin: Optional origin pixel (default: image center)
        
    Returns:
        Image with overlay
    """
    img_overlay = image.copy()
    h, w = image.shape[:2]
    
    # Get origin
    if origin is None:
        origin = (w // 2, h // 2)
    
    # Create camera matrix based on image size
    # Standard camera parameters: fx=fy=2.8mm, cx=width/2, cy=height/2
    K = create_camera_matrix(w, h, focal_length_mm=2.8)
    
    # Draw text with prediction values
    text_lines = [
        f"Prediction (meters):",
        f"  x: {offset[0]:7.4f}",
        f"  y: {offset[1]:7.4f}",
        f"  z: {offset[2]:7.4f}",
    ]
    
    y_offset_text = 30
    for i, line in enumerate(text_lines):
        y_pos = y_offset_text + i * 25
        # Draw text background
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_overlay, (10, y_pos - 20), (20 + text_w, y_pos + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(
            img_overlay, line, (15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    
    # Project the 3D offset point to 2D image plane
    # Assume the offset is relative to a point 1 meter in front of camera
    origin_3d = np.array([0.0, 0.0, 1.0])  # 1 meter ahead
    target_3d = origin_3d + offset
    
    # Project target point to image
    try:
        target_2d = project_3d_to_2d(target_3d, K)
        
        # Clamp to image bounds
        target_2d_clamped = (
            max(0, min(w - 1, target_2d[0])),
            max(0, min(h - 1, target_2d[1]))
        )
        
        # Draw arrow from origin to projected target
        cv2.arrowedLine(
            img_overlay,
            origin,
            target_2d_clamped,
            (0, 255, 255),  # Yellow
            3,
            tipLength=0.2
        )
        
        # Draw origin point (current position)
        cv2.circle(img_overlay, origin, 7, (0, 0, 255), -1)  # Red
        
        # Draw target point (predicted position)
        cv2.circle(img_overlay, target_2d_clamped, 7, (0, 255, 0), -1)  # Green
        
        # Add label for target point
        label = f"Target"
        cv2.putText(
            img_overlay, label,
            (target_2d_clamped[0] + 10, target_2d_clamped[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        
    except Exception as e:
        logger.warning(f"Failed to project point: {e}")
        # Fallback: just draw origin
        cv2.circle(img_overlay, origin, 7, (0, 0, 255), -1)
    
    return img_overlay


def create_side_by_side_visualization(
    left_img: np.ndarray,
    right_img: np.ndarray,
    offset: np.ndarray
) -> np.ndarray:
    """
    Create side-by-side visualization with prediction overlay.
    
    Args:
        left_img: Left RGB image
        right_img: Right RGB image
        offset: Predicted 3D offset [x, y, z] in meters
        
    Returns:
        Side-by-side visualization image
    """
    # Add overlay to left image
    left_with_overlay = draw_prediction_overlay(left_img, offset)
    
    # Concatenate images horizontally
    vis = np.hstack([left_with_overlay, right_img])
    
    # Add title
    title = "Visual Servoing Prediction - Left (with overlay) | Right"
    cv2.putText(
        vis, title, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    return vis