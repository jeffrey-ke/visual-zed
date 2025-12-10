import json
from dataclasses import dataclass
import pdb
from math import sin, cos, radians, pi
from typing import Optional

import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torchvision import models
import jax.numpy as jnp
import jax

import dlt
from datastructs import StereoSample

@dataclass
class ModelOutputs:
    pred: torch.Tensor
    homography: Optional[torch.Tensor]

class DualStreamCNN(nn.Module):
    """! @brief Dual-stream CNN for processing stereo images and predicting offset.
    
    Siamese-like network because it has two identical processing paths for left & right images
    Core Idea: 
        1. Same feature extractor can find important visual cues (edges, textures, shapes)
        2. By using the same weights (for both images) -> network is forced to learn a consistent representation
    """
    
    def __init__(self, input_channels=3, output_dim=3):
        super(DualStreamCNN, self).__init__()
        
        #! @note Shared feature extractor for both streams
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3), # Convolutional Layer
            nn.BatchNorm2d(64), # Batch Normalization
            nn.ReLU(inplace=True), # Rectified Linear Unit
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Pooling -> Down-sampling Layer
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Final Pooling Layer -> Average the feature map to 1x1
        )
        
        """! @note Fusion and regression layers
        Once we have feature vectors for the left and right images, we need to combine them and predict the offset.
        """
        self.fusion_layers = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), #! @note Regularization Technique (randomly sets 50% of the inputs in this layer to zero)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim) #! @note `output_dim` -> (x, y, z offsets)
        )
        
    def forward(self, stereo_sample: StereoSample):
        """! @brief Method defines the actual flow of data
        
        1. Left and right images are passed into the feature extractor
        2. Ouputs are flattened from 4D tensors [Batch, Channels, H, W] to 2D tensors [Batch, Features]
        3. `torch.cat` stacks them side-by-side, creating the [512*2 = 1024] dimensional vector for each sample in the batch
        4. Combined vector -> passed through -> `fusion_layers` to produce the final prediction 
        """
        #! @note Extract features from both images
        left_features = self.feature_extractor(stereo_sample.left_img)
        right_features = self.feature_extractor(stereo_sample.right_img)
        
        #! @note Flatten features
        left_features = left_features.view(left_features.size(0), -1)
        right_features = right_features.view(right_features.size(0), -1)
        
        #! @note Concatenate features
        combined_features = torch.cat([left_features, right_features], dim=1)
        
        #! @note Predict offset
        offset = self.fusion_layers(combined_features)
        
        return ModelOutputs(offset, homography=None)

class EfficientNetDualStream(nn.Module):
    """! @brief A Dual-Stream model using a pre-trained EfficientNet-B0 as the feature extractor."""
    def __init__(self, output_dim=3, freeze_early_layers=True):
        super(EfficientNetDualStream, self).__init__()

        """! @note 1. Load a pre-trained EfficientNet-B0 model with default weights."""
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        """! @note 2. Define the feature extractor.
            .features contains the convolutional blocks.
            .avgpool performs the final pooling to create a feature vector.
        """
        self.feature_extractor = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )
        
        """! @note 3. Freeze the weights of the early convolutional blocks."""
        if freeze_early_layers:
            # EfficientNet-B0 has 8 feature blocks. Let's freeze the first 4.
            for ct, child in enumerate(self.feature_extractor[0].children()):
                if ct < 4:
                    for param in child.parameters():
                        param.requires_grad = False
        
        """! @note 4. Define the fusion and regression head.
            The output of EfficientNet-B0's feature extractor is 1280.
            We use a fusion strategy of concatenating left, right, and their difference.
        """
        fusion_input_features = 1280 * 3
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_features, 512),
            nn.BatchNorm1d(512), # Batch norm on the fused vector can help stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, stereo_sample: StereoSample):
        """! @brief Defines the forward pass of the model."""
        
        """! @note Extract features from both images using the same backbone."""
        left_features = self.feature_extractor(stereo_sample.left_img)
        right_features = self.feature_extractor(stereo_sample.right_img)
        
        """! @note Flatten the features from (B, C, 1, 1) to (B, C)."""
        left_features = left_features.view(left_features.size(0), -1)
        right_features = right_features.view(right_features.size(0), -1)
        
        """! @note Create a difference vector to explicitly model disparity."""
        diff_features = right_features - left_features
        
        """! @note Concatenate all three feature vectors for a rich input to the fusion head."""
        combined_features = torch.cat([left_features, right_features, diff_features], dim=1)
        
        """! @note Predict the offset."""
        offset = self.fusion_layers(combined_features)
        
        return offset

class GeometricServoing(nn.Module):

    def __init__(self, annotation_ndc, corner_model, dlt_model, path_to_camera_calibs, device):
        assert annotation_ndc.shape == (2,)

        super().__init__()
        self.corner_detector = corner_model
        self.annotation = annotation_ndc.to(device)
        self.K = self._load_intrinsics(path_to_camera_calibs).to(device)
        self.dlt_model = dlt_model
        self.device = device

    def forward(self, stereo_sample: StereoSample):
        assert stereo_sample.left_img.shape[0] == 1, "Batch size is not 1!!"
        left_img = stereo_sample.left_img
        img_height, img_width = 1080, 1920
        left_depth = stereo_sample.left_depth.squeeze()

        x_primes = (
                self.hom(self.corner_detector(left_img).squeeze())
        )
        xs = self.hom(
                torch.tensor(
                    [ [-1, 1], [1, 1], [1, -1], [-1, -1] ], 
                    dtype=torch.float32,
                    device=self.device
                )
        )
        homography = self.dlt_model(x_primes, xs)
        grasp_point = self.hom(self.annotation) #, shape 2,
        image_grasp_point = torch.mv(homography, grasp_point)
        ray = F.normalize(
            torch.mv(torch.linalg.inv(self.K), image_grasp_point),
            dim=0
        )
        coords = torch.clamp(
                self.dehom(image_grasp_point),
                torch.tensor([0, 0], device=self.device),
                torch.tensor([img_width - 1, img_height - 1], device=self.device)
        )
        coords = coords.int().tolist()[::-1]
        depth_at_grasp_point = left_depth[coords[0], coords[1]]
        offset = ray[None, ...] * depth_at_grasp_point

        return ModelOutputs(
                pred=self.cam2grasp_transform(offset)[..., :3, -1], 
                homography=homography,
        )

    def get_corner_detector(self):
        return self.corner_detector

    @staticmethod
    def _load_intrinsics(path_to_calibs):
        """
        Load camera intrinsics matrix K from Isaac Sim camera parameter JSON.

        Args:
            path_to_calibs: Path to camera_params_*.json file

        Returns:
            K: 3x3 intrinsics matrix as torch.Tensor
               [fx  0  cx]
               [0  fy  cy]
               [0   0   1]
        """
        with open(path_to_calibs, 'r') as f:
            params = json.load(f)

        # Extract parameters from Isaac Sim JSON format
        focal_length_mm = params['cameraFocalLength']
        aperture_width_mm, aperture_height_mm = params['cameraAperture']
        aperture_offset_x, aperture_offset_y = params['cameraApertureOffset']
        image_width, image_height = params['renderProductResolution']

        # Compute focal lengths in pixels
        # fx = f_mm * (image_width_px / sensor_width_mm)
        fx = focal_length_mm * (image_width / aperture_width_mm)
        fy = focal_length_mm * (image_height / aperture_height_mm)

        # Compute principal point (optical center) in pixels
        # Default is center of image, adjusted by aperture offset
        cx = (image_width / 2.0) + (aperture_offset_x * image_width / aperture_width_mm)
        cy = (image_height / 2.0) + (aperture_offset_y * image_height / aperture_height_mm)

        # Assemble intrinsics matrix
        K = torch.tensor([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=torch.float32)

        return K

    @staticmethod
    def hom(x: torch.Tensor):
        return torch.cat((x, torch.ones_like(x[..., -1:])), axis=-1) # type: ignore

    @staticmethod
    def dehom(x: torch.Tensor, keepdims=False):
        x = x / x[..., -1:]
        return x if keepdims else x[..., :-1]

    @staticmethod
    def cross_mat(x):
        assert len(x.shape) == 1
        return torch.cross(torch.eye(3), x[None, ...], dim=-1)

    @staticmethod
    def cam2grasp_transform(grasps_in_camera):
        assert len(grasps_in_camera.shape) == 2, "Needs to be batched!"
        assert grasps_in_camera.shape[1] == 3, "Needs to be 3-vectors"
        n_grasps, *_ = grasps_in_camera.shape
        R = (
                torch.as_tensor(
                    [
                        [0., 0., -1.],
                        [1., 0., 0.],
                        [0., -1., 0.]
                    ],
                )
                .to(grasps_in_camera)
                .reshape(-1, 3, 3)
                .expand(n_grasps, -1, -1)
        )
        T = torch.eye(4).repeat(n_grasps, 1, 1).to(grasps_in_camera)
        T[..., :3, :3] = R
        baseline = 0.063
        offsets = -torch.einsum('bij,bj->bi', R, grasps_in_camera) + torch.as_tensor([0., baseline/2, 0.0]).to(grasps_in_camera)
        assert offsets.shape == (n_grasps,3)
        T[..., :3, -1] = offsets
        assert T.shape == (n_grasps, 4, 4)
        return T

    @staticmethod
    def grasp2cam_transform(camera_in_grasp):
        assert len(camera_in_grasp.shape) == 2, "Must be batched!"
        assert camera_in_grasp.shape[1] == 3, "Must be 3-vectors"
        n_grasps, *_ = camera_in_grasp.shape
        R = (
                torch.as_tensor(
                    [
                        [0., 1., 0.],
                        [0., 0., -1.],
                        [-1., 0., 0.]
                    ],
                )
                .to(camera_in_grasp)
                .reshape(-1, 3, 3)
                .expand(n_grasps, -1, -1)
        )
        T = torch.eye(4).to(camera_in_grasp).repeat(n_grasps, 1, 1)
        T[..., :3, :3] = R
        T[..., :3, -1] = -camera_in_grasp @ R.mT
        return T



"""
utils.py:
    def hom
    def dehom
    def cross_matrix
    def normalize

annotation.py
    def get_annotation(path to img):
        im = np.array(Image.open(path to img))
        annotation = np.empty(2)
        height, width = im.shape[:2]

        def click_handler(event):
            nonlocal annotation
            x = event.xdata
            y = event.ydata
            annotation = np.array([x, y])
            plt.close()

        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.imshow(im)
        fig.canvas.mpl_connect('button_press_event', click)
        plt.show()

        return annotation

dlt.py:
    def np2jax2np(func):
        @functools.wraps
        def new_function(*args, **kwargs):
            args = [jax.asarray(arg) if isinstance(arg, np.ndarray or whatever the type is) else arg]
            # pretty sure I did this wrong...
            kwargs = { k : jax.asarray(v) if isinstance(v, np.ndarray) else k : v for (k,v) in kwargs }
            result = func(*args, **kwargs)
            return np.asarray(result)
        return new_function

    def dlt(x_primes, xs of shape 4,2):
        x_primes, xs are already jax arrays
        xprimes = utils.normalize(to mean 0, to average magnitude sqrt(2))
        xs with the same
        matrix = jnp.cat(
            [create_row(x_prime, x) for x_prime, x in zip(xprimes, xs)],
            axis=0
        )
        *_, h = jnp.linalg.svd(matrix)
        # by necessity, least squares needs a np array as input
        result = scipy.least_squares(
                            residual, 
                            np.asarray(h)
                        )
        assert result.success
        return inv(Tx_prime) @ result.x.reshape(3,3) @ Tx

    def create_row(x_prime, x):
        x_prime, x = utils.homogenize(jnp.stack((x_prime, x), axis=0))
        rhs = jnp.zeros((2, 9))
        rhs[0, :3] = x
        rhs[1, 3:6] = x
        return (utils.cross_matrix(x_prime) @ rhs)[:2, ...]

    def epsilon(X a (4,) measurement vector, h):
        x_prime, x = X[:2], X[2:]
        Ai = create_row(x_prime, x)
        assert Ai.shape == (2, 9)
        return jnp.dot(Ai, h)

    def to_measurement_vector(x_primes, xs):
        return jnp.concatenate((x_primes, xs), axis=-1) N,4

    @np2jax2np
    def residual(h, x_primes, xs):
        x_primes, xs are JAX arrs
        Xs = to_measurement_vector(x_primes, xs) becomes 4,4! We only have 4 correspondences
        assert Xs.shape == (4, 4)
        J_fun = jax.jacobian(epsilon)
        Js = jnp.array([J_fun(X, h) for X in Xs]), or if this is too slow, learn vmap, but it's only 9 parameters, it should be fine
        # shape N,2,4

        JJ_T_inv = jnp.linalg.inv(Js @ Js.tranpose(the last two dims)) # shape N,2,2
        epsilons = jnp.array([epsilon(X, h) for X in Xs])
        # shape N,2
        costs = jnp.einsum('ni,nij,nj->n',
                epsilons,
                JJ_T_inv,
                epsilons
        )
        return costs
"""


