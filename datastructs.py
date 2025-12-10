from dataclasses import dataclass, fields
from typing import List, Optional, Callable

import numpy as np
import torch
import torchvision.transforms.v2 as v2

@dataclass
class StereoSample:
    left_img: np.ndarray | torch.Tensor
    right_img: np.ndarray | torch.Tensor
    left_depth: np.ndarray | torch.Tensor
    right_depth: np.ndarray | torch.Tensor
    offset: np.ndarray | torch.Tensor

    orig_left_img: np.ndarray
    orig_right_img: np.ndarray

    @staticmethod
    def collate(samples: List["StereoSample"]):
        return StereoSample(
            left_img = torch.stack([torch.as_tensor(s.left_img) for s in samples]),
            right_img = torch.stack([torch.as_tensor(s.right_img) for s in samples]),
            left_depth = torch.stack([torch.as_tensor(s.left_depth) for s in samples]),
            right_depth = torch.stack([torch.as_tensor(s.right_depth) for s in samples]),
            offset = torch.stack([torch.as_tensor(s.offset) for s in samples]),
            orig_left_img=np.stack([s.orig_left_img for s in samples]),
            orig_right_img=np.stack([s.orig_right_img for s in samples]),
        )

    def transform(self, transforms: v2.Compose | Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor]):
        not_batched = len(self.left_img.shape) == 3
        self.left_img = transforms(self.left_img) if not_batched else torch.stack([transforms(img) for img in self.left_img])
        self.right_img = transforms(self.right_img) if not_batched else torch.stack([transforms(img) for img in self.right_img])

    def move_to(self, device: str):
        # diagnostics.fprint(f"Moving StereoSample fields to device {device}")
        for field in fields(self):
            # diagnostics.fprint(f"Found attribute {field.name} of type {field.type}")
            if isinstance(getattr(self, field.name), torch.Tensor):
                # diagnostics.fprint(f"Moving attribute {field}")
                setattr(self, field.name, getattr(self, field.name).to(device))
@dataclass
class GraphInfo:
    graph_buf: np.ndarray
    name: str

@dataclass
class LossInfo:
    eval_idx: int
    distance: float
    deviation: np.ndarray
    loss: float
    predicted: np.ndarray
    gt: np.ndarray
    debug_left_img: np.ndarray
    debug_right_img: np.ndarray
    pred_corners: Optional[np.ndarray]
    homography: np.ndarray

@dataclass
class EvalInfo:
    epoch: int
    avg_val_loss: float
    val_losses: np.ndarray
    performance_graph: np.ndarray
    abnormals: List[LossInfo]
    loss_infos: List[LossInfo]
    canonical_grasp_point: np.ndarray
    canonical_test_points: np.ndarray
    is_geometry_model: bool
    tag: str

