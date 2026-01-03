import numpy as np
import pdb
import cv2
import imageio.v3 as iio

from . import zed_wrapper
from model import GeometricServoing
from datastructs import Intrinsics, CaptureResult, CaptureSequence
from dataset import StereoImageDataset
import diagnostics
from mathutils import hom as make_rgba

class VideoZed(zed_wrapper.ZedWrapper):
    def __init__(self, video_path, intrinsics_json="calibs.json"):
        self.path = video_path
        self.frame_it = enumerate(iio.imiter(self.path))
        self.intrinsics_json = intrinsics_json
        diagnostics.fprint(f"Using video zed at path {video_path}")

    def __enter__(self): 
        return self

    def __exit__(self, *args):
        pass

    def get_K(self):
        return GeometricServoing._load_intrinsics(self.intrinsics_json).numpy()

    def get_intrinsics(self) -> Intrinsics:
        return Intrinsics(fx=K[0,0].item(), fy=K[1,1].item(), cx=K[0,-1].item(), cy=K[1,-1].item())

    def capture_image(self) -> CaptureResult:
        try:
            i, frame = next(self.frame_it)
        except StopIteration:
            self.frame_it = enumerate(iio.imiter(self.path))
            i, frame = next(self.frame_it)

        height, width, _ = frame.shape
        left, right = (
                cv2.resize(frame[:, :width//2, :], (1920, 1080), interpolation=cv2.INTER_AREA),
                cv2.resize(frame[:, width//2:, :], (1920, 1080), interpolation=cv2.INTER_AREA)
        )

        depth = np.empty((1080, 1920), dtype=np.float32)
        depth_img = np.empty((1080, 1920, 3), dtype=np.uint8)
        return CaptureResult(i, 0, left, right, depth_img, depth, True, None) #type: ignore

class DatasetZed(zed_wrapper.ZedWrapper):
    def __init__(self, dataset_path, intrinsics_json="calibs.json"):
        self.dataset = StereoImageDataset(dataset_path, isIsaacSim=True)
        self.intrinsics_json = intrinsics_json
        diagnostics.fprint(f"Using dataset zed at path {dataset_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_K(self):
        return GeometricServoing._load_intrinsics(self.intrinsics_json).numpy()

    def get_intrinsics(self) -> Intrinsics:
        K = GeometricServoing._load_intrinsics(self.intrinsics_json)
        return Intrinsics(fx=K[0,0].item(), fy=K[1,1].item(), cx=K[0,-1].item(), cy=K[1,-1].item())

    def capture_image(self) -> CaptureResult:
        selected = np.random.choice(self.dataset.samples)
        left_img = cv2.cvtColor(cv2.imread(selected['img_left_path']), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(cv2.imread(selected['img_right_path']), cv2.COLOR_BGR2RGB)
        depth = np.load(selected['depth_left_path'])
        depth_rescaled = (
                (
                    (depth - depth.min()) / (depth.max() - depth.min()) * 255
                )
                .astype(np.uint8)
        )
        depth_img =  cv2.applyColorMap(
            depth_rescaled,
            cv2.COLORMAP_JET
        )
        return CaptureResult(0, 0.0, left_img, right_img, depth, depth_img, True, None) # type: ignore

    def capture_sequence(self, num_captures: int, skip_on_failure: bool = False, delay_between_captures: float = 0) -> CaptureSequence:
        return None # type: ignore

if __name__ == "__main__":

    diagnostics.start("diagnostics/throwaway")
    with DatasetZed("datasets/cornerset") as zm:
        capt = zm.capture_image()
        written = cv2.imwrite("file.png", capt.image_left) # type: ignore
