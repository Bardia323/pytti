from pytti.LossAug import MSELoss
import torch
import sys
import math
from pytti import DEVICE, vram_usage_mode
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image
import os
from contextlib import contextmanager

infer_helper = None

@contextmanager
def temporary_change_dir(path):
    original_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_dir)

def init_AdaBins():
    global infer_helper
    if infer_helper is None:
        with vram_usage_mode('AdaBins'):
            print('Loading AdaBins...')
            ada_bins_path = './AdaBins'
            if ada_bins_path not in sys.path:
                sys.path.append(ada_bins_path)
            with temporary_change_dir(ada_bins_path):
                from infer import InferenceHelper
                infer_helper = InferenceHelper(dataset='nyu')
            print('AdaBins loaded.')

class DepthLoss(MSELoss):
    MAX_DEPTH_AREA = 500000  # Maximum area to process without resizing

    @torch.no_grad()
    def set_comp(self, pil_image):
        self.comp.set_(self.make_comp(pil_image))
        if self.use_mask and self.mask.shape[-2:] != self.comp.shape[-2:]:
            self.mask.set_(TF.resize(self.mask, self.comp.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR))

    def get_loss(self, input, img):
        depth_input = self.prepare_input(input)
        depth_map = self.compute_depth_map(depth_input)
        depth_map = F.interpolate(depth_map, self.comp.shape[-2:], mode='bilinear', align_corners=True)
        return super().get_loss(depth_map, img)

    @classmethod
    @vram_usage_mode("Depth Loss")
    @torch.no_grad()
    def make_comp(cls, pil_image, device=DEVICE):
        depth_map, _ = cls.get_depth(pil_image)
        return torch.from_numpy(depth_map).to(device)

    @staticmethod
    @torch.no_grad()
    def get_depth(pil_image):
        init_AdaBins()
        depth_input = DepthLoss.resize_image_if_needed(pil_image)
        with temporary_change_dir('./AdaBins'):
            _, depth_map = infer_helper.predict_pil(depth_input)
        # Return depth_map and a placeholder for depth_resized to match the original interface
        return depth_map, False

    @staticmethod
    @torch.no_grad()
    def compute_depth_map(tensor_input):
        # Use the model directly without changing directories
        with temporary_change_dir('./AdaBins'):
            _, depth_map = infer_helper.model(tensor_input)
        return depth_map

    @staticmethod
    def prepare_input(input_tensor):
        height, width = input_tensor.shape[-2:]
        image_area = width * height
        if image_area > DepthLoss.MAX_DEPTH_AREA:
            scale_factor = math.sqrt(DepthLoss.MAX_DEPTH_AREA / image_area)
            new_size = (int(height * scale_factor), int(width * scale_factor))
            return TF.resize(input_tensor, new_size, interpolation=TF.InterpolationMode.BILINEAR)
        else:
            return input_tensor

    @staticmethod
    def resize_image_if_needed(pil_image):
        width, height = pil_image.size
        image_area = width * height
        if image_area > DepthLoss.MAX_DEPTH_AREA:
            scale_factor = math.sqrt(DepthLoss.MAX_DEPTH_AREA / image_area)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            return pil_image.resize(new_size, Image.LANCZOS)
        else:
            return pil_image
