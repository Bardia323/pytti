import math, re
from PIL import Image
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from pytti.LossAug import Loss
from pytti.Notebook import Rotoscoper
from pytti import *
import torch

class MSELoss(Loss):
    @torch.no_grad()
    def __init__(self, comp, weight=0.5, stop=-math.inf, name="direct target loss", image_shape=None, device=DEVICE):
        super().__init__(weight, stop, name)
        self.register_buffer('comp', comp.to(device, memory_format=torch.channels_last))
        if image_shape is None:
            height, width = comp.shape[-2:]
            image_shape = (width, height)
        self.image_shape = image_shape
        self.register_buffer('mask', torch.ones(1, 1, 1, 1, device=device).to(memory_format=torch.channels_last))
        self.use_mask = False
        self._cached_mask_size = None
        self._resized_mask = None

    @classmethod
    @vram_usage_mode('Loss Augs')
    @torch.no_grad()
    def TargetImage(cls, prompt_string, image_shape, pil_image=None, is_path=False, device=DEVICE):
        text, weight, stop = parse(prompt_string, r"(?<!^http)(?<!s):|:(?!/)", ['', '1', '-inf'])
        weight, mask = parse(weight, r"_", ['1', ''])
        text = text.strip()
        mask = mask.strip()
        if pil_image is None and text != '' and is_path:
            pil_image = Image.open(fetch(text)).convert("RGB")
            pil_image = pil_image.resize(image_shape, Image.LANCZOS)
            comp = cls.make_comp(pil_image, device=device)
        elif pil_image is None:
            comp = torch.zeros(1, 1, 1, 1, device=device).to(memory_format=torch.channels_last)
        else:
            pil_image = pil_image.resize(image_shape, Image.LANCZOS)
            comp = cls.make_comp(pil_image, device=device)
        if image_shape is None and pil_image is not None:
            image_shape = pil_image.size
        out = cls(comp, float(weight), float(stop), text + " (direct)", image_shape, device=device)
        out.set_mask(mask, device=device)
        return out

    @torch.no_grad()
    def set_mask(self, mask, inverted=False, device=DEVICE):
        if isinstance(mask, str) and mask != '':
            if mask[0] == '-':
                mask = mask[1:]
                inverted = True
            if mask.strip()[-4:] == '.mp4':
                r = Rotoscoper(mask, self)
                r.update(0)
                return
            mask = Image.open(fetch(mask)).convert('L')
        if isinstance(mask, Image.Image):
            with vram_usage_mode('Masks'):
                mask = TF.to_tensor(mask).unsqueeze(0).to(device=device).to(memory_format=torch.channels_last)
        if mask not in ['', None]:
            self.mask.set_(mask if not inverted else (1 - mask))
            self._cached_mask_size = None  # Reset cached mask size
            self._resized_mask = None      # Reset resized mask
        self.use_mask = mask not in ['', None]

    @classmethod
    def convert_input(cls, input, img):
        return input

    @classmethod
    def make_comp(cls, pil_image, device=DEVICE):
        out = TF.to_tensor(pil_image).unsqueeze(0).to(device=device).to(memory_format=torch.channels_last)
        return cls.convert_input(out, None)

    def set_comp(self, pil_image, device=DEVICE):
        self.comp.set_(type(self).make_comp(pil_image, device=device))

    def get_loss(self, input, img):
        input = type(self).convert_input(input, img)
        if self.comp.device != input.device or self.comp.dtype != input.dtype:
            self.comp = self.comp.to(input.device, input.dtype)
        if self.use_mask:
            if self._cached_mask_size != input.shape[-2:]:
                with torch.no_grad():
                    self._resized_mask = TF.resize(
                        self.mask, input.shape[-2:], interpolation=TF.InterpolationMode.NEAREST
                    ).to(input.device, input.dtype)
                    self._cached_mask_size = input.shape[-2:]
            return F.mse_loss(input * self._resized_mask, self.comp * self._resized_mask)
        else:
            return F.mse_loss(input, self.comp)
