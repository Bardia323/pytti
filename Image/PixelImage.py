from pytti import *
from pytti.Image import DifferentiableImage
from pytti.LossAug import HSVLoss
from pytti.ImageGuide import DirectImageGuide
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

# Compile frequently called function with TorchScript for speed
@torch.jit.script
def break_tensor(tensor):
    floors = tensor.floor().long()
    ceils = tensor.ceil().long()
    rounds = tensor.round().long()
    fracs = tensor - floors
    return floors, ceils, rounds, fracs

class PalletLoss(nn.Module):
    def __init__(self, n_pallets, weight=0.15, device=DEVICE):
        super().__init__()
        self.n_pallets = n_pallets
        self.register_buffer('weight', torch.as_tensor(weight, device=device))

    def forward(self, input):
        if isinstance(input, PixelImage):
            tensor = input.tensor.movedim(0, -1).contiguous().view(-1, self.n_pallets)
            tensor = F.softmax(tensor, dim=-1)
            N, n = tensor.shape
            mu = tensor.mean(dim=0, keepdim=True)
            epsilon = 1e-8  # Small value to prevent division by zero
            sigma = tensor.std(dim=0, keepdim=True) + epsilon
            tensor_centered = tensor - mu  # Use a new variable to avoid in-place operations
            # SVD
            S = (tensor_centered.transpose(0, 1) @ tensor_centered).div(sigma * sigma.transpose(0, 1) * N)
            # Minimize correlation (anticorrelate palettes)
            S = S - torch.diag(S.diagonal())
            loss_raw = S.mean()
            # Maximize variance within each palette
            loss_raw = loss_raw + sigma.mul(N).pow(-1).mean()
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=DEVICE):
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return "Palette normalization"

class HdrLoss(nn.Module):
    def __init__(self, pallet_size, n_pallets, gamma=2.5, weight=0.15, device=DEVICE):
        super().__init__()
        self.register_buffer('comp', torch.linspace(0, 1, pallet_size).pow(gamma).view(pallet_size, 1).repeat(1, n_pallets).to(device))
        self.register_buffer('weight', torch.as_tensor(weight, device=device))

    def forward(self, input):
        if isinstance(input, PixelImage):
            pallet = input.sort_pallet()
            magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
            color_norms = torch.linalg.vector_norm(pallet * magic_color.sqrt(), dim=-1)
            loss_raw = F.mse_loss(color_norms, self.comp)
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=DEVICE):
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return "HDR normalization"

def get_closest_color(a, b):
    """
    a: h1 x w1 x 3 PyTorch tensor
    b: h2 x w2 x 3 PyTorch tensor
    Returns: h1 x w1 x 3 PyTorch tensor containing the nearest color in b to the corresponding pixels in a
    """
    # Flatten and normalize 'a' and 'b'
    a_flat = a.view(-1, 3)
    b_flat = b.view(-1, 3)
    # Compute distances efficiently using matrix operations
    dists = torch.cdist(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).squeeze(0)
    # Find the indices of the closest colors
    indices = torch.argmin(dists, dim=1)
    # Gather the closest colors
    closest_colors = b_flat[indices]
    # Reshape to original image shape
    return closest_colors.view(a.shape)

class PixelImage(DifferentiableImage):
    """
    Differentiable image format for pixel art images.
    """
    @vram_usage_mode('Limited Palette Image')
    def __init__(self, width, height, scale, pallet_size, n_pallets, gamma=1, hdr_weight=0.5, norm_weight=0.1, device=DEVICE):
        super().__init__(width * scale, height * scale)
        self.pallet_inertia = 2
        pallet = torch.linspace(0, self.pallet_inertia, pallet_size).pow(gamma).view(pallet_size, 1, 1).repeat(1, n_pallets, 3)
        self.pallet = nn.Parameter(pallet.to(device, dtype=torch.float32))  # Use float32 for stability
        self.pallet_size = pallet_size
        self.n_pallets = n_pallets
        self.value = nn.Parameter(torch.zeros(height, width, dtype=torch.float32, device=device))
        self.tensor = nn.Parameter(torch.zeros(n_pallets, height, width, dtype=torch.float32, device=device))
        self.output_axes = ('n', 's', 'y', 'x')
        self.latent_strength = 0.1
        self.scale = scale
        self.hdr_loss = HdrLoss(pallet_size, n_pallets, gamma, hdr_weight) if hdr_weight != 0 else None
        self.loss = PalletLoss(n_pallets, norm_weight)
        self.register_buffer('pallet_target', torch.empty_like(self.pallet))
        self.use_pallet_target = False

    def clone(self):
        width, height = self.image_shape
        dummy = PixelImage(width // self.scale, height // self.scale, self.scale, self.pallet_size, self.n_pallets,
                           hdr_weight=0 if self.hdr_loss is None else float(self.hdr_loss.weight),
                           norm_weight=float(self.loss.weight))
        with torch.no_grad():
            dummy.value.copy_(self.value)
            dummy.tensor.copy_(self.tensor)
            dummy.pallet.copy_(self.pallet)
            dummy.pallet_target.copy_(self.pallet_target)
            dummy.use_pallet_target = self.use_pallet_target
        return dummy

    def set_pallet_target(self, pil_image):
        if pil_image is None:
            self.use_pallet_target = False
            return
        dummy = self.clone()
        dummy.use_pallet_target = False
        dummy.encode_image(pil_image)
        with torch.no_grad():
            self.pallet_target.copy_(dummy.sort_pallet())
            self.pallet.copy_(self.pallet_target)
            self.use_pallet_target = True

    @torch.no_grad()
    def lock_pallet(self, lock=True):
        if lock:
            self.pallet_target.copy_(self.sort_pallet())
        self.use_pallet_target = lock

    def image_loss(self):
        return [x for x in [self.hdr_loss, self.loss] if x is not None]

    def sort_pallet(self):
        if self.use_pallet_target:
            return self.pallet_target
        pallet = (self.pallet / self.pallet_inertia).clamp(0, 1)
        # Calculate color norms for sorting
        magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
        color_norms = pallet.square().mul(magic_color).sum(dim=-1)
        # Optimized sorting using torch.gather
        pallet_indices = color_norms.argsort(dim=0)
        sorted_pallet = torch.gather(pallet, 0, pallet_indices.unsqueeze(-1).expand(-1, self.n_pallets, 3))
        return sorted_pallet

    def get_image_tensor(self):
        return torch.cat([self.value.unsqueeze(0), self.tensor])

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        self.value.copy_(tensor[0])
        self.tensor.copy_(tensor[1:])

    def decode_tensor(self):
        width, height = self.image_shape
        pallet = self.sort_pallet()

        # Brightness values of pixels
        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        pallet_weights = self.tensor.movedim(0, 2)
        pallet_indices = pallet_weights.argmax(dim=2)
        pallets = F.one_hot(pallet_indices, num_classes=self.n_pallets).float()

        pallet_weights = F.softmax(pallet_weights, dim=2).unsqueeze(-1)
        pallets = pallets.unsqueeze(-1)

        colors_disc = pallet[value_rounds]
        colors_disc = (colors_disc * pallets).sum(dim=2)
        colors_disc = F.interpolate(
            colors_disc.permute(2, 0, 1).unsqueeze(0).contiguous(),
            size=(height, width),
            mode='nearest'
        )

        colors_cont = pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        colors_cont = F.interpolate(
            colors_cont.permute(2, 0, 1).unsqueeze(0).contiguous(),
            size=(height, width),
            mode='nearest'
        )

        return replace_grad(colors_disc, colors_cont * 0.5 + colors_disc * 0.5)

    @torch.no_grad()
    def render_value_image(self):
        width, height = self.image_shape
        values = self.value.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 3)
        array = (values.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_pallet(self):
        pallet = self.sort_pallet()
        width, height = self.n_pallets * 16, self.pallet_size * 32
        array = (pallet.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_channel(self, pallet_i):
        width, height = self.image_shape
        pallet = self.sort_pallet()
        # Create a mask to highlight the selected palette index
        mask = torch.ones_like(pallet)
        mask[:, :pallet_i, :] = 0.5
        mask[:, pallet_i + 1:, :] = 0.5
        pallet = pallet * mask

        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        pallet_weights = self.tensor.movedim(0, 2)
        pallet_weights = F.softmax(pallet_weights, dim=2).unsqueeze(-1)

        colors_cont = pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        colors_cont = F.interpolate(
            colors_cont.permute(2, 0, 1).unsqueeze(0).contiguous(),
            size=(height, width),
            mode='nearest'
        )

        tensor = colors_cont.squeeze(0).permute(1, 2, 0)
        array = (tensor.mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8))
        return Image.fromarray(array)

    @torch.no_grad()
    def update(self):
        self.pallet.copy_(self.pallet.clamp(0, self.pallet_inertia))
        self.value.copy_(self.value.clamp(0, 1))
        self.tensor.copy_(self.tensor.clamp(0, float('inf')))

    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        width, height = self.image_shape

        scale = self.scale
        color_ref = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        color_ref = TF.to_tensor(color_ref).to(device)
        with torch.no_grad():
            # Calculate grayscale values
            magic_color = self.pallet.new_tensor([[[0.299]], [[0.587]], [[0.114]]])
            value_ref = torch.linalg.vector_norm(color_ref * magic_color.sqrt(), dim=0)
            self.value.copy_(value_ref)

        if smart_encode:
            mse = HSVLoss.TargetImage('HSV loss', self.image_shape, pil_image)

            if self.hdr_loss is not None:
                before_weight = self.hdr_loss.weight.clone()
                self.hdr_loss.set_weight(0.01)
            guide = DirectImageGuide(self, None, optimizer=optim.Adam([self.pallet, self.tensor], lr=0.1))
            guide.run_steps(100, [], [], [mse])  # Reduced from 201 to 100 steps
            if self.hdr_loss is not None:
                self.hdr_loss.set_weight(before_weight)

    @torch.no_grad()
    def encode_random(self, random_pallet=False):
        self.value.uniform_()
        self.tensor.uniform_()
        if random_pallet:
            self.pallet.uniform_(to=self.pallet_inertia)
