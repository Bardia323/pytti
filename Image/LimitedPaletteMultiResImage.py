# LimitedPaletteMultiResImage.py

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image

from pytti import DEVICE, clamp_with_grad
from pytti.Image import DifferentiableImage

class LimitedPaletteMultiResImage(DifferentiableImage):
    """
    Limited Palette Multi-resolution Image.
    
    This model represents an image using a multi-resolution decomposition 
    (learnable residuals at several scales) and then applies limited palette
    quantization. The continuous image (computed from the residuals via tanh)
    is converted to a discrete image by computing its luminance, quantizing that
    luminance into indices into a sorted palette, and then blending a discrete
    version with a continuous interpolation.
    
    Parameters:
      - width, height: Final output resolution.
      - palette_size: Number of palette entries.
      - scales: Tuple of downscale factors.
      - gamma: Gamma correction applied during encoding.
      - init: How to initialize the residuals ('random' or 'zeros').
    """
    def __init__(self, width, height, palette_size, scales=(1,2,4,8,16),
                 gamma=0.8, init='random', device=DEVICE):
        super().__init__(width, height, 'RGB')
        self.pixel_format = 'RGB'
        self.scales = scales
        self.gamma = gamma
        n_channels = 3

        # Multi-resolution residuals.
        self.residuals = nn.ParameterList()
        for s in scales:
            h_down = max(1, height // s)
            w_down = max(1, width // s)
            if init == 'random':
                data = 2 * torch.rand(n_channels, h_down, w_down, device=device) - 1
            else:
                data = torch.zeros(n_channels, h_down, w_down, device=device)
            self.residuals.append(nn.Parameter(data))
        
        # Palette: learnable parameter [palette_size, 3] with values in [0,1].
        self.palette = nn.Parameter(torch.rand(palette_size, 3, device=device))
        self.register_buffer('palette_target', torch.empty_like(self.palette))
        self.use_palette_target = False

        self.output_axes = ('n', 's', 'y', 'x')
        self.lr = 0.1

    def decode_tensor(self):
        """
        Decodes the image:
          1. Upsample each residual to full resolution, sum, apply tanh → [0,1] continuous image.
          2. Compute luminance and scale it to [0, palette_size-1].
          3. Compute discrete indices (rounded) and also floor/ceil interpolation.
          4. Blend the two versions (50:50) to produce the final quantized image.
        Returns a tensor of shape [1, 3, H, W].
        """
        width, height = self.image_shape
        device = self.residuals[0].device

        # 1. Compute continuous image.
        accum = torch.zeros(1, 3, height, width, device=device)
        for scale, param in zip(self.scales, self.residuals):
            up = F.interpolate(param.unsqueeze(0), size=(height, width),
                               mode='bilinear', align_corners=True)
            accum += up
        continuous = (torch.tanh(accum) + 1) / 2  # [1,3,H,W] in [0,1]

        # 2. Compute luminance.
        R = continuous[:, 0:1, :, :]
        G = continuous[:, 1:2, :, :]
        B = continuous[:, 2:3, :, :]
        lum = 0.299 * R + 0.587 * G + 0.114 * B  # [1,1,H,W]

        # 3. Scale luminance to palette indices.
        palette_size = self.palette.shape[0]
        scaled = lum * (palette_size - 1)  # [1,1,H,W]

        # 4. Compute discrete indices and interpolation.
        rounded = scaled.round().squeeze(0).squeeze(0).clamp(0, palette_size - 1).long()  # [H,W]
        floor_idx = scaled.floor().squeeze(0).squeeze(0).long()
        ceil_idx = scaled.ceil().squeeze(0).squeeze(0).long()
        frac = (scaled - scaled.floor()).squeeze(0).squeeze(0)

        # 5. Retrieve sorted palette.
        sorted_palette = self.sort_palette()  # [palette_size, 3]

        # 6. Discrete version.
        discrete_img = sorted_palette[rounded]  # [H,W,3]

        # 7. Continuous interpolation.
        floor_color = sorted_palette[floor_idx]
        ceil_color = sorted_palette[ceil_idx]
        continuous_interp = floor_color * (1 - frac.unsqueeze(-1)) + ceil_color * frac.unsqueeze(-1)

        # 8. Blend both versions.
        blended = 0.5 * (discrete_img + continuous_interp)  # [H,W,3]
        final_img = blended.permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
        return clamp_with_grad(final_img, 0, 1)

    def get_image_tensor(self):
        """Returns decoded image tensor with shape [3, H, W]."""
        return self.decode_tensor().squeeze(0)

    def set_image_tensor(self, tensor):
        """
        Inverts the tanh mapping and distributes evenly among scales.
        Expects tensor of shape [3, H, W] in [0,1].
        """
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor with shape [3, H, W], got {tensor.shape}")
        eps = 1e-5
        tensor = tensor.clamp(eps, 1 - eps)
        pre = 0.5 * torch.log(tensor / (1 - tensor))
        N = len(self.scales)
        for i in range(N):
            h_down = self.residuals[i].shape[1]
            w_down = self.residuals[i].shape[2]
            down = F.interpolate(pre.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_(down.squeeze(0) / N)

    @torch.no_grad()
    def encode_image(self, pil_image, smart_encode=True, device=DEVICE):
        """
        Improved encoding:
          - Converts the image to RGB and converts to tensor.
          - Adjusts the image by centering its mean brightness at 0.5 to reduce fade.
          - Applies gamma correction.
          - Downsamples to each resolution and maps values to [-1,1].
          - If smart_encode is True, updates the palette via k-means clustering on a subsample.
        """
        pil_image = pil_image.convert('RGB')
        full_tensor = TF.to_tensor(pil_image).to(device)
        # Adjust brightness: center the mean at 0.5.
        mean_val = full_tensor.mean()
        full_tensor = (full_tensor - mean_val) + 0.5
        full_tensor = full_tensor.clamp(0, 1)
        # Apply gamma correction.
        corrected = full_tensor.pow(self.gamma)
        width, height = self.image_shape
        N = len(self.scales)
        for i in range(N):
            h_down, w_down = self.residuals[i].shape[1], self.residuals[i].shape[2]
            down = F.interpolate(corrected.unsqueeze(0), size=(h_down, w_down),
                                 mode='bilinear', align_corners=True)
            self.residuals[i].data.copy_((down[0] * 2 - 1) / N)
        if smart_encode:
            sample_size = 4096
            flat = full_tensor.view(3, -1).transpose(0, 1)  # [num_pixels, 3]
            num_pixels = flat.shape[0]
            if num_pixels > sample_size:
                indices = torch.randperm(num_pixels)[:sample_size]
                flat_sample = flat[indices]
            else:
                flat_sample = flat
            from sklearn.cluster import KMeans
            n_clusters = self.palette.shape[0]
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
            labels = kmeans.fit_predict(flat_sample.cpu().numpy())
            centers = torch.tensor(kmeans.cluster_centers_, device=device, dtype=self.palette.dtype)
            brightness = 0.299 * centers[:,0] + 0.587 * centers[:,1] + 0.114 * centers[:,2]
            sorted_indices = torch.argsort(brightness)
            sorted_palette = centers[sorted_indices]
            self.palette.data.copy_(sorted_palette)

    @torch.no_grad()
    def encode_random(self, random_palette=False):
        """
        Randomly initializes the residuals in [-1,1]. If random_palette is True,
        reinitializes the palette with random values in [0,1].
        """
        for param in self.residuals:
            param.uniform_(-1, 1)
        if random_palette:
            self.palette.uniform_(0, 1)

    def update(self):
        """
        Clamps the residuals to [-1,1] and the palette to [0,1].
        """
        for param in self.residuals:
            param.data.clamp_(-1, 1)
        self.palette.data.clamp_(0, 1)

    @torch.no_grad()
    def set_palette_target(self, pil_image):
        """
        Updates the palette using k-means clustering on the target image.
        A random subset of up to 4096 pixels is used to speed up clustering.
        The resulting palette is sorted by brightness and copied into self.palette.
        Also sets use_palette_target to True.
        """
        if pil_image is None:
            self.use_palette_target = False
            return
        pil_image = pil_image.convert('RGB')
        target_tensor = TF.to_tensor(pil_image).to(self.palette.device)  # [3, H, W]
        flat = target_tensor.view(3, -1).transpose(0, 1)  # [num_pixels, 3]
        sample_size = 4096
        num_pixels = flat.shape[0]
        if num_pixels > sample_size:
            indices = torch.randperm(num_pixels)[:sample_size]
            flat_sample = flat[indices]
        else:
            flat_sample = flat
        from sklearn.cluster import KMeans
        n_clusters = self.palette.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5)
        labels = kmeans.fit_predict(flat_sample.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, device=self.palette.device, dtype=self.palette.dtype)
        brightness = 0.299 * centers[:,0] + 0.587 * centers[:,1] + 0.114 * centers[:,2]
        sorted_indices = torch.argsort(brightness)
        sorted_palette = centers[sorted_indices]
        with torch.no_grad():
            self.palette.copy_(sorted_palette)
            self.use_palette_target = True

    def lock_palette(self, lock=True):
        """
        When locked, the palette remains fixed during optimization.
        """
        self.use_palette_target = lock

    def lock_pallet(self, lock=True):
        """
        Alias for compatibility.
        """
        self.lock_palette(lock)

    def sort_palette(self):
        """
        Returns the palette sorted by brightness.
        """
        brightness = 0.299 * self.palette[:,0] + 0.587 * self.palette[:,1] + 0.114 * self.palette[:,2]
        sorted_indices = torch.argsort(brightness)
        return self.palette[sorted_indices]

def render_palette(palette, cell_size=32):
    """
    Renders the palette (tensor of shape [N, 3]) as an image.
    Each palette color is displayed as a cell of size (cell_size x cell_size).
    """
    import numpy as np
    palette_np = palette.detach().cpu().numpy()
    n = palette_np.shape[0]
    out = np.zeros((cell_size, cell_size * n, 3), dtype=np.uint8)
    for i, color in enumerate(palette_np):
        out[:, i*cell_size:(i+1)*cell_size, :] = (color * 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(out)
