from pytti import *
from pytti.Image import DifferentiableImage
from pytti.LossAug import HSVLoss
from pytti.ImageGuide import DirectImageGuide
import math
import os
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

_LUMA = {}
def luma_weights(like):
    """[[[0.299, 0.587, 0.114]]] cached on the device (building it per call was a blocking copy)"""
    key = (like.device, like.dtype)
    if key not in _LUMA:
        _LUMA[key] = torch.tensor([[[0.299, 0.587, 0.114]]], device=like.device, dtype=like.dtype)
    return _LUMA[key]

class _PalletLookup(torch.autograd.Function):
    """
    pallet[idx] with a fast backward. The default index backward sorts all H*W indices and then
    reduces each palette entry's duplicates serially -- with ~2M pixels hitting a few dozen entries
    that one kernel was ~70% of a training step. index_add_ does the same reduction in ~1-2ms.
    """
    @staticmethod
    def forward(ctx, pallet, idx):
        ctx.save_for_backward(idx)
        ctx.pallet_shape = pallet.shape
        return pallet[idx]

    @staticmethod
    def backward(ctx, grad):
        idx, = ctx.saved_tensors
        n_entries = ctx.pallet_shape[0]
        grad = grad.reshape(idx.numel(), -1)
        out = grad.new_zeros(n_entries, grad.shape[1])
        out.index_add_(0, idx.reshape(-1), grad)
        return out.view(ctx.pallet_shape), None

def pallet_lookup(pallet, idx):
    return _PalletLookup.apply(pallet, idx)

def _cont_blend(pallet, floors, ceils, fracs, tensor):
    """the continuous palette blend of decode_tensor (it only shapes the gradient)"""
    weights = F.softmax(tensor.movedim(0, 2), dim=2).unsqueeze(-1)
    fracs = fracs.unsqueeze(-1).unsqueeze(-1)
    return ((pallet[floors] * (1 - fracs) + pallet[ceils] * fracs) * weights).sum(dim=2)

def _pallet_loss_raw(tensor, n_pallets):
    tensor = tensor.movedim(0, -1).contiguous().view(-1, n_pallets)
    tensor = F.softmax(tensor, dim=-1)
    N, n = tensor.shape
    mu = tensor.mean(dim=0, keepdim=True)
    sigma = tensor.std(dim=0, keepdim=True) + 1e-8          # epsilon prevents division by zero
    tensor_centered = tensor - mu
    S = (tensor_centered.transpose(0, 1) @ tensor_centered).div(sigma * sigma.transpose(0, 1) * N)
    S = S - torch.diag(S.diagonal())                      # minimize correlation (anticorrelate palettes)
    return S.mean() + sigma.mul(N).pow(-1).mean()         # maximize variance within each palette

# ---- Automatic colour (opt-in: PixelImage(auto_colour=True) or PixelImage.auto_colour = True) ----
# Vector quantisation done the standard way instead of learning palettes by gradient:
#  * each pixel holds a free colour x in Oklab (perceptually uniform, so one learning rate fits all three axes);
#  * the palette is the codebook that best quantises the image: lightness rounded to `pallet_size` even levels, and per
#    level `n_pallets` colours (a, b) kept at the k-means optimum of that level's pixels by one Lloyd step per decode
#    (unused entries re-seeded at the worst-quantised pixel). Entries get as many pixels as the image needs; nothing
#    is learned by gradient, so nothing can collapse or go stale;
#  * the picture is exactly Q(x) (the limited palette); the gradient passes straight through to x (as in VQ-VAE);
#  * x is projected back into the sRGB gamut after every change, along constant lightness and hue (CSS Color 4's
#    chroma reduction), so nothing clips or crushes;
#  * global tone is CLIP's call, not a drifting by-product: the image's mean lightness, lightness spread and mean
#    chroma flow toward a target `tone` (a few frames' time constant, never a snap), because CLIP's per-pixel gradient barely
#    constrains them and its noise piles up into a random walk (dark / washed out / neon over a long run). The target
#    itself is moved by asking CLIP: cell 2.15 renders the picture with one of the three nudged up and down and moves
#    the target when CLIP prefers it on two independent cutout sets (common random numbers, so noise can't move it).
#    Prompt-dependent, nothing to tune; pinning freezes the target, a target palette image sets it.
_OKLAB_M2 = ((1.0, 0.3963377774, 0.2158037573), (1.0, -0.1055613458, -0.0638541728),
             (1.0, -0.0894841775, -1.2914855480))
_OKLAB_M1 = ((4.0767416621, -3.3077115913, 0.2309699292), (-1.2684380046, 2.6097574011, -0.3413193965),
             (-0.0041960863, -0.7034186147, 1.7076147010))
_LIN2LMS = ((0.4122214708, 0.5363325363, 0.0514459929), (0.2119034982, 0.6806995451, 0.1073969566),
            (0.0883024619, 0.2817188376, 0.6299787005))
_LMS2LAB = ((0.2104542553, 0.7936177850, -0.0040720468), (1.9779984951, -2.4285922050, 0.4505937099),
            (0.0259040371, 0.7827717662, -0.8086757660))
# tone = (mean L, log std L, log mean chroma); the step CLIP's tone search tries in each
TONE_STEPS = (0.02, 0.05, 0.08)
# The picture isn't snapped to its tone target: each update moves it this fraction of the way (a gradient flow on the
# distance to the target), so its darks / lights change gradually. Cell 2.15 sets it per step for a time constant of
# TONE_FLOW_FRAMES frames; drift still can't build up because the pull never stops.
TONE_FLOW_FRAMES = 4
_OKM = {}

def _okm(like):
    key = (like.device, like.dtype)
    if key not in _OKM:
        _OKM[key] = tuple(torch.tensor(m, device=like.device, dtype=like.dtype).T.contiguous()
                          for m in (_OKLAB_M2, _OKLAB_M1, _LIN2LMS, _LMS2LAB))
    return _OKM[key]

def _lab_to_lin(lab):
    m2, m1, _, _ = _okm(lab)
    return (lab @ m2).pow(3) @ m1

def _lin_to_srgb(lin):
    lin = lin.clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp_min(1e-8).pow(1 / 2.4) - 0.055)

def _srgb_to_oklab(rgb):
    """... x 3 sRGB in [0, 1] -> ... x 3 Oklab"""
    _, _, to_lms, to_lab = _okm(rgb)
    rgb = rgb.clamp(0, 1)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    return (lin @ to_lms).clamp_min(0).pow(1 / 3) @ to_lab

def _lin_of(L, a, b):
    """Oklab -> linear sRGB, written out (elementwise only, so the bisection below compiles into one kernel)"""
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)

def _in_gamut(L, a, b, t):
    ok = None
    for c in _lin_of(L, a * t, b * t):
        o = (c >= -1e-5) & (c <= 1 + 1e-5)
        ok = o if ok is None else ok & o
    return ok

def _gamut_project(lab, iters=14):
    """N x 3 Oklab -> the nearest displayable colour of the same lightness and hue (chroma reduced by bisection)"""
    L, a, b = lab[:, 0].clamp(0, 1), lab[:, 1], lab[:, 2]
    lo, hi = torch.zeros_like(L), torch.ones_like(L)
    ok1 = _in_gamut(L, a, b, hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = _in_gamut(L, a, b, mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    t = torch.where(ok1, hi, lo)
    return torch.stack([L, a * t, b * t], dim=-1)

def _assign(ab, level, centres):
    """nearest palette colour of each pixel's lightness level: N x 2, N, S x n x 2 -> N"""
    return (ab.unsqueeze(1) - centres[level]).square().sum(-1).argmin(dim=1)

def _measure_tone(x):
    """N x 3 Oklab -> tone (mean L, log std L, log mean chroma)"""
    L, c = x[:, 0], x[:, 1:].norm(dim=-1)
    return torch.stack([L.mean(), L.std().clamp_min(1e-4).log(), c.mean().clamp_min(1e-4).log()])

def _apply_tone(x, tone):
    """the smallest change (per moment) giving x this tone: lightness standardised to the target mean / spread,
    chroma scaled to the target mean; then into [0, 1] and the gamut"""
    L, ab = x[:, 0], x[:, 1:]
    L = (L - L.mean()) * (tone[1].exp() / L.std().clamp_min(1e-4)) + tone[0]
    ab = ab * (tone[2].exp() / ab.norm(dim=-1).mean().clamp_min(1e-4))
    return _fused(_gamut_project, torch.cat([L.clamp(0, 1).unsqueeze(-1), ab], dim=-1).contiguous())

# Fused Triton kernels for the blend and the palette normalization loss (same math, one or two passes
# instead of many; see _palette_kernels.py). PYTTI_FAST_PALETTE=0 turns them off; any failure falls back.
_FAST = {"on": os.environ.get("PYTTI_FAST_PALETTE", "1") != "0"}
def _fast(name, *args):
    if not _FAST["on"]:
        return None
    try:
        from pytti.Image import _palette_kernels as K
        if not K.supported(args[-1] if name == "blend" else args[0], args[0] if name == "blend" else None):
            return None
        return K.cont_blend(*args) if name == "blend" else K.pallet_norm_loss(*args)
    except torch.cuda.OutOfMemoryError:
        raise                    # out of memory is the run's problem, not the kernel's: keep it on
    except Exception as e:
        print(f"[PixelImage] fast {name} kernel failed ({type(e).__name__}: {e}); using PyTorch")
        _FAST["on"] = False
        return None

_FUSED = {}
def _fused(fn, *args):
    """fn compiled with torch.compile: the gather/blend/reduce fuse into a few kernels, so the
    H x W x n_pallets x 3 intermediates never hit memory (~5x faster colour decode). Falls back to
    eager if compiling isn't possible."""
    c = _FUSED.get(fn)
    if c is None:
        try:
            c = torch.compile(fn, dynamic=False)     # one static compile per image size
        except Exception:
            c = fn
        _FUSED[fn] = c
    if c is not fn:
        try:
            return c(*args)
        except torch.cuda.OutOfMemoryError:
            raise                # likewise: don't fall back to the slow path for good
        except Exception as e:
            print(f"[PixelImage] torch.compile of {fn.__name__} failed ({type(e).__name__}); using eager")
            _FUSED[fn] = fn
    return fn(*args)

class PalletLoss(nn.Module):
    def __init__(self, n_pallets, weight=0.15, device=DEVICE):
        super().__init__()
        self.n_pallets = n_pallets
        self.register_buffer('weight', torch.as_tensor(float(weight), dtype=torch.float32, device=device))

    def forward(self, input):
        if isinstance(input, PixelImage):
            loss_raw = _fast("norm", input.tensor)
            if loss_raw is None:
                loss_raw = _fused(_pallet_loss_raw, input.tensor, self.n_pallets)
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=DEVICE):
        self.weight.set_(torch.as_tensor(float(weight), dtype=torch.float32, device=device))

    def __str__(self):
        return "Palette normalization"

class HdrLoss(nn.Module):
    # Ramp-in: 0 = full weight from the first step. N = the weight rises linearly from 0 to its value over
    # the first N steps of this image (so an init image isn't jolted on frame one). Class attribute so it
    # can be changed live; the step count is per image.
    ramp_steps = 0
    # keep_hue: the loss only rescales each palette colour (its push keeps hue and saturation). The plain loss pushes
    # along the luma weights (R .299, G .587, B .114), so every time it brightens a palette ~60% of the change is
    # green: over a long run with prompts that darken the image, the palettes drift to green. Same loss value either
    # way; only the direction of the push differs. Class attribute (cell 2.15 `hdr_keeps_hue`).
    keep_hue = False

    def __init__(self, pallet_size, n_pallets, gamma=2.5, weight=0.15, device=DEVICE):
        super().__init__()
        self._steps = 0
        self.register_buffer('comp', torch.linspace(0, 1, pallet_size).pow(gamma).view(pallet_size, 1).repeat(1, n_pallets).to(device))
        self.register_buffer('weight', torch.as_tensor(float(weight), dtype=torch.float32, device=device))

    def forward(self, input):
        if isinstance(input, PixelImage):
            pallet = input.sort_pallet()
            magic_color = luma_weights(pallet)
            color_norms = torch.linalg.vector_norm(pallet * magic_color.sqrt(), dim=-1)
            if self.keep_hue:
                # same value; the gradient reaches the palette only through each colour's length (radial = a pure
                # brightness change of that colour)
                with torch.no_grad():
                    u = pallet / pallet.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                r = (pallet * u).sum(-1)
                color_norms = color_norms.detach() * r / r.detach().clamp_min(1e-6)
            loss_raw = F.mse_loss(color_norms, self.comp)
            ramp = 1.0
            if self.ramp_steps > 0 and self.training:
                self._steps += 1
                ramp = min(1.0, self._steps / float(self.ramp_steps))
            return loss_raw * self.weight * ramp, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=DEVICE):
        self.weight.set_(torch.as_tensor(float(weight), dtype=torch.float32, device=device))

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
    auto_colour = False      # default for new images (see "Automatic colour" above); the notebook / studio set it

    @vram_usage_mode('Limited Palette Image')
    def __init__(self, width, height, scale, pallet_size, n_pallets, gamma=1, hdr_weight=0.5, norm_weight=0.1, device=DEVICE,
                 auto_colour=None):
        super().__init__(width * scale, height * scale)
        self.pallet_inertia = 2
        # Initialize palette with gamma=1 (no correction) initially
        pallet = torch.linspace(0, self.pallet_inertia, pallet_size).view(pallet_size, 1, 1).repeat(1, n_pallets, 3)
        self.pallet = nn.Parameter(pallet.to(device, dtype=torch.float32))  # Use float32 for stability
        self.pallet_size = pallet_size
        self.n_pallets = n_pallets
        self.value = nn.Parameter(torch.zeros(height, width, dtype=torch.float32, device=device))
        self.tensor = nn.Parameter(torch.zeros(n_pallets, height, width, dtype=torch.float32, device=device))
        # palette-choice scores need per-element Adam: a block / pyramid step size would starve the palettes that are
        # losing (tiny gradients) and let one palette take over the image
        self.tensor._pytti_precond = False
        self.output_axes = ('n', 's', 'y', 'x')
        self.latent_strength = 0.1
        self.scale = scale
        self.hdr_loss = HdrLoss(pallet_size, n_pallets, 1.0, hdr_weight) if hdr_weight != 0 else None
        self.loss = PalletLoss(n_pallets, norm_weight)
        self.register_buffer('pallet_target', torch.empty_like(self.pallet))
        self.use_pallet_target = False
        # Store target gamma. Start AT the target so the very first frame already shows the
        # intended gamma/saturation (the old 1.0 start ramped up over ~120 frames, making the
        # init look washed-out until it "caught up").
        self.target_gamma = gamma
        self.current_gamma = gamma
        self.gamma_step = 0.01  # How quickly to approach target gamma (now a no-op unless lowered)
        auto = (type(self).auto_colour if type(self) is PixelImage else False) if auto_colour is None else auto_colour
        mono = getattr(__import__("builtins"), "MONO", {}).get("enable", False)
        self.auto_colour = bool(auto) and not mono
        self.tone = None              # auto colour: (mean L, log std L, log mean chroma) target; None = not yet measured
        self.tone_frozen = False      # pinned / target palette: CLIP's tone search leaves it alone
        self.tone_rate = 1 - math.exp(-1 / (TONE_FLOW_FRAMES * 10))   # per update; 2.15 sets it from the frame length
        if self.auto_colour:
            # the pixels' wanted colours (Oklab) are the only optimised parameter; palette = k-means codebook
            self.pallet.requires_grad_(False)
            self.value.requires_grad_(False)
            self.tensor = nn.Parameter(torch.zeros(n_pallets, 1, 1, device=device), requires_grad=False)  # unused
            self.lab = nn.Parameter(torch.zeros(3, height, width, dtype=torch.float32, device=device))
            self.register_buffer('levels', torch.linspace(0, 1, pallet_size, device=device))
            self.register_buffer('centres', torch.zeros(pallet_size, n_pallets, 2, device=device))

    def clone(self):
        width, height = self.image_shape
        dummy = PixelImage(width // self.scale, height // self.scale, self.scale, self.pallet_size, self.n_pallets,
                           hdr_weight=0 if self.hdr_loss is None else float(self.hdr_loss.weight),
                           norm_weight=float(self.loss.weight), auto_colour=self.auto_colour)
        with torch.no_grad():
            if self.auto_colour:
                for k in ("lab", "centres"):
                    getattr(dummy, k).copy_(getattr(self, k))
                dummy.tone = None if self.tone is None else self.tone.clone()
                dummy.tone_frozen = self.tone_frozen
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
        if self.auto_colour:
            # the palette = that image's k-means codebook, locked; its brightness / saturation become the reference
            with torch.no_grad():
                lab = self._lab_of(pil_image)
                self._fit_centres(lab)
                self.tone, self.tone_frozen = _measure_tone(lab), True
            self.use_pallet_target = True
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
        if self.auto_colour:
            self.use_pallet_target = bool(lock)      # freezes the codebook (no Lloyd steps)
            return
        if lock:
            self.pallet_target.copy_(self.sort_pallet())
        self.use_pallet_target = lock

    def image_loss(self):
        if self.auto_colour:
            return []        # nothing to weigh: the palette is fitted, the gamut and distributions projected
        return [x for x in [self.hdr_loss, self.loss] if x is not None]

    def sort_pallet(self):
        if self.auto_colour:
            return self._palette_rgb()
        if self.use_pallet_target:
            return self.pallet_target
        pallet = (self.pallet / self.pallet_inertia).clamp(0, 1)
        # Calculate color norms for sorting
        magic_color = luma_weights(pallet)
        color_norms = pallet.square().mul(magic_color).sum(dim=-1)
        # Optimized sorting using torch.gather
        pallet_indices = color_norms.argsort(dim=0)
        sorted_pallet = torch.gather(pallet, 0, pallet_indices.unsqueeze(-1).expand(-1, self.n_pallets, 3))
        return sorted_pallet

    def get_image_tensor(self):
        if self.auto_colour:
            return self.lab
        return torch.cat([self.value.unsqueeze(0), self.tensor])

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        if self.auto_colour:
            self.lab.copy_(tensor)
            self._project_state()
            return
        self.value.copy_(tensor[0])
        self.tensor.copy_(tensor[1:])

    # ---- automatic colour (see the notes at the top of this file)
    def _flat_lab(self):
        return self.lab.permute(1, 2, 0).reshape(-1, 3)

    @torch.no_grad()
    def _lab_of(self, pil_image):
        h, w = self.lab.shape[-2:]
        ref = TF.to_tensor(pil_image.convert('RGB').resize((w, h), Image.LANCZOS)).to(self.lab.device)
        return _srgb_to_oklab(ref.permute(1, 2, 0).reshape(-1, 3))

    def _levels_of(self, L):
        return (L.clamp(0, 1) * (self.pallet_size - 1)).round().long()

    @torch.no_grad()
    def _palette_rgb(self):
        S, n = self.pallet_size, self.n_pallets
        lab = torch.cat([self.levels.view(S, 1, 1).expand(S, n, 1), self.centres], dim=-1).reshape(-1, 3)
        return _lin_to_srgb(_lab_to_lin(_fused(_gamut_project, lab.contiguous()))).view(S, n, 3)

    @torch.no_grad()
    def _lloyd(self, ab, level, idx):
        """one Lloyd step: every palette colour -> the mean of its pixels; one unused colour per level re-seeded at
        that level's worst-quantised pixel"""
        S, n, N = self.pallet_size, self.n_pallets, ab.shape[0]
        C = self.centres.view(-1, 2)
        sums = torch.zeros_like(C).index_add_(0, idx, ab)
        cnt = C.new_zeros(S * n).index_add_(0, idx, C.new_ones(N))
        new = torch.where(cnt.unsqueeze(-1) > 0, sums / cnt.clamp_min(1).unsqueeze(-1), C)
        # worst-quantised pixel per level: sort by (level, error); each level's last entry
        err = (ab - new[idx]).square().sum(-1)
        order = (level.float() * 4 + err.clamp(max=3.9)).argsort()
        lcnt = cnt.view(S, n).sum(dim=1).long()
        last = order[(lcnt.cumsum(0) - 1).clamp(min=0)]
        dead = (cnt == 0).view(S, n)
        j = dead.float().argmax(dim=1)
        do = dead.any(dim=1) & (lcnt > 0) & (err[last] > 1e-8)
        rows = torch.arange(S, device=ab.device)
        newv = new.view(S, n, 2)
        newv[rows, j] = torch.where(do.unsqueeze(-1), ab[last], newv[rows, j])
        self.centres.copy_(newv)

    @torch.no_grad()
    def _fit_centres(self, lab, iters=8):
        """k-means codebook for these Oklab pixels: n random pixels per level, then Lloyd steps"""
        S, n = self.pallet_size, self.n_pallets
        level = self._levels_of(lab[:, 0])
        perm = torch.randperm(lab.shape[0], device=lab.device)
        order = perm[level[perm].argsort(stable=True)]
        cnt = torch.bincount(level, minlength=S)
        start = cnt.cumsum(0) - cnt
        pos = start.view(S, 1) + torch.arange(n, device=lab.device).view(1, n) % cnt.clamp_min(1).view(S, 1)
        seeds = lab[order[pos.clamp(max=lab.shape[0] - 1)], 1:]
        self.centres.copy_(torch.where((cnt > 0).view(S, 1, 1), seeds, torch.zeros_like(seeds)))
        ab = lab[:, 1:].contiguous()
        for _ in range(iters):
            k = _assign(ab, level, self.centres)
            self._lloyd(ab, level, level * n + k)

    @torch.no_grad()
    def _project_state(self):
        """after every change of the pixels: a step of the flow toward the tone target (measured on first use, so the
        starting picture is its own target), then the gamut"""
        x = self._flat_lab()
        if self.tone is None:
            self.tone = _measure_tone(_fused(_gamut_project, x.contiguous()))
        x = x + self.tone_rate * (_apply_tone(x, self.tone) - x)
        x = _fused(_gamut_project, x.contiguous())
        h, w = self.lab.shape[-2:]
        self.lab.copy_(x.view(h, w, 3).permute(2, 0, 1))

    @torch.no_grad()
    def tone_candidate(self, tone):
        """the picture CLIP would see with this tone (palette fixed; nothing changes)"""
        x = _apply_tone(self._flat_lab(), tone)
        level = self._levels_of(x[:, 0])
        idx = level * self.n_pallets + _fused(_assign, x[:, 1:].contiguous(), level, self.centres)
        out = self._palette_rgb().reshape(-1, 3)[idx]
        h, w = self.lab.shape[-2:]
        out = out.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        width, height = self.image_shape
        if (h, w) != (height, width):
            out = F.interpolate(out, size=(height, width), mode='nearest')
        return out

    @torch.no_grad()
    def pin_colours(self):
        """freeze the current tone; returns it (list) to store"""
        if self.tone is None:
            self._project_state()
        self.tone_frozen = True
        return self.tone.tolist()

    @torch.no_grad()
    def set_colour_reference(self, tone=None):
        """freeze a stored tone (from pin_colours); None = CLIP picks it again"""
        if tone is None:
            self.tone_frozen = False
            return
        self.tone = torch.as_tensor(tone, dtype=torch.float32, device=self.lab.device)
        self.tone_frozen = True
        self._project_state()

    def _decode_auto(self):
        x = self._flat_lab()
        with torch.no_grad():
            xd = x.detach()
            level = self._levels_of(xd[:, 0])
            ab = xd[:, 1:].contiguous()
            idx = level * self.n_pallets + _fused(_assign, ab, level, self.centres)
            if not self.use_pallet_target:
                self._lloyd(ab, level, idx)
            rgb_q = self._palette_rgb().reshape(-1, 3)[idx]
        rgb_c = _lin_to_srgb(_lab_to_lin(x))
        out = rgb_c + (rgb_q - rgb_c).detach()               # shows Q(x); the gradient goes straight to x
        h, w = self.lab.shape[-2:]
        out = out.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        width, height = self.image_shape
        if (h, w) != (height, width):
            out = F.interpolate(out, size=(height, width), mode='nearest')
        return out

    def decode_tensor(self):
        if self.auto_colour:
            return self._decode_auto()
        width, height = self.image_shape
        pallet = self.sort_pallet()

        # Apply current gamma to the palette during decoding
        gamma_corrected_pallet = pallet
        logits = self.tensor
        if self.current_gamma != 1.0:
            # Apply gamma correction to palette values
            gamma_corrected_pallet = pallet.pow(self.current_gamma)

        # Brightness values of pixels
        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)

        # discrete colour = the entry of each pixel's strongest palette: one flat gather
        # (same values as (pallet[rounds] * one_hot(argmax)).sum(), without the n_pallets-wide product)
        pallet_indices = logits.argmax(dim=0)
        colors_disc = pallet_lookup(gamma_corrected_pallet.reshape(-1, 3),
                                    value_rounds * self.n_pallets + pallet_indices)
        # continuous blend: only shapes the gradient; fused so it never materialises H*W*n_pallets*3
        colors_cont = _fast("blend", gamma_corrected_pallet, value_floors, value_ceils, value_fracs, logits)
        if colors_cont is None:
            colors_cont = _fused(_cont_blend, gamma_corrected_pallet, value_floors, value_ceils, value_fracs, logits)

        colors_disc = colors_disc.permute(2, 0, 1).unsqueeze(0)
        colors_cont = colors_cont.permute(2, 0, 1).unsqueeze(0)
        if colors_disc.shape[-2:] != (height, width):      # pixel_size > 1: upscale (nearest)
            colors_disc = F.interpolate(colors_disc.contiguous(), size=(height, width), mode='nearest')
            colors_cont = F.interpolate(colors_cont.contiguous(), size=(height, width), mode='nearest')

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
        # Gradually adjust gamma towards target
        if self.current_gamma < self.target_gamma:
            self.current_gamma = min(self.current_gamma + self.gamma_step, self.target_gamma)
            if self.hdr_loss is not None:
                self.hdr_loss.gamma = self.current_gamma
        # in place (the old clamp-then-copy made two passes over ~n_pallets*H*W values each step)
        self.pallet.clamp_(0, self.pallet_inertia)
        if self.auto_colour:
            self._project_state()
        self.value.clamp_(0, 1)
        self.tensor.clamp_(min=0)

    def encode_image_old(self, pil_image, smart_encode=True, device=DEVICE):
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

    def encode_image(self, pil_image, smart_encode=True, device=DEVICE, saturation=1.0):
      if self.auto_colour:
          return self._encode_image_auto(pil_image, saturation)
      width, height = self.image_shape

      scale = self.scale
      color_ref = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
      color_ref = TF.to_tensor(color_ref).to(device)

      # All operations below are done without tracking gradients
      with torch.no_grad():
          # Step 1: Compute grayscale values (self.value)
          magic_color = torch.tensor([0.299, 0.587, 0.114], device=device).view(3, 1, 1)
          value_ref = (color_ref * magic_color).sum(dim=0)
          self.value.copy_(value_ref)

          # Step 2: Quantize self.value into pallet_size levels
          pallet_size = self.pallet_size
          n_pallets = self.n_pallets
          # quantize by TRUE luma (decode indexes the palette by true luma, not value/max);
          # also avoids a divide-by-zero on all-black images
          value_quantized = (self.value.clamp(0, 1) * (pallet_size - 1)).long()
          value_quantized = value_quantized.clamp(0, pallet_size - 1)

          # Step 3: Compute normalized colors
          epsilon = 1e-6
          normalized_color = color_ref / (self.value.unsqueeze(0) + epsilon)

          # Prepare data for clustering
          normalized_color = normalized_color.permute(1, 2, 0).contiguous()  # H x W x 3
          normalized_color = normalized_color.view(-1, 3)
          value_quantized_flat = value_quantized.view(-1)

          # Prefill a grayscale brightness ramp so EVERY level is populated and brightness-monotonic.
          # Empty levels left at 0 make sort_pallet reorder rows and scramble colours (esp. reds).
          _ramp = (torch.arange(pallet_size, device=device, dtype=self.pallet.dtype)
                   / max(1, pallet_size - 1) * self.pallet_inertia).view(pallet_size, 1, 1)
          new_pallet = _ramp.repeat(1, n_pallets, 3).contiguous()
          new_tensor = torch.zeros_like(self.tensor)

          # Step 4: Cluster colors per brightness level
          from sklearn.cluster import KMeans
          for i in range(pallet_size):
              mask = (value_quantized_flat == i)
              if mask.sum() == 0:
                  continue  # Skip if no pixels at this brightness level

              colors = normalized_color[mask]

              # Perform K-means clustering
              n_clusters = min(n_pallets, colors.shape[0])  # Ensure n_clusters does not exceed number of samples
              if n_clusters == 0:
                  continue  # Skip if no colors to cluster
              kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=10, random_state=0)
              labels = kmeans.fit_predict(colors.cpu().numpy())
              cluster_centers = torch.tensor(kmeans.cluster_centers_, device=device)
              if saturation != 1.0:
                  # boost chroma away from gray (1,1,1) to counter palette desaturation;
                  # preserves luma (both gray and chroma have luma 1), so brightness is unchanged
                  cluster_centers = 1.0 + saturation * (cluster_centers - 1.0)

              # Step 5: Set new_pallet
              brightness_value = (i / (pallet_size - 1))
              # scale by pallet_inertia so decode's (pallet / pallet_inertia) reproduces the
              # original colour instead of halving it (encode/decode round-trip fix)
              new_pallet[i, :n_clusters, :] = cluster_centers * brightness_value * self.pallet_inertia
              if n_clusters < n_pallets:
                  # Fill remaining pallets with the last cluster center
                  new_pallet[i, n_clusters:, :] = cluster_centers[-1] * brightness_value * self.pallet_inertia

              # Step 6: Assign new_tensor
              indices = torch.nonzero(mask).squeeze()
              labels = torch.tensor(labels, device=device)
              h = self.value.shape[0]
              w = self.value.shape[1]
              k_indices = labels.long()
              y_indices = (indices // w).long()
              x_indices = (indices % w).long()
              new_tensor[k_indices, y_indices, x_indices] = 1

          # Normalize new_tensor
          new_tensor = new_tensor.clamp(0, 1)

          # Assign new tensors to self.pallet and self.tensor
          self.pallet.copy_(new_pallet)
          self.tensor.copy_(new_tensor)


    @torch.no_grad()
    def _encode_image_auto(self, pil_image, saturation=1.0):
        """auto colour: the pixels take the image's colours exactly, the palette is fitted to them"""
        lab = self._lab_of(pil_image)
        lab[:, 1:] *= saturation
        h, w = self.lab.shape[-2:]
        self.lab.copy_(lab.view(h, w, 3).permute(2, 0, 1))
        self._project_state()
        self._fit_centres(self._flat_lab())

    @torch.no_grad()
    def encode_random(self, random_pallet=False):
        self.value.uniform_()
        self.tensor.uniform_()
        if self.auto_colour:
            # random sRGB pixels (or random greys)
            h, w = self.lab.shape[-2:]
            rgb = torch.rand(h * w, 3 if random_pallet else 1, device=self.lab.device).expand(-1, 3)
            self.lab.copy_(_srgb_to_oklab(rgb).view(h, w, 3).permute(2, 0, 1))
            self._project_state()
            self._fit_centres(self._flat_lab())
            return
        if random_pallet:
            self.pallet.uniform_(to=self.pallet_inertia)
