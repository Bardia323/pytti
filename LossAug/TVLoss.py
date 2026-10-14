from torch.nn import functional as F
import math, os
import torch
from pytti.LossAug import Loss

def tv_loss(input):
  """L2 total variation loss, as in Mahendran et al."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
  y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
  return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def tv_loss_edge(input):
  """edge-aware TV (opt-in, OPTS["edge_tv"]): each neighbour difference = brightness difference + colour difference;
  the colour part is weighted by 1 / (1 + (dY / s)^2) (Perona-Malik), s = the image's mean brightness difference, so
  colour is smoothed inside regions but not dragged across brightness edges (no colour bleed). Without edges it is
  exactly tv_loss.
  The smoothed brightness is the grey axis (channel mean), not luma: a luma-weighted difference pushes along
  (.299, .587, .114), ~60% green, and at smoothing weight ~10 that turned the whole image green in ~25 frames.
  Luma only picks where the edges are (weights detached)."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  dx = input[..., :-1, 1:] - input[..., :-1, :-1]
  dy = input[..., 1:, :-1] - input[..., :-1, :-1]
  mx, my = dx.mean(1, keepdim=True), dy.mean(1, keepdim=True)        # grey-axis (brightness) part
  cx, cy = dx - mx, dy - my                                          # colour part, orthogonal to grey
  with torch.no_grad():
    luma = input.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    lx, ly = (dx * luma).sum(1, keepdim=True), (dy * luma).sum(1, keepdim=True)
    s = (0.5 * (lx.abs().mean() + ly.abs().mean())).clamp_min(1e-4)
    wx = 1 / (1 + (lx / s) ** 2)
    wy = 1 / (1 + (ly / s) ** 2)
  # per channel: m^2 + w c_k^2; summed over channels = 3 m^2 + w |c|^2 = |d|^2 when w = 1
  return (mx**2 + wx * cx**2 + my**2 + wy * cy**2).mean([1, 2, 3])

OPTS = {}         # cell 2.15 points this at its OPT dict ("edge_tv")
_TV = {}
def _tv_fused(input, fn0=tv_loss):
  """tv_loss compiled into one forward and one backward kernel (same math; falls back to eager)"""
  fn = _TV.get(fn0)
  if fn is None:
    try:
      import torch
      fn = torch.compile(fn0, dynamic=False)
    except Exception:
      fn = fn0
    _TV[fn0] = fn
  try:
    return fn(input)
  except Exception as e:
    print(f"[TVLoss] compiled {fn0.__name__} failed ({type(e).__name__}); using eager")
    _TV[fn0] = fn0
    return fn0(input)

class TVLoss(Loss):
  def __init__(self, weight = 0.15, stop = -math.inf, name = "smoothing loss (TV)"):
    super().__init__(weight, stop, name)
  def get_loss(self, input, img):
    fn0 = tv_loss_edge if OPTS.get("edge_tv") else tv_loss
    return _tv_fused(input, fn0) if input.is_cuda and os.environ.get("PYTTI_FAST_TV", "1") != "0" else fn0(input)