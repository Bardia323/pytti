import torch
from CLIP import clip
from pytti import DEVICE, vram_usage_mode

CLIP_PERCEPTORS = None

# Every model loaded in this process, by name, kept with whatever was attached to it (e.g. its compiled
# encode_image). Switching a model off parks it in system RAM; switching it back on moves it to the GPU
# again instead of reloading + recompiling.
_LOADED = {}

def _attention_no_setattr(self, x):
  # OpenAI CLIP's ResidualAttentionBlock.attention stores the cast mask back on the module every call;
  # torch.compile can't trace that setattr and splits the model at every block (23 graph breaks for
  # ViT-B/32), which keeps ~2x the activation memory per cutout. Same maths, no module mutation.
  m = self.attn_mask
  if m is not None:
    m = m.to(dtype=x.dtype, device=x.device)
  return self.attn(x, x, x, need_weights=False, attn_mask=m)[0]

def _patch_openai_attention():
  from CLIP.clip import model as _cm
  if _cm.ResidualAttentionBlock.attention is not _attention_no_setattr:
    _cm.ResidualAttentionBlock.attention = _attention_no_setattr

def _load_openai(name):
  _patch_openai_attention()
  return clip.load(name, jit=False)[0].eval().requires_grad_(False).to(DEVICE, memory_format=torch.channels_last)

def cached_perceptor(name, loader):
  m = _LOADED.get(name)
  if m is None:
    m = _LOADED[name] = loader()
  elif getattr(m, "_pytti_parked", False):
    m.to(DEVICE)
    m._pytti_parked = False
  return m

def park_unused(in_use):
  """move loaded models that aren't selected to system RAM (frees their VRAM, keeps them compiled)"""
  keep = {id(m) for m in in_use}
  for m in _LOADED.values():
    if id(m) not in keep and not getattr(m, "_pytti_parked", False):
      m.to("cpu")
      m._pytti_parked = True
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

@vram_usage_mode('CLIP')
def init_clip(clip_models):
  global CLIP_PERCEPTORS
  if CLIP_PERCEPTORS is None:
    CLIP_PERCEPTORS = [cached_perceptor(model, lambda model=model: _load_openai(model)) for model in clip_models]

def free_clip():
  global CLIP_PERCEPTORS
  CLIP_PERCEPTORS = None
