#this library is designed for use with google colab runtimes.
#This file defines utility functions for use with notebooks.

#https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
import open_clip
def is_notebook():
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    elif shell == 'Shell':
      return True   # Google Colab
    else:
      print("DEGBUG: unknown shell type:",shell)
      return False
  except NameError:
    return False    # Probably standard Python interpreter

def change_tqdm_color():
  if not is_notebook():
    return

  from IPython import display
  from IPython.display import HTML
  def set_css_in_cell_output():
    display.display(HTML('''
      <style>
          .jupyter-widgets {color: #d5d5d5 !important;}
          .widget-label {color: #d5d5d5 !important;}
      </style>
    '''))
  get_ipython().events.register('pre_run_cell', set_css_in_cell_output)

if is_notebook():
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

def get_tqdm():
  return tqdm

def get_last_file(directory, pattern):
  import os, re
  def key(f):
    index = re.match(pattern, f).group('index')
    return 0 if index == '' else int(index)
  files = [f for f in os.listdir(directory) if re.match(pattern, f)]
  if len(files) == 0:
    return None, None
  files.sort(key=key)
  index = key(files[-1])
  return files[-1], index

def get_next_file(directory, pattern, templates):
  import os, re
  files = [f for f in os.listdir(directory) if re.match(pattern, f)]
  if len(files) == 0:
    return templates[0], 0
  def key(f):
    index = re.match(pattern, f).group('index')
    return 0 if index == '' else int(index)
  files.sort(key=key)
  n = len(templates)-1
  for i, f in enumerate(files):
    index = key(f)
    if i != index:
      return (templates[0],0) if i == 0 else (re.sub(pattern, lambda m:f"{m.group('pre')}{i}{m.group('post')}", templates[min(i,n)]), i)
  return re.sub(pattern, lambda m:f"{m.group('pre')}{i+1}{m.group('post')}", templates[min(i,n)]), i+1


def make_hbox(im, fig):
  #https://stackoverflow.com/questions/51315566/how-to-display-the-images-side-by-side-in-jupyter-notebook/51316108
  import io
  import ipywidgets as widgets
  from ipywidgets import Layout
  with io.BytesIO() as buf:
    im.save(buf, format="png")
    buf.seek(0)
    wi1 = widgets.Image(value=buf.read(), format='png', layout = Layout(border='0',margin='0',padding='0'))
  with io.BytesIO() as buf:
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    wi2 = widgets.Image(value=buf.read(), format='png', layout = Layout(border='0',margin='0',padding='0'))
  return widgets.HBox([wi1, wi2], layout = Layout(border='0',margin='0',padding='0', align_items='flex-start'))

def load_settings(settings_string, random_seed = True):
  import json, random
  from bunch import Bunch
  params = Bunch(json.loads(settings_string))
  if random_seed or params.seed is None:
    params.seed = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)
    print("using seed:", params.seed)
  return params

def write_settings(settings_dict, f):
  import json
  json.dump(settings_dict, f)
  f.write('\n\n')
  params = settings_dict
  scenes = [(params.scene_prefix + scene + params.scene_suffix).strip() for scene in params.scenes.split('||') if scene]
  for i,scene in enumerate(scenes):
    frame = i * params.steps_per_scene/params.save_every
    f.write(str(f'{frame:.2f}: {scene}'.encode('utf-8', 'ignore')))
    f.write('\n')

def save_settings(settings_dict, path):
  with open(path, 'w+') as f:
    write_settings(settings_dict, f)

def save_batch(settings_list, path):
  from bunch import Bunch
  with open(path, 'w+') as f:
    for batch_index, settings_dict in enumerate(settings_list):
      f.write(f'batch_index: {batch_index}')
      f.write('\n')
      write_settings(Bunch(settings_dict), f)
      f.write('\n\n')


CLIP_MODEL_NAMES = None

# parameter key -> (open_clip architecture, pretrained tag, label)
OPENCLIP_MODELS = {
    "ConvNeXtLarge": ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup", "ConvNeXt-Large"),
    "DFN_ViT_H": ("ViT-H-14-quickgelu", "dfn5b", "ViT-H-14 (DFN)"),
    "MetaCLIP_ViT_B": ("ViT-B-16-quickgelu", "metaclip_400m", "ViT-B-16 (MetaCLIP 400M)"),
    "MetaCLIP_ViT_B_fullcc": ("ViT-B-16-quickgelu", "metaclip_fullcc", "ViT-B-16 (MetaCLIP 2.5B)"),
    "SigLIP2_ViT_B": ("ViT-B-16-SigLIP2-256", "webli", "ViT-B-16 SigLIP 2 (256px)"),
    "PE_Core_B": ("PE-Core-B-16", "meta", "Perception Encoder Core B/16"),
    "MobileCLIP2_S2": ("MobileCLIP2-S2", "dfndr2b", "MobileCLIP2-S2"),
    "SigLIP2_ViT_B32": ("ViT-B-32-SigLIP2-256", "webli", "ViT-B-32 SigLIP 2 (256px)"),
    "MobileCLIP2_S0": ("MobileCLIP2-S0", "dfndr2b", "MobileCLIP2-S0"),
    # adversarially fine-tuned (FARE, eps 4/255) vision towers of LAION models: their pixel gradients are
    # "perceptually aligned", so an image forms in fewer steps. Text towers are the originals.
    "ViTB32_FARE4": ("hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K", None, "ViT-B/32 robust (FARE4)"),
    "ViTB16_FARE4": ("hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K", None, "ViT-B/16 robust (FARE4)"),
    "ConvNeXtB_FARE4": ("hf-hub:chs20/FARE4-convnext_base_w-laion2B-s13B-b82K-augreg", None, "ConvNeXt-B robust (FARE4)"),
}
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def _siglip2_tokenizer(repo, context_length):
    """SigLIP 2's Gemma tokenizer.json is in the newer `tokenizers` format (BPE merges as pairs), which
    the tokenizers version transformers 4.40 needs can't read. Convert it once (merges -> "a b" strings;
    verified to give identical ids) into clip_tokenizers/<arch> and load that."""
    import json, os, shutil
    from huggingface_hub import hf_hub_download
    from open_clip.tokenizer import HFTokenizer
    out = os.path.join("clip_tokenizers", repo.split("/")[-1])
    if not os.path.isfile(os.path.join(out, "tokenizer.json")):
        os.makedirs(out, exist_ok=True)
        for f in ("tokenizer_config.json", "special_tokens_map.json"):
            shutil.copy(hf_hub_download(repo, f), os.path.join(out, f))
        d = json.load(open(hf_hub_download(repo, "tokenizer.json"), encoding="utf-8"))
        m = d["model"]
        m["merges"] = [" ".join(x) if isinstance(x, list) else x for x in m["merges"]]
        m.pop("ignore_merges", None)
        json.dump(d, open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return HFTokenizer(out, context_length=context_length, clean="canonicalize")

def _as_openai_vit(visual):
    """open_clip's plain ViT image tower rebuilt as OpenAI CLIP's VisionTransformer (same weights, same maths:
    identical fp32 output). torch.compile makes OpenAI's version much faster (ViT-B/16: 21 vs 30 ms per
    20-cutout step). Returns None when the tower has anything OpenAI's lacks (attention pooling, layer scale,
    other pooling, a different activation)."""
    import torch.nn as nn
    from open_clip.transformer import VisionTransformer, QuickGELU
    from CLIP.clip.model import VisionTransformer as OpenAIViT
    v = visual
    if type(v) is not VisionTransformer or v.attn_pool is not None or v.pool_type != "tok" \
            or getattr(v, "final_ln_after_pool", False) or not isinstance(v.patch_dropout, nn.Identity) \
            or v.proj is None or v.conv1.bias is not None:
        return None
    blocks = v.transformer.resblocks
    act = blocks[0].mlp.gelu
    if any(not isinstance(b.ls_1, nn.Identity) or not isinstance(b.ls_2, nn.Identity) or hasattr(b, "ln_1_kv")
           or type(b.mlp.gelu) is not type(act) for b in blocks):
        return None
    if isinstance(act, nn.GELU) and act.approximate == "none":
        new_act = nn.GELU
    elif isinstance(act, QuickGELU):
        new_act = None                             # OpenAI's blocks already use QuickGELU
    else:
        return None
    res = v.image_size[0] if isinstance(v.image_size, (tuple, list)) else v.image_size
    ov = OpenAIViT(res, v.conv1.kernel_size[0], v.conv1.out_channels, len(blocks), blocks[0].attn.num_heads,
                   v.proj.shape[1])
    if new_act is not None:
        for b in ov.transformer.resblocks:
            b.mlp.gelu = new_act()
    try:
        ov.load_state_dict(v.state_dict(), strict=True)
    except RuntimeError:
        return None
    p = next(v.parameters())
    return ov.to(device=p.device, dtype=p.dtype).eval().requires_grad_(False)

def load_openclip(arch, tag, device):
    """An OpenCLIP model set up the way pytti uses CLIP: frozen, eval mode, `visual.input_resolution`,
    plus `pytti_normalize` / `pytti_tokenize` when its input scaling or tokenizer differ from OpenAI CLIP's."""
    import open_clip
    from pytti import _ClipNormalize
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag, device=device)
    if "MobileCLIP" in arch:                       # fold the training-time branches (MobileOne / FastViT)
        from timm.utils.model import reparameterize_model
        model = reparameterize_model(model)
    model = model.eval().requires_grad_(False)     # weights are never trained: skip their gradients
    ov = _as_openai_vit(model.visual)
    if ov is not None:
        model.visual = ov
    if "PE-Core" in arch:                          # torch.compile (torch 2.2) gets its rotary-attention backward wrong
        model.pytti_no_compile = True
    if not hasattr(model.visual, 'input_resolution'):
        res = getattr(model.visual, 'image_size', 224)
        model.visual.input_resolution = res[0] if isinstance(res, (tuple, list)) else res
    cfg = getattr(model.visual, "preprocess_cfg", {}) or {}
    mean, std = tuple(cfg.get("mean", _CLIP_MEAN)), tuple(cfg.get("std", _CLIP_STD))
    if mean != _CLIP_MEAN or std != _CLIP_STD:
        model.pytti_normalize = _ClipNormalize(mean, std)
    tcfg = (open_clip.get_model_config(arch) or {}).get("text_cfg", {})    # None for hf-hub: models
    if "SigLIP2" in arch:
        model.pytti_tokenize = _siglip2_tokenizer(tcfg.get("hf_tokenizer_name") or "timm/" + arch,
                                                  tcfg.get("context_length", 64))
    elif tcfg.get("hf_tokenizer_name") or tcfg.get("context_length", 77) != 77:
        model.pytti_tokenize = open_clip.get_tokenizer(arch)
    return model

def load_clip(params):
    from pytti import Perceptor
    import torch 
    import open_clip

    # ------------------------------------------------------
    # PART 1: Define Standard OpenAI Models
    # ------------------------------------------------------
    global CLIP_MODEL_NAMES
    if CLIP_MODEL_NAMES is not None:
        last_names = CLIP_MODEL_NAMES
    else:
        last_names = []
    CLIP_MODEL_NAMES = []
    
    # Standard flags
    if params.RN50x4:  CLIP_MODEL_NAMES.append("RN50x4")
    if params.RN50:    CLIP_MODEL_NAMES.append("RN50")
    if params.ViTB32:  CLIP_MODEL_NAMES.append("ViT-B/32")
    if params.ViTB16:  CLIP_MODEL_NAMES.append("ViT-B/16")
    # OpenCLIP models, loaded after the OpenAI ones in this order (cell 2.15 relies on it)
    for key in OPENCLIP_MODELS:
        if getattr(params, key, False):
            CLIP_MODEL_NAMES.append(key)

    if CLIP_MODEL_NAMES == []:
        Perceptor.free_clip()
        raise RuntimeError("Please select at least one CLIP model")

    Perceptor.free_clip()
    print("Loading CLIP...")

    standard_models = [m for m in CLIP_MODEL_NAMES if m not in OPENCLIP_MODELS]
    if standard_models:
        Perceptor.init_clip(standard_models)
    else:
        Perceptor.CLIP_PERCEPTORS = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for key, (arch, tag, label) in OPENCLIP_MODELS.items():
        if key in CLIP_MODEL_NAMES:
            print(f"Loading OpenCLIP: {label}...")
            Perceptor.CLIP_PERCEPTORS.append(Perceptor.cached_perceptor(
                key, lambda arch=arch, tag=tag: load_openclip(arch, tag, device)))
    Perceptor.park_unused(Perceptor.CLIP_PERCEPTORS)

    print("CLIP loaded.")

def get_frames(path):
  """reads the frames of the mp4 file `path` and returns them as a list of PIL images"""
  import imageio, subprocess
  from PIL import Image
  from os.path import exists as path_exists
  if not path_exists(path+'_converted.mp4'):
    print(f'Converting {path}...')
    subprocess.run(['ffmpeg', '-i', path, path+'_converted.mp4'])
    print(f'Converted {path} to {path}_converted.mp4.')
    print(f'WARNING: future runs will automatically use {path}_converted.mp4, unless you delete it.')
  vid = imageio.get_reader(path+'_converted.mp4',  'ffmpeg')
  n_frames = vid._meta['nframes']
  print(f'loaded {n_frames} frames. for {path}')
  return vid

def build_loss(weight_name, weight, name, img, pil_target):
  from pytti.LossAug import LOSS_DICT
  weight_name, suffix = weight_name.split('_', 1)
  if weight_name == 'direct':
    Loss = type(img).get_preferred_loss()
  else:
    Loss = LOSS_DICT[weight_name]
  out = Loss.TargetImage(f"{weight_name} {name}:{weight}", img.image_shape, pil_target)
  out.set_enabled(pil_target is not None)
  return out

def format_params(params, *args):
  return [params[x] for x in args]

rotoscopers = []

def clear_rotoscopers():
  global rotoscopers
  rotoscopers = []

def update_rotoscopers(frame_n):
  global rotoscopers
  for r in rotoscopers:
    r.update(frame_n)

from PIL import Image

class Rotoscoper:
  def __init__(self, video_path, target = None, thresh = None):
    global rotoscopers
    if video_path[0] == '-':
      video_path = video_path[1:]
      inverted = True
    else:
      inverted = False
    
    self.frames = get_frames(video_path)
    self.target = target
    self.inverted = inverted
    rotoscopers.append(self)
  def update(self, frame_n):
    if self.target is None:
      return
    mask_pil = Image.fromarray(self.frames.get_data(frame_n)).convert('L')
    self.target.set_mask(mask_pil, self.inverted)

