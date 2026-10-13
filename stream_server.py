#!/usr/bin/env python
"""pytti studio: the Colab notebook as a local web app.

    cd ~/pytti && venv/bin/python pytti/stream_server.py            # then open http://127.0.0.1:8765

Two modes on one page:
  Live    renders continuously and streams to the page; every change applies at the next frame.
  Render  runs the notebook's own 2.3 pipeline (scenes, interpolation, stabilisation, video source,
          restore, PNG / MP4 / stream output) with the same parameters, previewed live. Stop ends it
          like an interrupted notebook run: everything written so far is kept.

The notebook stays the single source of truth: cells 2.1, 2.15, 2.17.2, 2.22 and 2.3 are read from
the .ipynb next to this file, and the page's controls are generated from their #@param lines.
Output-neutral optimizations are always on; the ones that can change the image are exposed.
Settings persist in ~/pytti/studio_settings.json.

Listens on localhost only by default. Motion fields are evaluated as Python expressions, so only
use --host 0.0.0.0 on a network you trust.
"""
import argparse, ast, gc, io, json, math, os, re, sys, threading, time, traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]   # don't shadow modules with pytti/*
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "AdaBins"))
os.chdir(ROOT)                      # AdaBins / Depth-Anything / outputs / compile cache are relative to ~/pytti

NOTEBOOK = os.path.join(HERE, "Pytti_6_4_(NeuralCellularAutomata_New_Optimization_and_Methods).ipynb")
SETTINGS_FILE = os.path.join(ROOT, "studio_settings.json")
CELLS = ("#@title #2.1", "#@title 2.15", "#@title 2.17.2", "#@title 2.22", "#@title 2.3")

# Fresh-install defaults: streaming-friendly (ViT-B/32 only, 1 step/frame, ~35 fps on a 4080).
STREAM_DEFAULTS = {
    "2.1": dict(ViTB32=True, ViTB16=False, RN50=False, RN50x4=False, MetaCLIP_ViT_B=False,
                ConvNeXtLarge=False, DFN_ViT_H=False, ViTL14=False,
                width=900, height=1600, cutouts=20, steps_per_frame=1, save_every=1, display_every=1,
                backups=0, translate_z_3d="20", learning_rate=0.05, animation_mode="3D"),
    "2.15": dict(frame_output='"stream"'),     # Render streams by default (switch to video / png in Output to save)
}
# Output-neutral 2.15 switches: always on, not shown.
BAKED = {
    "2.15": dict(enable_torch_compile="True", fix_dataframe_cpu="True", sanitize_grads="True",
                 no_sync_logging="True", compile_mode='"default"', persistent_compile_cache="True",
                 fast_depth="True", async_save="True", live_preview="False", quiet_frames="True",
                 profile_steps="False", profile_every="20", preview_jpeg_quality="85"),
    "2.3": dict(batch_mode="False"),
}

# 2.1 keys -> page sections (anything unlisted lands in "Other")
GROUPS_21 = [
    ("Prompts", "scenes scene_prefix scene_suffix interpolation_steps steps_per_scene direct_image_prompts "
                "init_image direct_init_weight semantic_init_weight"),
    ("Animation", "animation_mode steps_per_frame frames_per_second pre_animation_steps translate_x translate_y "
                  "translate_z_3d rotate_3d rotate_2d zoom_x_2d zoom_y_2d lock_camera field_of_view near_plane "
                  "far_plane sampling_mode infill_mode"),
    ("Optimizer & cutouts", "learning_rate reset_lr_each_frame cutouts cut_pow cutout_border border_mode seed"),
    ("CLIP models", "ViTB32 ViTB16 RN50 RN50x4 ViTL14 ConvNeXtLarge DFN_ViT_H MetaCLIP_ViT_B"),
    ("Image", "image_model width height pixel_size smoothing_weight random_initial_palette palette_size palettes "
              "gamma hdr_weight palette_normalization_weight target_palette lock_palette show_palette vqgan_model"),
    ("Stabilization", "direct_stabilization_weight semantic_stabilization_weight depth_stabilization_weight "
                      "edge_stabilization_weight flow_stabilization_weight"),
    ("Video source", "video_path frame_stride reencode_each_frame flow_long_term_samples"),
    ("Saving", "file_namespace allow_overwrite save_every backups display_every clear_every display_scale "
               "show_graphs approximate_vram_usage"),
]
GROUPS_CELL = {   # 2.15 / 2.17.2 / 2.22 / 2.3 keys -> sections
    "Output": "frame_output stream_target stream_fps stream_bitrate video_quality",
    "Depth": "depth_model depth_match_adabins depth_scale depth_flip_tta depth_max_area depth_every "
             "use_midas_depth midas_model midas_invert",
    "Image": "enable_monochrome mono_levels mono_palettes",
    "Quality-affecting optimizations": "enable_tf32 enable_amp fp16_clip_weights antialiased_cutouts vectorized_cutouts "
                                       "fused_augs share_cutouts lite_augs perceptors_per_step slerp_interpolation "
                                       "directional_clip directional_source sgld_sampling sgld_noise",
    "Frame coherence (render only, changes output)": "enable_frame_opt grad_reuse recompute_every adaptive_corrector "
                                                     "adaptive_min_steps adaptive_patience adaptive_tol coast_cutn "
                                                     "coast_cutn_value verbose",
    "Restore (render)": "restore reencode restore_run restore_frame",
}
SECTION_ORDER = ["Prompts", "Animation", "Optimizer & cutouts", "CLIP models", "Image", "Depth", "Output",
                 "Saving", "Stabilization", "Video source", "Quality-affecting optimizations",
                 "Frame coherence (render only, changes output)", "Restore (render)", "Other"]
OPEN_SECTIONS = {"Prompts", "Animation"}

# how a live change is applied (2.1 keys; cell keys re-run their cell)
LIVE_MODELS = set("ViTB32 ViTB16 RN50 RN50x4 ViTL14 ConvNeXtLarge DFN_ViT_H MetaCLIP_ViT_B".split())
LIVE_PROMPTS = {"scenes", "scene_prefix", "scene_suffix"}
LIVE_IMAGE = set("image_model width height pixel_size palette_size palettes gamma hdr_weight "
                 "palette_normalization_weight random_initial_palette target_palette lock_palette init_image seed".split())
LIVE_EMBEDDER = {"cutouts", "cut_pow", "cutout_border", "border_mode"}
CELL_REBUILD_MODELS = {"enable_torch_compile", "fp16_clip_weights"}
CELL_REBUILD_IMAGE = {"enable_monochrome", "mono_levels", "mono_palettes"}

PARAM_RE = re.compile(r"^(\s*)(\w+)\s*=\s*(.*?)\s*#\s*@param\s*(.*)$")


def _cell_sources():
    nb = json.load(open(NOTEBOOK))
    out = {}
    for c in nb["cells"]:
        src = "".join(c["source"]) if c["cell_type"] == "code" else ""
        for t in CELLS:
            if src.lstrip().startswith(t):
                out[t.split()[-1].lstrip("#")] = src
    missing = [t for t in CELLS if t.split()[-1].lstrip("#") not in out]
    if missing:
        raise RuntimeError(f"notebook is missing cells: {missing}")
    return out


def parse_form(src):
    """[(key, literal_text, type, options)] for every #@param line, in order (first occurrence wins)."""
    seen, out = set(), []
    for line in src.splitlines():
        m = PARAM_RE.match(line)
        if not m or m.group(2) in seen:
            continue
        key, lit, spec = m.group(2), m.group(3), m.group(4).strip()
        seen.add(key)
        if spec.startswith("["):
            try:
                opts = ast.literal_eval(spec)
            except Exception:
                opts = []
            out.append((key, lit, "select", [str(o) for o in opts]))
        else:
            t = re.search(r'type\s*:\s*"(\w+)"', spec)
            out.append((key, lit, t.group(1) if t else "raw", None))
    return out


def set_form_value(src, key, literal):
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*#\s*@param.*)$", re.M)
    return pat.sub(lambda m: m.group(1) + literal + m.group(3), src, count=1)


def strip_magics(src):
    return re.sub(r"^(\s*)[%!].*$", r"\1pass", src, flags=re.M)


def to_literal(type_, value):
    """page value -> Python source literal for a #@param line"""
    if type_ == "boolean":
        return "True" if value in (True, "true", "True", 1, "1", "on") else "False"
    if type_ in ("number", "integer"):
        v = float(value)
        return str(int(v)) if (type_ == "integer" or v.is_integer()) else repr(v)
    if type_ in ("string", "select"):
        return json.dumps(str(value))
    return str(value).strip() or "None"     # raw: an expression


def literal_to_value(type_, lit, env):
    try:
        v = eval(lit, env)
    except Exception:
        return lit
    return v


class Tee(io.TextIOBase):
    """stdout -> terminal + ring buffer shown on the page"""
    def __init__(self, real, lines=None, lock=None):
        self.real, self.cur = real, ""
        self.lines = lines if lines is not None else deque(maxlen=300)
        self.lock = lock or threading.Lock()

    def write(self, s):
        self.real.write(s)
        with self.lock:
            for ch in re.split(r"(\n|\r)", s):
                if ch == "\n":
                    self.lines.append(self.cur); self.cur = ""
                elif ch == "\r":
                    self.cur = ""
                else:
                    self.cur += ch
        return len(s)

    def flush(self):
        self.real.flush()

    def tail(self, n=60):
        with self.lock:
            return list(self.lines)[-n:] + ([self.cur] if self.cur else [])


class FramePublisher:
    """GPU frame -> pinned buffer (non-blocking) -> encoder thread -> latest JPEG for every viewer.
    If the encoder is still busy, the preview skips that frame; rendering never waits on viewers."""
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.seq, self.busy, self.buf, self._job = None, 0, False, None, None
        self.quality, self.scale = 85, 1.0
        threading.Thread(target=self._encoder, daemon=True).start()

    def push_image(self, img):
        import torch
        import torch.nn.functional as Fn
        from pytti import named_rearrange
        if self.busy:
            return
        with torch.no_grad():
            t = img.decode_tensor().detach()
            if self.scale < 0.999:
                h, w = t.shape[-2:]
                t = Fn.interpolate(t.float(), size=(max(1, int(h * self.scale)), max(1, int(w * self.scale))),
                                   mode="bilinear", antialias=True, align_corners=False)
            u8 = named_rearrange(t, img.output_axes, ('y', 'x', 's')).mul(255).clamp(0, 255)
            u8 = u8.to(torch.uint8).contiguous()
        if self.buf is None or self.buf.shape != u8.shape:
            self.buf = torch.empty(u8.shape, dtype=torch.uint8, pin_memory=True)
        self.buf.copy_(u8, non_blocking=True)
        ev = torch.cuda.Event(); ev.record()
        with self.cond:
            self.busy, self._job = True, ev
            self.cond.notify_all()

    def _encoder(self):
        from PIL import Image
        while True:
            with self.cond:
                while self._job is None:
                    self.cond.wait()
                ev, self._job = self._job, None
            try:
                ev.synchronize()
                b = io.BytesIO()
                Image.fromarray(self.buf.numpy()).save(b, "JPEG", quality=int(self.quality))
                with self.cond:
                    self.jpeg, self.seq = b.getvalue(), self.seq + 1
                    self.cond.notify_all()
            except Exception:
                traceback.print_exc()
            finally:
                with self.cond:
                    self.busy = False


class Studio:
    def __init__(self, tee):
        self.tee = tee
        self.lock = threading.Lock()
        self.pending, self.actions = {}, set()
        self.pub = FramePublisher()
        self.mode, self.paused, self.cancel = "live", False, False
        self.status = dict(mode="live", state="starting", fps=0.0, step_ms=0.0, frame=0, loss=None,
                           error=None, models=[], paused=False, message="loading notebook cells")
        self.model = self.img = self.embedder = None
        self.src = _cell_sources()
        self.forms = {c: parse_form(s) for c, s in self.src.items()}
        self.cell_of = {k: c for c, f in self.forms.items() for (k, *_ ) in f if c != "2.1"}
        self.type_of = {k: t for f in self.forms.values() for (k, _, t, _) in f}
        self.G = {"__name__": "__pytti_studio__"}
        self._load_settings()
        self._exec_21()
        for c in ("2.15", "2.17.2", "2.22"):
            self._exec_cell(c)

    # ---- settings ----------------------------------------------------------------------------------
    def _load_settings(self):
        # cell literals: notebook value <- streaming defaults <- saved settings ; baked always win
        self.lits = {c: {k: lit for k, lit, *_ in f} for c, f in self.forms.items()}
        for c, d in STREAM_DEFAULTS.items():
            if c != "2.1":
                self.lits[c].update(d)
        self.p21 = dict(STREAM_DEFAULTS["2.1"])
        if os.path.exists(SETTINGS_FILE):
            try:
                saved = json.load(open(SETTINGS_FILE))
                self.p21.update(saved.get("2.1", {}))
                for c, d in saved.get("cells", {}).items():
                    self.lits.setdefault(c, {}).update(d)
            except Exception:
                traceback.print_exc()
        for c, d in BAKED.items():
            self.lits[c].update(d)

    def _save_settings(self):
        p = self.p
        keys21 = [k for k, *_ in self.forms["2.1"]]
        data = {"2.1": {k: p.get(k) for k in keys21 if k in p},
                "cells": {c: {k: v for k, v in d.items() if k not in BAKED.get(c, {})}
                          for c, d in self.lits.items() if c != "2.1"}}
        tmp = SETTINGS_FILE + ".tmp"
        json.dump(data, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, SETTINGS_FILE)

    def _exec_21(self):
        exec(compile(strip_magics(self.src["2.1"]), "cell 2.1", "exec"), self.G)
        p = self.G["params"]
        for k, v in self.p21.items():
            p[k] = v
        self.p = p

    def _cell_source(self, c):
        s = self.src[c]
        for k, lit in self.lits[c].items():
            s = set_form_value(s, k, lit)
        return s

    def _exec_cell(self, c):
        exec(compile(strip_magics(self._cell_source(c)), f"cell {c}", "exec"), self.G)
        if c == "2.15":
            self.OPT = self.G["OPT"]
            self.OPT["_need_next_pil"] = False

    # ---- page state ----------------------------------------------------------------------------------
    def schema(self):
        sec = {}
        def put(section, item):
            sec.setdefault(section, []).append(item)
        g21 = {k: g for g, ks in GROUPS_21 for k in ks.split()}
        for k, lit, t, opts in self.forms["2.1"]:
            v = self.p.get(k)
            put(g21.get(k, "Other"), dict(key=k, type=t, options=opts, value=v if t != "raw" else repr(v)
                                          if not isinstance(v, str) else v))
        gc_ = {k: g for g, ks in GROUPS_CELL.items() for k in ks.split()}
        for c in ("2.15", "2.17.2", "2.22", "2.3"):
            for k, lit, t, opts in self.forms[c]:
                if k in BAKED.get(c, {}) or k not in gc_:
                    continue
                cur = self.lits[c].get(k, lit)
                val = literal_to_value(t, cur, {}) if t != "raw" else cur
                put(gc_[k], dict(key=k, type=t, options=opts, value=val))
        return [[s, sec[s]] for s in SECTION_ORDER if s in sec]

    def state(self):
        st = dict(self.status)
        st["log"] = self.tee.tail(40)
        return dict(status=st)

    # ---- changes -----------------------------------------------------------------------------------
    def submit(self, changes, actions):
        with self.lock:
            self.pending.update(changes)
            self.actions |= set(actions)
            if "stop" in actions:
                self.cancel = True

    def _value_21(self, k, v):
        t = self.type_of.get(k, "raw")
        if t == "boolean":
            return v in (True, "true", "True", 1, "1", "on")
        if t in ("number", "integer"):
            v = float(v)
            return int(v) if v.is_integer() else v
        if t in ("string", "select"):
            return str(v)
        try:                                        # raw: number / JSON / expression
            return json.loads(v) if isinstance(v, str) else v
        except Exception:
            return eval(str(v), {"math": math, "inf": math.inf, "all": math.inf, **self.p})

    def _apply_pending(self):
        with self.lock:
            changes, actions = self.pending, self.actions
            self.pending, self.actions = {}, set()
        if not changes and not actions:
            return set()
        p = self.p
        what, rerun = set(), []
        for k, v in changes.items():
            if k in ("view_scale", "jpeg_quality"):
                setattr(self.pub, "scale" if k == "view_scale" else "quality", float(v))
                continue
            c = self.cell_of.get(k)
            if c:
                if k in BAKED.get(c, {}):
                    continue
                self.lits[c][k] = to_literal(self.type_of[k], v)
                if c not in rerun:
                    rerun.append(c)
                if k in CELL_REBUILD_MODELS: what.add("models")
                if k in CELL_REBUILD_IMAGE: what.add("image")
                continue
            if k not in self.type_of:
                continue
            p[k] = self._value_21(k, v)
            if k in LIVE_MODELS: what.add("models")
            elif k in LIVE_PROMPTS: what.add("prompts")
            elif k in LIVE_IMAGE: what.add("image")
            elif k in LIVE_EMBEDDER: what.add("embedder")
            elif k == "learning_rate": what.add("lr")
            elif k == "smoothing_weight": what.add("tv")
        for c in ("2.15", "2.17.2", "2.22"):
            if c in rerun:
                self._exec_cell(c)
        if "2.15" in rerun:
            what.add("models")         # the embedder patch was re-installed
        if changes:
            self._save_settings()
        for a in actions:
            what.add(a)
        if "defaults" in what:
            if os.path.exists(SETTINGS_FILE):
                os.remove(SETTINGS_FILE)
            self._load_settings(); self._exec_21()
            for c in ("2.15", "2.17.2", "2.22"):
                self._exec_cell(c)
            what |= {"models", "image"}
        if "pause" in what:
            self.paused = not self.paused
            self.status["paused"] = self.paused
        return what

    # ---- live mode ---------------------------------------------------------------------------------
    def _free(self):
        import torch
        self.model = self.img = self.embedder = None
        self.prompts = []
        gc.collect(); torch.cuda.empty_cache()

    def _build_models(self):
        from pytti.Notebook import load_clip
        from pytti.Perceptor.Embedder import HDMultiClipEmbedder
        p = self.p
        self.status["message"] = "loading CLIP models"
        self.embedder = None
        gc.collect()
        load_clip(p)
        self.embedder = HDMultiClipEmbedder(cutn=int(p.cutouts), cut_pow=p.cut_pow,
                                            padding=p.cutout_border, border_mode=p.border_mode)
        self.status["models"] = [k for k in "ViTB32 ViTB16 RN50 RN50x4 ViTL14 MetaCLIP_ViT_B ConvNeXtLarge DFN_ViT_H".split()
                                 if p.get(k)]
        self._parse_prompts()
        if self.model is not None:
            self.model.embedder = self.embedder

    def _parse_prompts(self):
        from pytti.Perceptor.Prompt import parse_prompt
        p = self.p
        text = (p.scene_prefix + p.scenes.split("||")[0] + p.scene_suffix).strip()
        self.prompts = [parse_prompt(self.embedder, s.strip()) for s in text.split("|") if s.strip()]

    def _build_image(self):
        import builtins, torch
        from PIL import Image
        from pytti import fetch
        from pytti.Image import PixelImage
        from pytti.ImageGuide import DirectImageGuide
        from pytti.LossAug import TVLoss
        from pytti.Notebook import format_params
        p = self.p
        if p.image_model != "Limited Palette":
            raise RuntimeError(f"Live mode supports the Limited Palette image model; use Render for {p.image_model}")
        self.status["message"] = "building image"
        self.model = self.img = None
        gc.collect()
        if p.seed is not None and str(p.seed).strip() not in ("", "None"):
            torch.manual_seed(int(float(p.seed)) % (2 ** 63))
        self.img = PixelImage(*format_params(p, 'width', 'height', 'pixel_size', 'palette_size', 'palettes',
                                             'gamma', 'hdr_weight', 'palette_normalization_weight'))
        self.img.encode_random(random_pallet=p.random_initial_palette)
        if p.target_palette.strip():
            self.img.set_pallet_target(Image.open(fetch(p.target_palette)).convert('RGB'))
        else:
            self.img.lock_pallet(p.lock_palette)
        if str(p.init_image).strip():
            self.img.encode_image(Image.open(fetch(p.init_image)).convert('RGB'), smart_encode=True)
        self.tv = TVLoss(weight=p.smoothing_weight)
        self.model = DirectImageGuide(self.img, self.embedder, lr=p.learning_rate)
        self.frame, self.i = 0, 0

    def _live_apply(self, what):
        p = self.p
        if "models" in what or self.embedder is None:
            self._build_models()
        elif "prompts" in what:
            self._parse_prompts()
        if "embedder" in what and self.embedder is not None:
            self.embedder.cutn, self.embedder.cut_pow = int(p.cutouts), float(p.cut_pow)
            self.embedder.padding, self.embedder.border_mode = float(p.cutout_border), p.border_mode
        if "image" in what or "reset" in what or self.model is None:
            self._build_image()
        if "lr" in what:
            for g in self.model.optimizer.param_groups:
                g["lr"] = float(p.learning_rate)
        if "tv" in what:
            self.tv.set_weight(float(p.smoothing_weight))

    def _live_frame(self):
        import pytti
        import pytti.Transforms as T
        p = self.p
        pytti.set_t(self.frame / max(1e-6, float(p.frames_per_second)))
        if self.frame > 0:
            if p.animation_mode == "3D":
                T.zoom_3d(self.img, (p.translate_x, p.translate_y, p.translate_z_3d), p.rotate_3d,
                          p.field_of_view, p.near_plane, p.far_plane, border_mode=p.infill_mode,
                          sampling_mode=p.sampling_mode, stabilize=p.lock_camera)
            elif p.animation_mode == "2D":
                ev = pytti.parametric_eval
                T.zoom_2d(self.img, (ev(p.translate_x), ev(p.translate_y)), (ev(p.zoom_x_2d), ev(p.zoom_y_2d)),
                          ev(p.rotate_2d), border_mode=p.infill_mode, sampling_mode=p.sampling_mode)
        augs = [self.tv] if float(p.smoothing_weight) != 0 else []
        losses = None
        for _ in range(max(1, int(p.steps_per_frame))):
            losses = self.model.train(self.i, self.prompts, [], augs, interp_steps=0)
            self.i += 1
        self.frame += 1
        self.pub.push_image(self.img)
        return losses

    # ---- render mode ---------------------------------------------------------------------------------
    def _render(self):
        from pytti.ImageGuide import DirectImageGuide
        p, studio = self.p, self
        self._free()
        self.cancel = False
        self.status.update(mode="render", state="rendering", message="starting render", error=None, frame=0)
        keep = dict(display_every=p.display_every, clear_every=p.clear_every, show_graphs=p.show_graphs)
        p.display_every, p.clear_every, p.show_graphs = 0, 0, False      # the page is the preview
        orig_train = DirectImageGuide.train
        t_last = [time.perf_counter()]; ema = [None]; n = [0]

        def train(self_, *a, **k):
            if studio.cancel:
                raise KeyboardInterrupt
            r = orig_train(self_, *a, **k)
            n[0] += 1
            now = time.perf_counter(); dt = now - t_last[0]; t_last[0] = now
            ema[0] = dt if ema[0] is None else 0.9 * ema[0] + 0.1 * dt
            spf = max(1, int(p.steps_per_frame))
            studio.status.update(step_ms=round(1000 * ema[0], 1), fps=round(1 / (ema[0] * spf), 2),
                                 frame=n[0] // spf, message=f"step {n[0]}")
            if self_.embedder is not None:
                studio.pub.push_image(self_.image_rep)
            # live edits to motion / learning rate also reach a running render
            with studio.lock:
                ch = {k_: v for k_, v in studio.pending.items() if k_ in studio.type_of and k_ not in studio.cell_of}
            for k_, v in ch.items():
                if k_ in LIVE_MODELS | LIVE_IMAGE | LIVE_PROMPTS:
                    continue
                p[k_] = studio._value_21(k_, v)
                if k_ == "learning_rate":
                    for g in self_.optimizer.param_groups:
                        g["lr"] = float(p.learning_rate)
            return r

        DirectImageGuide.train = train
        try:
            self.G["params"] = p
            exec(compile(strip_magics(self._cell_source("2.3")), "cell 2.3", "exec"), self.G)
            self.status.update(message="render stopped" if self.cancel else "render complete")
        except Exception as e:
            self.status.update(error=f"{type(e).__name__}: {e}", message=traceback.format_exc()[-1500:])
            traceback.print_exc()
        finally:
            DirectImageGuide.train = orig_train
            for k_, v in keep.items():
                p[k_] = v
            self.mode, self.cancel = "live", False
            self.status.update(mode="live", state="idle")
            self.paused = True                     # don't jump straight back into live after a render
            self.status["paused"] = True
            self._free()

    # ---- main loop -----------------------------------------------------------------------------------
    def run(self):
        ema, last = None, time.perf_counter()
        first = True
        while True:
            try:
                what = self._apply_pending()
                if "render" in what:
                    self._render()
                    ema, last = None, time.perf_counter()
                    continue
                if self.paused:
                    self.status.update(state="paused")
                    time.sleep(0.05); last = time.perf_counter()
                    continue
                if first or what or self.model is None:
                    self._live_apply(what | ({"models"} if self.embedder is None else set()))
                    first = False
                    self.status.update(state="running", message="", mode="live")
                    last = time.perf_counter()
                losses = self._live_frame()
                now = time.perf_counter(); dt = now - last; last = now
                ema = dt if ema is None else 0.9 * ema + 0.1 * dt
                self.status.update(fps=round(1 / ema, 1), frame=self.frame, error=None, state="running",
                                   step_ms=round(1000 * ema / max(1, int(self.p.steps_per_frame)), 1))
                if self.frame % 12 == 0 and losses is not None:
                    self.status["loss"] = round(float(losses["TOTAL"]), 4)   # the only host sync, every 12 frames
            except Exception as e:
                self.status.update(error=f"{type(e).__name__}: {e}", state="error")
                traceback.print_exc()
                self.paused = True; self.status["paused"] = True
                time.sleep(0.3); last = time.perf_counter()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>pytti studio</title>
<style>
:root{--bg:#0c0d0f;--panel:#131518;--panel2:#181b1f;--line:#23262b;--fg:#dde0e5;--dim:#7c828c;--accent:#a7c7ff;--bad:#ff8f86;--ok:#8fd19e}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.app{display:flex;height:100vh}
.view{flex:1;display:flex;flex-direction:column;min-width:0}
.stage{flex:1;display:flex;align-items:center;justify-content:center;min-height:0;position:relative}
.stage img{max-width:100%;max-height:100%;display:block;image-rendering:auto}
.hud{display:flex;gap:14px;align-items:center;padding:8px 14px;border-top:1px solid var(--line);
font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);white-space:nowrap;overflow:hidden}
.hud b{color:var(--fg);font-weight:500}.dot{width:8px;height:8px;border-radius:50%;background:var(--dim);flex:none}
.dot.running,.dot.rendering{background:var(--ok)}.dot.error{background:var(--bad)}
.err{color:var(--bad);overflow:hidden;text-overflow:ellipsis}
aside{width:340px;flex:none;background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column}
.top{padding:12px 14px 10px;border-bottom:1px solid var(--line)}
.title{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.title h1{font-size:13px;font-weight:600;letter-spacing:.02em;margin:0}.title span{color:var(--dim);font-size:11px}
.seg{display:grid;grid-template-columns:1fr 1fr;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:2px;margin-bottom:8px}
.seg button{border:0;background:transparent;color:var(--dim);padding:6px 0;border-radius:6px;cursor:pointer;font:inherit}
.seg button.on{background:#262a30;color:var(--fg)}
.bar{display:flex;gap:6px}
.btn{flex:1;background:#1d2024;color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:7px 0;cursor:pointer;font:inherit}
.btn:hover{border-color:#383d44}.btn.primary{background:#22324a;border-color:#2e4566;color:#d8e6ff}
.btn.danger{background:#3a2222;border-color:#5a3030;color:#ffd6d2}
.scroll{overflow-y:auto;flex:1;padding:4px 14px 30px}
details{border-bottom:1px solid var(--line)}
summary{cursor:pointer;color:var(--dim);font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:11px 0;list-style:none;display:flex;justify-content:space-between}
summary::-webkit-details-marker{display:none}summary::after{content:"+";color:#555b64}details[open] summary::after{content:"–"}
details[open]{padding-bottom:8px}
.row{display:grid;grid-template-columns:118px 1fr;align-items:center;gap:8px;margin:6px 0}
.row label{color:#a3a9b2;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row.wide{grid-template-columns:1fr}.row.wide label{margin-bottom:-2px}
input[type=text],input[type=number],select,textarea{width:100%;background:#0e1012;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:5px 7px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;outline:none}
input:focus,select:focus,textarea:focus{border-color:#3b4f6e}
textarea{resize:vertical;min-height:72px}
input[type=checkbox]{accent-color:var(--accent);justify-self:start;width:15px;height:15px}
.saved{animation:f .7s}@keyframes f{from{border-color:var(--accent);box-shadow:0 0 0 2px #a7c7ff33}}
.log{font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);white-space:pre-wrap;word-break:break-all;
max-height:220px;overflow:auto;background:#0e1012;border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:4px 0 8px}
.note{color:var(--dim);font-size:11px;margin:10px 0}
@media (max-width:820px){.app{flex-direction:column}aside{width:auto;height:55vh;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<div class="app">
  <div class="view">
    <div class="stage"><img id="v" src="/stream" alt="live render"></div>
    <div class="hud"><span class="dot" id="dot"></span><span id="hud">connecting…</span></div>
  </div>
  <aside>
    <div class="top">
      <div class="title"><h1>pytti studio</h1><span id="sub"></span></div>
      <div class="seg"><button id="m-live" class="on">Live</button><button id="m-render">Render</button></div>
      <div class="bar" id="bar-live"><button class="btn" id="pause">Pause</button><button class="btn" id="reset">Reset image</button></div>
      <div class="bar" id="bar-render" hidden><button class="btn primary" id="start">Start render</button><button class="btn danger" id="stop">Stop</button></div>
    </div>
    <div class="scroll">
      <div id="form"></div>
      <details><summary>Preview & log</summary>
        <div class="row"><label>preview scale</label><input type="number" id="vs" min="0.25" max="1" step="0.05" value="1"></div>
        <div class="row"><label>JPEG quality</label><input type="number" id="jq" min="40" max="95" step="1" value="85"></div>
        <div class="log" id="log"></div>
      </details>
      <div class="note">Text fields apply on Enter (scenes: Ctrl+Enter or leave the box). Motion fields take expressions (t = seconds).
      Model / image changes reload. Settings are saved automatically. <a href="#" id="defaults" style="color:var(--dim)">Restore streaming defaults</a></div>
    </div>
  </aside>
</div>
<script>
const $=s=>document.querySelector(s);
const post=(changes,actions=[])=>fetch('/set',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({changes,actions})});
const pretty=k=>k.replace(/_3d$/,' (3D)').replace(/_2d$/,' (2D)').replace(/_/g,' ');
function flash(el){el.classList.remove('saved');void el.offsetWidth;el.classList.add('saved')}
let openState={};try{openState=JSON.parse(localStorage.getItem('pytti-open')||'{}')}catch(e){}
function field(f){
  const wide=f.key==='scenes'||f.key==='scene_suffix'||f.key==='scene_prefix';
  const row=document.createElement('div');row.className='row'+(wide?' wide':'');
  const lab=document.createElement('label');lab.textContent=pretty(f.key);lab.title=f.key;row.append(lab);
  const send=(v,el)=>{post({[f.key]:v});flash(el)};
  const val=f.value;
  if(f.type==='boolean'){const c=document.createElement('input');c.type='checkbox';c.checked=!!val;
    c.onchange=()=>send(c.checked,c);row.append(c)}
  else if(f.type==='select'){const s=document.createElement('select');
    for(const o of f.options){const op=document.createElement('option');op.value=o;op.textContent=o;s.append(op)}
    s.value=val;s.onchange=()=>send(s.value,s);row.append(s)}
  else if(f.key==='scenes'){const t=document.createElement('textarea');t.value=val??'';
    t.onkeydown=e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)){send(t.value,t);e.preventDefault()}};
    t.onchange=()=>send(t.value,t);row.append(t)}
  else{const t=document.createElement('input');
    t.type=(f.type==='number'||f.type==='integer')?'number':'text';if(t.type==='number')t.step='any';
    t.value=val??'';t.onkeydown=e=>{if(e.key==='Enter')send(t.value,t)};
    if(t.type==='number')t.onchange=()=>send(t.value,t);row.append(t)}
  return row}
async function build(){
  const s=await (await fetch('/schema')).json();const form=$('#form');form.innerHTML='';
  for(const [g,fs] of s){const d=document.createElement('details');
    d.open=g in openState?openState[g]:(g==='Prompts'||g==='Animation');
    d.ontoggle=()=>{openState[g]=d.open;localStorage.setItem('pytti-open',JSON.stringify(openState))};
    const sm=document.createElement('summary');sm.textContent=g;d.append(sm);
    for(const f of fs)d.append(field(f));form.append(d)}}
function setMode(m){$('#m-live').classList.toggle('on',m==='live');$('#m-render').classList.toggle('on',m==='render');
  $('#bar-live').hidden=m!=='live';$('#bar-render').hidden=m!=='render'}
let uiMode='live';
$('#m-live').onclick=()=>{uiMode='live';setMode('live')};$('#m-render').onclick=()=>{uiMode='render';setMode('render')};
$('#pause').onclick=()=>post({},['pause']);$('#reset').onclick=()=>post({},['reset']);
$('#start').onclick=()=>post({},['render']);$('#stop').onclick=()=>post({},['stop']);
$('#defaults').onclick=async e=>{e.preventDefault();if(confirm('Reset every setting to the streaming defaults?')){await post({},['defaults']);setTimeout(build,1500)}};
$('#vs').onchange=e=>post({view_scale:e.target.value});$('#jq').onchange=e=>post({jpeg_quality:e.target.value});
$('#v').onerror=()=>setTimeout(()=>{$('#v').src='/stream?'+Date.now()},1000);
async function poll(){
  try{const s=(await (await fetch('/state')).json()).status;
    $('#dot').className='dot '+s.state;
    const parts=[`<b>${s.fps}</b> fps`,`${s.step_ms} ms/step`,`frame ${s.frame}`];
    if(s.loss!=null)parts.push(`loss ${s.loss}`);if(s.models&&s.models.length)parts.push(s.models.join(' + '));
    parts.push(s.mode==='render'?'rendering':(s.paused?'paused':s.state));
    if(s.message&&!s.error&&s.message.length<80)parts.push(s.message);
    $('#hud').innerHTML=parts.join(' · ')+(s.error?` · <span class="err">${s.error}</span>`:'');
    $('#pause').textContent=s.paused?'Resume':'Pause';
    if(s.mode==='render'&&uiMode!=='render'){uiMode='render';setMode('render')}
    $('#sub').textContent=s.mode==='render'?'rendering':'';
    const lg=$('#log');const atEnd=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-4;
    lg.textContent=(s.log||[]).join('\n');if(atEnd)lg.scrollTop=lg.scrollHeight}
  catch(e){$('#hud').textContent='server unreachable';$('#dot').className='dot error'}
  setTimeout(poll,700)}
build();poll();
</script></body></html>"""


def make_handler(studio):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/schema":
                return self._send(200, json.dumps(studio.schema(), default=str).encode(), "application/json")
            if path == "/state":
                return self._send(200, json.dumps(studio.state(), default=str).encode(), "application/json")
            if path == "/snapshot.jpg":
                return self._send(200, studio.pub.jpeg or b"", "image/jpeg")
            if path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                pub, seen = studio.pub, -1
                try:
                    while True:
                        with pub.cond:
                            pub.cond.wait_for(lambda: pub.seq != seen, timeout=2)
                            jpg, seen = pub.jpeg, pub.seq
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.split("?")[0] != "/set":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            try:
                msg = json.loads(self.rfile.read(n) or b"{}")
                studio.submit(msg.get("changes", {}), msg.get("actions", []))
                self._send(200, b'{"ok":true}', "application/json")
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    tee = Tee(sys.stdout)
    sys.stdout = tee
    sys.stderr = Tee(sys.stderr, tee.lines, tee.lock)     # errors and progress bars land in the page log too
    studio = Studio(tee)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(studio))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"pytti studio: http://{a.host}:{a.port}", flush=True)
    studio.run()      # GPU work stays on the main thread


if __name__ == "__main__":
    main()
