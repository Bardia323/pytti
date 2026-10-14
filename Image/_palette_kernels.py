"""Fused Triton kernels for the two heaviest non-CLIP parts of a Limited Palette training step.

Both compute exactly the same math as the PyTorch code they replace (PixelImage._cont_blend and
PixelImage._pallet_loss_raw); results differ only by float summation order. They exist because the
PyTorch versions make many full passes over the (n_pallets, H, W) tensor, and the palette gradient
of the blend was an index_put with millions of atomic adds into a few hundred palette entries.

    cont_blend(pallet, floors, ceils, fracs, tensor)  -> (H, W, 3)    same as _cont_blend
    pallet_norm_loss(tensor)                          -> scalar      same as _pallet_loss_raw

`supported(...)` says whether the fast path applies (CUDA, fp32, small enough palettes); callers fall
back to the PyTorch code otherwise.
"""
import torch
import triton
import triton.language as tl

BLOCK = 128


def _p2(x, lo=16):
    n = lo
    while n < x:
        n *= 2
    return n


def supported(tensor, pallet=None):
    if not (tensor.is_cuda and tensor.dtype == torch.float32):
        return False
    if tensor.shape[0] > 32:
        return False
    if pallet is not None and (pallet.dtype != torch.float32 or pallet.shape[0] > 64 or pallet.shape[-1] != 3):
        return False
    return True


# =====================================================================================================
# continuous palette blend:  out[y,x,c] = sum_p softmax(t[:,y,x])_p * ((1-f) P[fl,p,c] + f P[cl,p,c])
# =====================================================================================================
@triton.jit
def _blend_fwd(t_ptr, pal_ptr, fl_ptr, cl_ptr, f_ptr, out_ptr, HW, n,
               NP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    pm = pix < HW
    p = tl.arange(0, NP)
    cm = p < n
    m2 = cm[:, None] & pm[None, :]
    t = tl.load(t_ptr + p[:, None] * HW + pix[None, :], mask=m2, other=-float("inf"))
    t = tl.where(pm[None, :], t, 0.0)                       # padded pixels: finite (results unused)
    t = tl.where(cm[:, None], t, -float("inf"))
    mx = tl.max(t, 0)
    e = tl.exp(t - mx[None, :])
    w = e / tl.sum(e, 0)[None, :]
    fl = tl.load(fl_ptr + pix, mask=pm, other=0)
    cl = tl.load(cl_ptr + pix, mask=pm, other=0)
    f = tl.load(f_ptr + pix, mask=pm, other=0.0)
    base_fl = (fl[None, :] * n + p[:, None]) * 3
    base_cl = (cl[None, :] * n + p[:, None]) * 3
    for c in tl.static_range(3):
        a = tl.load(pal_ptr + base_fl + c, mask=m2, other=0.0)
        b = tl.load(pal_ptr + base_cl + c, mask=m2, other=0.0)
        v = tl.sum((a * (1 - f[None, :]) + b * f[None, :]) * w, 0)
        tl.store(out_ptr + pix * 3 + c, v, mask=pm)


@triton.jit
def _split_tf32(x):
    """x = hi + lo with hi exactly representable in TF32 (low 13 mantissa bits cleared)"""
    hi = (x.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
    return hi, x - hi


@triton.jit
def _dot3(a, b):
    """fp32-accurate a @ b from three TF32 tensor-core products (error ~2^-22 vs ~2^-11 for one)"""
    ah, al = _split_tf32(a)
    bh, bl = _split_tf32(b)
    return tl.dot(ah, bh, allow_tf32=True) + (tl.dot(ah, bl, allow_tf32=True) + tl.dot(al, bh, allow_tf32=True))


@triton.jit
def _blend_bwd(t_ptr, pal_ptr, fl_ptr, cl_ptr, f_ptr, g_ptr, gt_ptr, gf_ptr, gpal_ptr, HW, n,
               NP: tl.constexpr, LP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    pm = pix < HW
    p = tl.arange(0, NP)
    cm = p < n
    m2 = cm[:, None] & pm[None, :]
    t = tl.load(t_ptr + p[:, None] * HW + pix[None, :], mask=m2, other=-float("inf"))
    t = tl.where(pm[None, :], t, 0.0)
    t = tl.where(cm[:, None], t, -float("inf"))
    mx = tl.max(t, 0)
    e = tl.exp(t - mx[None, :])
    w = e / tl.sum(e, 0)[None, :]
    w = tl.where(m2, w, 0.0)
    fl = tl.load(fl_ptr + pix, mask=pm, other=0)
    cl = tl.load(cl_ptr + pix, mask=pm, other=0)
    f = tl.load(f_ptr + pix, mask=pm, other=0.0)
    base_fl = (fl[None, :] * n + p[:, None]) * 3
    base_cl = (cl[None, :] * n + p[:, None]) * 3
    lv = tl.arange(0, LP)
    # level weights of each pixel: (1-f) at its floor level, f at its ceil level (both add when equal)
    coef = tl.where(fl[:, None] == lv[None, :], 1 - f[:, None], 0.0) + tl.where(cl[:, None] == lv[None, :], f[:, None], 0.0)
    coef = tl.where(pm[:, None], coef, 0.0)
    dw = tl.zeros([NP, BLOCK], dtype=tl.float32)
    gf = tl.zeros([BLOCK], dtype=tl.float32)
    for c in tl.static_range(3):
        a = tl.load(pal_ptr + base_fl + c, mask=m2, other=0.0)
        b = tl.load(pal_ptr + base_cl + c, mask=m2, other=0.0)
        g = tl.load(g_ptr + pix * 3 + c, mask=pm, other=0.0)
        dw += (a * (1 - f[None, :]) + b * f[None, :]) * g[None, :]
        gf += tl.sum(w * (b - a), 0) * g
        # palette gradient for this channel, summed over the block's pixels: [NP, BLOCK] @ [BLOCK, LP]
        part = _dot3(w * g[None, :], coef)
        tl.store(gpal_ptr + ((pid * 3 + c) * NP + p[:, None]) * LP + lv[None, :], part)
    gt = w * (dw - tl.sum(w * dw, 0)[None, :])
    tl.store(gt_ptr + p[:, None] * HW + pix[None, :], gt, mask=m2)
    tl.store(gf_ptr + pix, gf, mask=pm)


class _ContBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pallet, floors, ceils, fracs, tensor):
        L, n, _ = pallet.shape
        H, W = fracs.shape
        HW = H * W
        pal = pallet.contiguous()
        t = tensor.contiguous()
        fl = floors.contiguous().view(-1)
        cl = ceils.contiguous().view(-1)
        f = fracs.contiguous().view(-1)
        out = torch.empty(H, W, 3, device=t.device, dtype=torch.float32)
        NP = _p2(n)
        grid = (triton.cdiv(HW, BLOCK),)
        _blend_fwd[grid](t, pal, fl, cl, f, out, HW, n, NP=NP, BLOCK=BLOCK)
        ctx.save_for_backward(pal, fl, cl, f, t)
        ctx.dims = (L, n, H, W)
        return out

    @staticmethod
    def backward(ctx, gout):
        pal, fl, cl, f, t = ctx.saved_tensors
        L, n, H, W = ctx.dims
        HW = H * W
        NP, LP = _p2(n), _p2(L)
        nb = triton.cdiv(HW, BLOCK)
        g = gout.contiguous()
        gt = torch.empty_like(t)
        gf = torch.empty(HW, device=t.device, dtype=torch.float32)
        gpal = torch.empty(nb, 3, NP, LP, device=t.device, dtype=torch.float32)
        _blend_bwd[(nb,)](t, pal, fl, cl, f, g, gt, gf, gpal, HW, n, NP=NP, LP=LP, BLOCK=BLOCK)
        gp = gpal.sum(0)[:, :n, :L].permute(2, 1, 0).contiguous()      # (L, n, 3)
        return gp, None, None, gf.view(H, W), gt


def cont_blend(pallet, floors, ceils, fracs, tensor):
    return _ContBlend.apply(pallet, floors, ceils, fracs, tensor)


# =====================================================================================================
# palette normalization loss (anticorrelate palettes, maximise variance within each)
# =====================================================================================================
@triton.jit
def _softmax_tile(t_ptr, pix, HW, n, NP: tl.constexpr):
    pm = pix < HW
    p = tl.arange(0, NP)
    cm = p < n
    m2 = cm[:, None] & pm[None, :]
    t = tl.load(t_ptr + p[:, None] * HW + pix[None, :], mask=m2, other=-float("inf"))
    t = tl.where(pm[None, :], t, 0.0)
    t = tl.where(cm[:, None], t, -float("inf"))
    mx = tl.max(t, 0)
    e = tl.exp(t - mx[None, :])
    x = e / tl.sum(e, 0)[None, :]
    return tl.where(m2, x, 0.0), m2


@triton.jit
def _pn_sum(t_ptr, s_ptr, HW, n, NP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    x, m2 = _softmax_tile(t_ptr, pix, HW, n, NP)
    tl.store(s_ptr + pid * NP + tl.arange(0, NP), tl.sum(x, 1))


@triton.jit
def _pn_cov(t_ptr, mu_ptr, a_ptr, HW, n, NP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    x, m2 = _softmax_tile(t_ptr, pix, HW, n, NP)
    p = tl.arange(0, NP)
    mu = tl.load(mu_ptr + p)
    xc = tl.where(m2, x - mu[:, None], 0.0)
    a = tl.dot(xc, tl.trans(xc), allow_tf32=False)                  # [NP, NP] partial X_c^T X_c (exact fp32)
    tl.store(a_ptr + pid * NP * NP + p[:, None] * NP + p[None, :], a)


@triton.jit
def _pn_bwd(t_ptr, mu_ptr, g2_ptr, h_ptr, gt_ptr, HW, n, NP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pix = pid * BLOCK + tl.arange(0, BLOCK)
    x, m2 = _softmax_tile(t_ptr, pix, HW, n, NP)
    p = tl.arange(0, NP)
    mu = tl.load(mu_ptr + p)
    h = tl.load(h_ptr + p)
    g2 = tl.load(g2_ptr + p[:, None] * NP + p[None, :])
    xc = tl.where(m2, x - mu[:, None], 0.0)
    gx = tl.dot(g2, xc, allow_tf32=False) + xc * h[:, None]
    gt = x * (gx - tl.sum(x * gx, 0)[None, :])
    tl.store(gt_ptr + p[:, None] * HW + pix[None, :], gt, mask=m2)


class _PalletNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        n = tensor.shape[0]
        t = tensor.contiguous()
        HW = t.numel() // n
        NP = _p2(n)
        nb = triton.cdiv(HW, BLOCK)
        s = torch.empty(nb, NP, device=t.device, dtype=torch.float32)
        _pn_sum[(nb,)](t, s, HW, n, NP=NP, BLOCK=BLOCK)
        mu = s.sum(0) / HW                                               # (NP,), zero past n
        a = torch.empty(nb, NP, NP, device=t.device, dtype=torch.float32)
        _pn_cov[(nb,)](t, mu, a, HW, n, NP=NP, BLOCK=BLOCK)
        A = a.sum(0)[:n, :n]                                             # X_c^T X_c
        N = float(HW)
        sigma0 = (A.diagonal() / (N - 1)).sqrt()                         # torch.std (unbiased)
        sigma = sigma0 + 1e-8
        S = A / (sigma[:, None] * sigma[None, :] * N)
        off = S - torch.diag(S.diagonal())
        loss = off.mean() + (sigma * N).reciprocal().mean()
        ctx.save_for_backward(t, mu, sigma0, sigma, off)
        ctx.dims = (n, HW, NP)
        return loss

    @staticmethod
    def backward(ctx, go):
        t, mu, sigma0, sigma, off = ctx.saved_tensors
        n, HW, NP = ctx.dims
        N = float(HW)
        eye = torch.eye(n, device=t.device, dtype=torch.float32)
        # d loss / d A  (A = X_c^T X_c): mean over n^2 of the off-diagonal S
        Q = (1 - eye) / (n * n * N * sigma[:, None] * sigma[None, :])
        # d loss / d sigma: through both S terms, and the 1/(sigma N) term
        dsig = -2.0 / (n * n) * off.sum(1) / sigma - 1.0 / (n * N * sigma * sigma)
        g2 = torch.zeros(NP, NP, device=t.device, dtype=torch.float32)
        g2[:n, :n] = 2 * Q * go
        h = torch.zeros(NP, device=t.device, dtype=torch.float32)
        h[:n] = dsig * go / ((N - 1) * sigma0)                           # torch.std backward
        gt = torch.empty_like(t)
        nb = triton.cdiv(HW, BLOCK)
        _pn_bwd[(nb,)](t, mu, g2.contiguous(), h, gt, HW, n, NP=NP, BLOCK=BLOCK)
        return gt


def pallet_norm_loss(tensor):
    return _PalletNorm.apply(tensor)
