def dst2_fft(x):
    """DST-II, 沿最后一维。x: (..., N) -> (..., N), 正交归一。

    Y_k = c_k · Σ_n x_n sin(π(2n+1)(k+1)/(2N)),  k = 0..N-1
    实现: W_m = e^{-iπm/(2N)} · FFT_{2N}([x, 0..0])[m],  Y_k = -Im(W_{k+1})
    """
    N = x.shape[-1]
    xp = torch.cat([x, torch.zeros_like(x)], dim=-1)          # (..., 2N)
    F = torch.fft.fft(xp)                                     # (..., 2N) complex
    m = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    W = torch.exp(-1j * torch.pi * m / (2 * N)) * F[..., 1:N + 1]
    S = -W.imag                                               # DST-II 未归一化
    scale = torch.full((N,), math.sqrt(2.0 / N), dtype=x.dtype, device=x.device)
    scale[-1] = math.sqrt(1.0 / N)                            # k=N-1 特殊缩放
    return S * scale


def dst2_2d(x):
    """二维 DST-II (可分离): 先沿 W 再沿 H。x: (..., H, W) -> (..., H, W)"""
    y = dst2_fft(x)                                           # 沿 W
    y = dst2_fft(y.transpose(-1, -2))                         # 沿 H
    return y.transpose(-1, -2)


"""DCT 消融版 (D5/D6/D7): 基于 DCT v4 (fii-dct-v4-energy-focus) 的能量集中结构,
用 env 开关拆解 v4 的两个机制并测试 v2×v4 相位组合:

  - D5 (v4b, 只对数 μ):    FRBNET_DCT_NO_SEG_WG=1   -> SegmentWg 换回 v1 单高斯窗
  - D6 (v4c, 只分段窗):     FRBNET_DCT_NO_LOG_MU=1   -> 径向基 μ 回均匀分布 (v1)
  - D7 (v2+v4 组合):        FRBNET_DCT_DST_BRANCH=1  -> +DST 支 (相位类自由度, 初值 0)
                           (DCT 支按 D5/D6 开关取 log μ + 窗; DST 支用均匀 μ + 同一窗)
  - 默认 (全关) = D4 (v4a 完整版)
config: D5/D6 用 yolov3_frbnet_exdark_dct_v4.py (K=20); D7 用 dct_v2 (K=10, 双支参数对等)

参数: D4=44 / D5=42 / D6=43 / D7=45 (K=10 双支 21+21+3) | 初始化 coeff=0 (保守起步)
"""
import math
import os

import torch
import torch.nn as nn
import torch_dct as td


def _dct_flag(name):
    return os.environ.get(name, "0") == "1"

"""DCT v4: 能量集中低频分带版 (fii-dct-v4-energy-focus)

完整 frbnet_utils.py 平替 —— 保持 FIINet(number_K, lamda) 接口契约:
  (B, 3, H, W) -> (B, 3, H, W)

相对 v1 (LearnableDCTFilter) 的单变量修改:
  1. 径向基 μ 对数分布 (alpha=2.5): 低频密集 (能量集中区), 高频稀疏
  2. Wg 单高斯 -> SegmentWg 分段窗: DC 邻域 (r<=r_c) 精细高斯衰减 + 中高频平坦小量 γ_hi
     (利用 DCT 频率分辨率 ×2 与无边界伪高频优势)
  3. 其余同 v1 (通道差 DCT -> 实数掩膜 -> idct)

参数: RadialBasisFilterReal(log-μ, K=20) 41 + SegmentWg 3 = 44
初始化: coeff=0 -> 分支初始输出全 0 (保守起步)

已知边界: 同 v1 (D2 等变, 无相位自由度)
"""
import math

import torch
import torch.nn as nn
import torch_dct as td


def log_mu(n_coeff, alpha=2.5):
    """对数频率分带的 μ: 低频密集, 高频稀疏, 映射到 [0,1]."""
    t = torch.linspace(0.0, 1.0, steps=n_coeff)
    return (torch.exp(alpha * t) - 1) / (math.exp(alpha) - 1)


class RadialBasisFilterReal(nn.Module):
    """K 个径向基的实数增益 (v4: μ 对数分布)."""

    def __init__(self, n_coeff, lamda, mu_mode='log'):
        super().__init__()
        self.n_coeff = n_coeff
        self.n_ang_freq = 1
        self.coeff_mag = nn.Parameter(torch.zeros(n_coeff))   # 初值 0 -> 保守起步
        self.raw_gate_mag = nn.Parameter(torch.ones(n_coeff))
        self.lamda = lamda

        if mu_mode == 'log':
            mu = log_mu(n_coeff)
        else:
            mu = torch.linspace(0.0, 1.0, steps=n_coeff)
        self.register_buffer('mu', mu)
        self.log_bwh = nn.Parameter(torch.tensor(0.0))

    def forward(self, H: int, W: int, device, dtype):
        fy = torch.arange(H, dtype=dtype, device=device) / (2 * H)
        fx = torch.arange(W, dtype=dtype, device=device) / (2 * W)
        r_hat = torch.sqrt(fx ** 2 + fy[:, None] ** 2)
        r_hat = r_hat / r_hat.max()  # normalize to 0..1

        bwh = torch.exp(self.log_bwh) + 1e-6
        basis = torch.exp(-((r_hat.unsqueeze(0) - self.mu[:, None, None]) ** 2) / (2 * bwh ** 2))

        gate_mag = torch.sigmoid(self.raw_gate_mag)[:, None, None]

        angular_mod = 0
        theta = torch.atan2(fy[:, None], fx[None, :] + 1e-8)
        for n in range(1, self.n_ang_freq + 1):
            angular_mod = angular_mod + torch.cos(n * theta) + torch.sin(n * theta)
        angular_mod = angular_mod / (2 * self.n_ang_freq)
        angular_mod = 1 + 0.1 * angular_mod

        basis = basis * angular_mod.unsqueeze(0)

        diff_mag = (gate_mag * self.coeff_mag[:, None, None] * basis).sum(0, keepdim=True)
        return diff_mag  # (1,H,W) 实数


class SegmentWg(nn.Module):
    """DC 邻域分段窗: 精细高斯 (r<=r_c) + 中高频平坦小量 (r>r_c), Wg[0,0]=0."""

    def __init__(self, init_sigma_ratio=0.2):
        super().__init__()
        self.init_sigma_ratio = init_sigma_ratio
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))          # DC 邻域带宽
        self.log_rc = nn.Parameter(torch.tensor(math.log(0.07)))   # 分段点 r_c
        self.log_gamma_hi = nn.Parameter(torch.tensor(math.log(1e-3)))  # 高频平坦量
        self._init = False

    def forward(self, H: int, W: int, device, dtype):
        if not self._init:
            sigma_px = self.init_sigma_ratio * min(H, W)
            with torch.no_grad():
                self.log_sigma1.copy_(torch.tensor(math.log(sigma_px), dtype=dtype, device=device))
            self._init = True

        fy = torch.arange(H, dtype=dtype, device=device) / (2 * H)
        fx = torch.arange(W, dtype=dtype, device=device) / (2 * W)
        r = torch.sqrt(fx ** 2 + fy[:, None] ** 2)
        s1 = torch.exp(self.log_sigma1)
        rc = torch.exp(self.log_rc)
        ghi = torch.exp(self.log_gamma_hi)
        w = torch.where(r <= rc,
                        torch.exp(-(r / s1) ** 2),
                        ghi * torch.exp(-(rc / s1) ** 2))
        w = w.clone()
        w[0, 0] = 0.0  # 挖 DC (去全局均值)
        return w


class GaussianWg(nn.Module):
    """v1 风格单高斯窗: Wg = exp(-(r/sigma)^2), Wg[0,0]=0 (挖 DC)."""

    def __init__(self, init_sigma_ratio=0.2):
        super().__init__()
        self.init_sigma_ratio = init_sigma_ratio
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self._init = False

    def forward(self, H: int, W: int, device, dtype):
        if not self._init:
            sigma_px = self.init_sigma_ratio * min(H, W)
            with torch.no_grad():
                self.log_sigma.copy_(torch.tensor(math.log(sigma_px), dtype=dtype, device=device))
            self._init = True
        fy = torch.arange(H, dtype=dtype, device=device) / (2 * H)
        fx = torch.arange(W, dtype=dtype, device=device) / (2 * W)
        r = torch.sqrt(fx ** 2 + fy[:, None] ** 2)
        sigma = torch.exp(self.log_sigma)
        w = torch.exp(-(r / sigma) ** 2)
        w = w.clone()
        w[0, 0] = 0.0
        return w


class LearnableDCTEnergyFilter(nn.Module):
    """DCT 消融版 FIM: v4 能量集中 + (可选) DST 相位支.

    forward: (B,3,H,W) -> (B,3,H,W)
      out = idct(Mr·Wg ⊙ DCT(Δ)) [+ idct(Md·Wg ⊙ DST(Δ))],  Δ = log 域通道差
    """

    def __init__(self, number_K=20, lamda=0.1):
        super().__init__()
        self._dst = _dct_flag("FRBNET_DCT_DST_BRANCH")
        mu_mode = 'uniform' if _dct_flag("FRBNET_DCT_NO_LOG_MU") else 'log'
        wg = GaussianWg() if _dct_flag("FRBNET_DCT_NO_SEG_WG") else SegmentWg()

        self.rad_mr = RadialBasisFilterReal(number_K, lamda, mu_mode=mu_mode)  # DCT 支
        self.wg = wg
        if self._dst:
            self.rad_md = RadialBasisFilterReal(number_K, lamda, mu_mode='uniform')  # DST 支 (v2 语义)
            with torch.no_grad():
                self.rad_md.coeff_mag.zero_()                  # Md 初值 0 -> 优雅降级为 DCT 单支

    def forward(self, img):
        B, C, H, W = img.shape
        dtype, device = img.dtype, img.device
        assert C == 3

        Mr = self.rad_mr(H, W, device, dtype)                  # (1,H,W)
        Wg = self.wg(H, W, device, dtype)                      # (H,W)

        x = torch.log(img.clamp(min=1e-6))
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dct_r, dct_g, dct_b = (td.dct_2d(ch, norm='ortho') for ch in (r, g, b))
        diff_dct_rg, diff_dct_gb, diff_dct_rb = dct_r - dct_g, dct_g - dct_b, dct_r - dct_b

        if self._dst:
            dst_r, dst_g, dst_b = dst2_2d(r), dst2_2d(g), dst2_2d(b)
            diff_dst_rg, diff_dst_gb, diff_dst_rb = dst_r - dst_g, dst_g - dst_b, dst_r - dst_b
            Md = self.rad_md(H, W, device, dtype)

            def _filt(diff_dct, diff_dst):
                out = (td.idct_2d((Mr * Wg) * diff_dct.squeeze(1), norm='ortho')
                       + td.idct_2d((Md * Wg) * diff_dst.squeeze(1), norm='ortho'))
                return out.unsqueeze(1)
            fccr_rg = _filt(diff_dct_rg, diff_dst_rg)
            fccr_gb = _filt(diff_dct_gb, diff_dst_gb)
            fccr_rb = _filt(diff_dct_rb, diff_dst_rb)
        else:
            def _filt(diff_dct):
                out = td.idct_2d((Mr * Wg) * diff_dct.squeeze(1), norm='ortho')
                return out.unsqueeze(1)
            fccr_rg = _filt(diff_dct_rg)
            fccr_gb = _filt(diff_dct_gb)
            fccr_rb = _filt(diff_dct_rb)
        fccr_feat = torch.cat([fccr_rg, fccr_gb, fccr_rb], dim=1)  # (B,3,H,W)
        return fccr_feat


class FIINet(nn.Module):
    """与 FFT 原版 FIINet 结构完全一致, 仅 fim 换成 LearnableDCTEnergyFilter."""

    def __init__(self, number_K, lamda):
        super(FIINet, self).__init__()

        self.spatial_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                           nn.BatchNorm2d(24),
                                           nn.LeakyReLU(),
                                           ])
        self.spectral_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                            nn.BatchNorm2d(24),
                                            nn.LeakyReLU(),
                                            ])
        self.fuse_net = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(32, 3, 3, 1, 1, groups=1)
                                        ])
        self.fim = LearnableDCTEnergyFilter(number_K, lamda)

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
