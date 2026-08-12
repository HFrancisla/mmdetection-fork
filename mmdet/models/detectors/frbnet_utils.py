"""DCT v1: 基础实数增益版 (fii-dct-v1-basic-real)

完整 frbnet_utils.py 平替 —— 保持 FIINet(number_K, lamda) 接口契约:
  (B, 3, H, W) -> (B, 3, H, W)

相对 FFT 原版 (LearnableFreFilter) 的单变量修改:
  1. 变换域: rfft2/irfft2 -> dct_2d/idct_2d (torch-dct, norm='ortho')
  2. 掩膜网格: (H, W/2+1) 复数半谱 -> (H, W) 实数全谱 (DCT 频率轴 k/(2N))
  3. 相位支线 (diff_phase) 删除, 径向基数量由 config 的 number_K 控制
     (number_K=20 时总参数 42, 与 FFT 原版持平 = 参数预算转移)
  4. Wg 零直流高斯窗保留: Wg[0,0]=0 ≡ 去全局均值 (DCT(0,0)=sqrt(HW)*mean)

参数: RadialBasisFilterReal(K=20) 41 + log_sigma (1) = 42
初始化: coeff=0 -> 分支初始输出全 0 (保守起步)

已知边界 (见 plans/DCT/DCT_前置_数学基础与修正清单.md §4 D2 等变性定理):
  纯 DCT 对角增益掩膜对镜像方向 (45° vs 135°) 不可区分; 可区分竖 vs 横 (90°)
"""
import math

import torch
import torch.nn as nn
import torch_dct as td


class RadialBasisFilterReal(nn.Module):
    """K 个径向基的实数增益 (删掉 FFT 版的 phase 支线).

    与 FFT 版 RadialBasisFilter 唯一结构差异: 返回单个实数增益 (1,H,W),
    无 diff_phase; 频率坐标换 DCT 轴 fx=k/(2W), fy=k/(2H) (全非负).
    """

    def __init__(self, n_coeff, lamda):
        super().__init__()
        self.n_coeff = n_coeff
        self.n_ang_freq = 1
        self.coeff_mag = nn.Parameter(torch.zeros(n_coeff))     # 初值 0 -> 保守起步
        self.raw_gate_mag = nn.Parameter(torch.ones(n_coeff))
        self.lamda = lamda

        mu = torch.linspace(0.0, 1.0, steps=n_coeff)
        self.register_buffer('mu', mu)
        self.log_bwh = nn.Parameter(torch.tensor(0.0))

    def forward(self, H: int, W: int, device, dtype):
        fy = torch.arange(H, dtype=dtype, device=device) / (2 * H)   # DCT 频率轴 [0,0.5)
        fx = torch.arange(W, dtype=dtype, device=device) / (2 * W)
        r_hat = torch.sqrt(fx ** 2 + fy[:, None] ** 2)
        r_hat = r_hat / r_hat.max()  # normalize to 0..1

        bwh = torch.exp(self.log_bwh) + 1e-6
        basis = torch.exp(-((r_hat.unsqueeze(0) - self.mu[:, None, None]) ** 2) / (2 * bwh ** 2))

        gate_mag = torch.sigmoid(self.raw_gate_mag)[:, None, None]

        angular_mod = 0
        theta = torch.atan2(fy[:, None], fx[None, :] + 1e-8)     # 第一象限
        for n in range(1, self.n_ang_freq + 1):
            angular_mod = angular_mod + torch.cos(n * theta) + torch.sin(n * theta)
        angular_mod = angular_mod / (2 * self.n_ang_freq)
        angular_mod = 1 + 0.1 * angular_mod

        basis = basis * angular_mod.unsqueeze(0)

        diff_mag = (gate_mag * self.coeff_mag[:, None, None] * basis).sum(0, keepdim=True)
        return diff_mag  # (1,H,W) 实数


class LearnableDCTFilter(nn.Module):
    """DCT v1: 通道差频谱上的实数掩膜 (Φ·M(θ))·Wg, idct 回空域.

    forward: (B,3,H,W) -> (B,3,H,W)
      x = log(img.clamp(min=1e-6))
      dct_r/g/b = dct_2d(通道, norm='ortho')
      diff_* = 通道差 (DCT 线性性 -> DCT(log(R/G)), 光照不变保留)
      out = idct_2d((D·Wg) * diff)
    """

    def __init__(self, number_K=20, lamda=0.1):
        super().__init__()
        self.init_sigma_ratio = 0.2
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self.rad_filter = RadialBasisFilterReal(number_K, lamda)

        self._sigma_init = False

    def forward(self, img):
        B, C, H, W = img.shape
        dtype, device = img.dtype, img.device
        assert C == 3

        if not self._sigma_init:
            sigma_px = self.init_sigma_ratio * min(H, W)
            with torch.no_grad():
                self.log_sigma.copy_(torch.tensor(math.log(sigma_px), dtype=dtype, device=device))
            self._sigma_init = True

        D = self.rad_filter(H, W, device, dtype)  # (1,H,W) 实数增益

        x = torch.log(img.clamp(min=1e-6))
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dct_r, dct_g, dct_b = (td.dct_2d(ch, norm='ortho') for ch in (r, g, b))
        diff_rg = dct_r - dct_g
        diff_gb = dct_g - dct_b
        diff_rb = dct_r - dct_b

        fy = torch.arange(H, dtype=dtype, device=device) / (2 * H)
        fx = torch.arange(W, dtype=dtype, device=device) / (2 * W)
        r_grid = torch.sqrt(fx ** 2 + fy[:, None] ** 2)
        sigma = torch.exp(self.log_sigma)
        Wg = torch.exp(-(r_grid / sigma) ** 2)
        Wg = Wg.clone()
        Wg[0, 0] = 0.0  # 挖 DC (去全局均值)

        def _filt(diff_dct):
            diff_dct = diff_dct.squeeze(1)
            out = td.idct_2d((D * Wg) * diff_dct, norm='ortho')
            return out.unsqueeze(1)

        fccr_rg = _filt(diff_rg)
        fccr_gb = _filt(diff_gb)
        fccr_rb = _filt(diff_rb)
        fccr_feat = torch.cat([fccr_rg, fccr_gb, fccr_rb], dim=1)  # (B,3,H,W)

        return fccr_feat


class FIINet(nn.Module):
    """与 FFT 原版 FIINet 结构完全一致, 仅 fim 换成 LearnableDCTFilter."""

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
        self.fim = LearnableDCTFilter(number_K, lamda)

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
