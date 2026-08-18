"""DCT v5: 块级光照白化版 (fii-dct-v5-block-illumination)

完整 frbnet_utils.py 平替 —— 保持 FIINet(number_K, lamda) 接口契约:
  (B, 3, H, W) -> (B, 3, H, W)

相对 v4 (LearnableDCTEnergyFilter) 的单变量修改 (方案: plans/DCT/FII_DCT_v5_块级光照白化方案.md):
  1. 全局 DCT -> 8×8 分块 DCT (sqrt-hann 窗, 步长 4 = 50% 重叠, 对称填充 8,
     完美重建框架沿用 v3 已验证方案)
  2. 全局谱整形 -> 块级 DC 平面空变光照补偿 (★ H1): DC'_c = (DC_c - L̂_c)·γ(x)
     L̂ = 固定 5×5 均值平滑 DC 平面 (零参数); γ(x) = σ(MLP(块均值/方差/梯度)) (输入相关, 打破家族定律 L2 属刻意)
  3. AC 频带可学习量化收缩 (★ H3, JPEG 语义): S'_b = S_b·Q_b(g)
     Q_b = 6 带收缩表 × 暗块门控 (暗块收缩强) + 软阈值 τ_b
  4. 重叠 IDCT 重建 (fold)
  5. 输出 +x 残差: x_out = fuse_net(feat_agg) + x (打破家族定律 L3, 起步保底)
  6. 按评审 E8-2 删除 DST 边缘支 (V2/D7 实测不稳定、不叠加, 家族证据倾向删)

参数: γ MLP (3->8->1) 41 + Q_b 6 + 门控 w_b 6 + 暗度尺度 1 + τ_b 6 = 60 (不含基础 net)
初始化: γ 末层 bias=-3 (σ≈0.047, 白化缓慢介入); Q=σ(0)=0.5; w=σ(0)=0.5; τ=0.01; 暗度尺度=2.0
已知边界: 块间独立整形 (同 v3); 无全局频点掩膜 (v4 分段窗被块级补偿替代);
          8×8 块内对角增益仍受 D2 等变限制 (方向性交空间支)
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as td


# 消融 env 开关 (默认全关 = 全量 v5; 与 fii-dct-ablation 同模式)
NO_AC_SHRINK = os.environ.get('FRBNET_DCT_V5_NO_AC_SHRINK', '0') == '1'   # D5b
NO_BLOCK = os.environ.get('FRBNET_DCT_V5_NO_BLOCK', '0') == '1'           # D5c
NO_OVERLAP = os.environ.get('FRBNET_DCT_V5_NO_OVERLAP', '0') == '1'      # D5d
NO_RESIDUAL = os.environ.get('FRBNET_DCT_V5_NO_RESIDUAL', '0') == '1'    # D5e
STRONG_GAMMA = os.environ.get('FRBNET_DCT_V5_STRONG_GAMMA', '0') == '1'  # D15: γ 初值 -1 (白化强起步)


class BlockIlluminationCompensation(nn.Module):
    """8×8 分块 DCT 光照白化: DC 平面空变补偿 + AC 频带量化收缩 + 软阈值.

    forward: (x, diffs) -> (B,3,H,W)
      x:      (B,3,H,W) log 域原图 (块统计 / 暗度门控来源)
      diffs:  (B,3,H,W) log 域通道差 (rg/gb/rb, 被补偿重建)
    """

    def __init__(self, block_size=8, step=4, pad=8, n_bands=6):
        super().__init__()
        self.bs = block_size
        self.step = step
        self.pad = pad
        self.n_bands = n_bands

        # 空变增益 γ(x) = σ(MLP([块均值, 块方差, 块梯度])) — 输入相关 (刻意打破 L2)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 1),
        )
        with torch.no_grad():
            for m in self.gamma_mlp:
                if isinstance(m, nn.Linear):
                    m.weight.normal_(0.0, 0.01)
            self.gamma_mlp[2].bias.fill_(-1.0 if STRONG_GAMMA else -3.0)  # σ≈0.27 (D15) / σ≈0.047 白化缓慢介入

        # AC 频带收缩表 (JPEG 语义): Q_b(g) = q_b·(1 - dark·w_b), dark 块 -> Q 小 -> 收缩强
        self.raw_q = nn.Parameter(torch.zeros(n_bands))        # σ -> (0,1), 初值 0.5
        self.raw_w = nn.Parameter(torch.zeros(n_bands))        # 门控强度 σ -> (0,1), 初值 0.5
        self.raw_dark_scale = nn.Parameter(torch.tensor(2.0))  # 暗度标定
        self.log_tau = nn.Parameter(torch.full((n_bands,), math.log(0.01)))  # 软阈值

        # 频带索引: r = sqrt(u²+v²) 整数带, 截断到 n_bands-1 (DC 属 band 0, 但 DC 另行处理)
        u = torch.arange(block_size, dtype=torch.float32)[:, None]
        v = torch.arange(block_size, dtype=torch.float32)[None, :]
        r = torch.sqrt(u * u + v * v)
        band = torch.clamp(r.long(), 0, n_bands - 1)           # (8,8)
        self.register_buffer('band_idx', band)

        # sqrt-hann 分析/合成窗 (50% 重叠 + 对称填充 -> 精确重建, v3 已验证)
        if NO_OVERLAP:
            # D5d: 无缝分块 (hop=8, 无窗, 无填充)
            self.step = 8
            self.pad = 0
            self.register_buffer('w2', torch.ones(block_size, block_size))
        else:
            hann = torch.sqrt(torch.hann_window(block_size))
            self.register_buffer('w2', hann[:, None] * hann[None, :])

    def _block_stats(self, xb):
        """xb: (B,C,nb,8,8) log 域加窗块 -> (B,C,nb,3) [均值, 方差, 梯度]."""
        bmean = xb.mean(dim=(-1, -2))                           # (B,C,nb)
        bvar = xb.var(dim=(-1, -2), unbiased=False)
        grad = (xb[..., 1:, :] - xb[..., :-1, :]).abs().mean(dim=(-1, -2))
        grad = grad + (xb[..., :, 1:] - xb[..., :, :-1]).abs().mean(dim=(-1, -2))
        grad = grad / 2.0
        return torch.stack([bmean, bvar, grad], dim=-1)         # (B,C,nb,3)

    def forward(self, x, diffs):
        B, C, H, W = x.shape
        dtype = x.dtype

        # 全局退化 (D5c): DC 平面与统计取全局标量, 无空变
        if NO_BLOCK:
            xg = torch.log(x.clamp(min=1e-6)).mean(dim=(-1, -2))     # (B,C)
            stats = torch.stack([
                xg,
                torch.log(x.clamp(min=1e-6)).var(dim=(-1, -2), unbiased=False),
                torch.zeros_like(xg),
            ], dim=-1)                                                 # (B,C,3)
            gamma = torch.sigmoid(self.gamma_mlp(stats)).squeeze(-1)  # (B,C)
            dark = 1 - torch.sigmoid(self.raw_dark_scale * (xg.mean(dim=1, keepdim=True) - math.log(0.5)))

        xp = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        dp = F.pad(diffs, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        HP, WP = H + 2 * self.pad, W + 2 * self.pad
        nb_h = (HP - self.bs) // self.step + 1
        nb_w = (WP - self.bs) // self.step + 1
        nb = nb_h * nb_w

        def _unfold(t):
            b = t.unfold(2, self.bs, self.step).unfold(3, self.bs, self.step)
            b = b.permute(0, 2, 3, 1, 4, 5).reshape(B, nb_h, nb_w, C, self.bs, self.bs)
            return b.permute(0, 3, 1, 2, 4, 5)                  # (B,C,nb_h,nb_w,8,8)

        xb = _unfold(xp)                                        # 原图块
        db = _unfold(dp)                                        # 通道差块
        w = self.w2.to(dtype)

        # 分块 DCT (加分析窗)
        dct_b = td.dct_2d((db * w).reshape(-1, self.bs, self.bs), norm='ortho')
        dct_b = dct_b.view(B, C, nb_h, nb_w, self.bs, self.bs)

        # DC 平面: 每块 (0,0) 系数 -> 局部光照图; L̂ = 固定 5×5 平滑 (零参数)
        DC = dct_b[..., 0, 0]                                   # (B,C,nb_h,nb_w)
        if NO_BLOCK:
            Lhat = DC.mean(dim=(-1, -2), keepdim=True)          # 全局标量 (B,C,1,1)
            gamma = gamma.view(B, C, 1, 1)
            dark = dark.view(B, 1, 1, 1)
        else:
            Lhat = F.avg_pool2d(F.pad(DC, (2, 2, 2, 2), mode='reflect'), 5, 1, 0)
            # 空变增益 γ(x): 原图块统计 -> (B,C,nb_h,nb_w), σ -> (0,1)
            stats = self._block_stats(xb * w)                   # (B,C,nb_h,nb_w,3)
            gamma = torch.sigmoid(self.gamma_mlp(stats)).squeeze(-1)

        # 块级白化补偿: DC'_c = (DC_c - L̂_c)·γ(x)   (log 域减法 ≡ 空域除法)
        DC_new = (DC - Lhat) * gamma

        # AC 频带量化收缩 + 软阈值: S'_b = sign(S)·ReLU(|S|-τ_b)·Q_b(g)
        q = torch.sigmoid(self.raw_q)                           # (6,)
        wgt = torch.sigmoid(self.raw_w)                         # (6,)
        tau = torch.exp(self.log_tau)                           # (6,)
        if not NO_BLOCK:
            xb_mean = (xb * w).mean(dim=(-1, -2))               # (B,C,nb_h,nb_w)
            lum = xb_mean.mean(dim=1, keepdim=True)             # (B,1,nb_h,nb_w) 亮度
            dark = 1 - torch.sigmoid(self.raw_dark_scale * (lum - math.log(0.5)))

        band = self.band_idx                                    # (8,8)
        if NO_AC_SHRINK:
            # D5b: 只留 DC 补偿 (无频带收缩/软阈值)
            ac = dct_b.clone()
            ac[..., 0, 0] = DC_new
        else:
            Q = q[band][None, None, None, None] * (1 - dark.unsqueeze(-1).unsqueeze(-1) * wgt[band][None, None, None, None])  # (B,1,nb_h,nb_w,8,8)
            ac = dct_b.clone()
            ac[..., 0, 0] = 0.0                                 # 抠 DC
            tau_b = tau[band]
            ac = torch.relu(ac.abs() - tau_b) * ac.sign() * Q   # 阈值 + 收缩
            ac[..., 0, 0] = DC_new                              # 补偿后 DC 写回

        # 重叠 IDCT + 合成窗 + fold 重建
        filt = td.idct_2d(ac.reshape(-1, self.bs, self.bs), norm='ortho')
        filt = filt.view(B, C, nb_h, nb_w, self.bs, self.bs) * w
        folded = filt.permute(0, 1, 2, 3, 4, 5).reshape(B, C, nb, self.bs * self.bs)
        folded = folded.permute(0, 1, 3, 2)                     # (B,C,64,nb)
        acc = F.fold(folded.reshape(B, C * self.bs * self.bs, nb),
                     output_size=(HP, WP), kernel_size=self.bs, stride=self.step)
        wsum = F.fold((w * w).reshape(1, 1, 1, self.bs * self.bs)
                      .expand(B, C, 1, self.bs * self.bs)
                      .reshape(B, C * self.bs * self.bs, 1)
                      .expand(B, C * self.bs * self.bs, nb),
                      output_size=(HP, WP), kernel_size=self.bs, stride=self.step)
        out = acc / wsum.clamp(min=1e-12)
        return out[:, :, self.pad:self.pad + H, self.pad:self.pad + W]


class LearnableBlockIlluminationFilter(nn.Module):
    """v5: 块级光照白化 (8×8 分块 DCT + DC 平面补偿 + AC 频带收缩), 通道差 -> (B,3,H,W).

    number_K 保留仅为接口兼容 (v5 无全局径向基).
    """

    def __init__(self, number_K=20, lamda=0.1):
        super().__init__()
        self.illum = BlockIlluminationCompensation()

    def forward(self, img):
        x = torch.log(img.clamp(min=1e-6))
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        diff_rg = r - g
        diff_gb = g - b
        diff_rb = r - b
        diffs = torch.cat([diff_rg, diff_gb, diff_rb], dim=1)
        return self.illum(x, diffs)


class FIINet(nn.Module):
    """与 FFT 原版 FIINet 结构一致, fim 换 LearnableBlockIlluminationFilter, 输出 +x 残差."""

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
        self.fim = LearnableBlockIlluminationFilter(number_K, lamda)

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        if not NO_RESIDUAL:
            x_out = x_out + x
        return x_out
