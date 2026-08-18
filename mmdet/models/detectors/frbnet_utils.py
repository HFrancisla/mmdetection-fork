"""DCT v8: 极简骨架版 MIN-DCT (fii-dct-v8-min)

完整 frbnet_utils.py 平替 —— 保持 FIINet(number_K, lamda) 接口契约:
  (B, 3, H, W) -> (B, 3, H, W)

设计: plans/DCT/20260816_FII_DCT_V7-V10_全新设计方案.md §2 (组件汤结论 + G1 骨架假设):
  v5 消融 D9–D12 全部持平 (中性组件汤) -> 收益在"骨架组合"而非任何单组件;
  D10 块≡全局 (0.7488) -> 分块结构不负载 -> v8 砍掉全部分块.

结构 (零分块, 41 参数):
  1. log 域通道差 FCCR (M1): Δ_rg/Δ_gb/Δ_rb, 光照不变骨架
  2. 整图 DCT (dct_2d, norm='ortho')
  3. 全局挖 DC: D[0,0] = 0  ≡ 去全图均值 (M1 语义, v4 Wg[0,0]=0 同款)
  4. 全图标量增益 γ_c(x) = σ(MLP([全图均值, 方差, 梯度范数])) (输入相关, 打破 L2)
     MLP 三通道共享权重, 逐通道独立统计 -> (B,3) 标量, 无任何空间变化 (F7 红线)
  5. IDCT -> 3 通道 -> spectral_net -> fuse_net(concat(spatial_net(x))) + x (M4 残差)

参数: γ MLP (3->8->1) 41 (共享, 不含基础 net)
初始化: γ 末层 bias=-3 (σ≈0.047, 白化缓慢介入, D15 口径)
假设: H-M1 骨架本身承载 D14 增益 (v8 ≥ D14 − 2σ) / H-M2 v8 ≥ 2σ vs A5
      / H-M3 全局 γ 输入相关必要 (开关: 固定标量)
"""
import torch
import torch.nn as nn
import torch_dct as td

NO_RESIDUAL = __import__('os').environ.get('FRBNET_DCT_V8_NO_RESIDUAL', '0') == '1'
FIXED_GAMMA = __import__('os').environ.get('FRBNET_DCT_V8_FIXED_GAMMA', '0') == '1'  # H-M3


class MinimalGlobalWhiten(nn.Module):
    """v8: 整图 DCT + 全局 DC 挖除 + 全局统计 γ (零分块, 零空间变化)."""

    def __init__(self):
        super().__init__()
        # 输入相关全图标量增益 γ(x): MLP(均值/方差/梯度范数), 三通道共享权重
        self.gamma_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 1),
        )
        with torch.no_grad():
            for m in self.gamma_mlp:
                if isinstance(m, nn.Linear):
                    m.weight.normal_(0.0, 0.01)
            self.gamma_mlp[2].bias.fill_(-3.0)  # σ≈0.047 白化缓慢介入 (D15 口径)

    @staticmethod
    def _global_stats(x):
        """x: (B,C,H,W) log 域 -> (B,C,3) [全图均值, 方差, 梯度范数]."""
        bmean = x.mean(dim=(-1, -2))
        bvar = x.var(dim=(-1, -2), unbiased=False)
        grad = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(-1, -2))
        grad = grad + (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(-1, -2))
        grad = grad / 2.0
        return torch.stack([bmean, bvar, grad], dim=-1)  # (B,C,3)

    def forward(self, x, diffs):
        """x: (B,3,H,W) log 域原图 (统计来源); diffs: (B,3,H,W) log 域通道差."""
        B, C, H, W = diffs.shape
        D = td.dct_2d(diffs, norm='ortho')                    # 整图 DCT (B,3,H,W)
        if FIXED_GAMMA:
            gamma = torch.full((B, C, 1, 1), 0.5, device=D.device, dtype=D.dtype)
        else:
            stats = self._global_stats(x)                     # (B,C,3)
            gamma = torch.sigmoid(self.gamma_mlp(stats))      # (B,C,1) -> 全图标量
            gamma = gamma[:, :, None, None]
        D = D * gamma                                         # 输入相关全图重缩放
        D[..., 0, 0] = 0.0                                    # 挖 DC ≡ 去全图均值
        return td.idct_2d(D, norm='ortho')


class LearnableMinimalGlobalFilter(nn.Module):
    """v8: 极简骨架 (通道差 -> 整图 DCT -> 全局 DC 白化 -> IDCT).

    number_K 保留仅为接口兼容 (无全局径向基).
    """

    def __init__(self, number_K=20, lamda=0.1):
        super().__init__()
        self.whiten = MinimalGlobalWhiten()

    def forward(self, img):
        x = torch.log(img.clamp(min=1e-6))
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        diff_rg = r - g
        diff_gb = g - b
        diff_rb = r - b
        diffs = torch.cat([diff_rg, diff_gb, diff_rb], dim=1)
        return self.whiten(x, diffs)


class FIINet(nn.Module):
    """与 FFT 原版 FIINet 结构一致, fim 换 LearnableMinimalGlobalFilter, 输出 +x 残差."""

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
        self.fim = LearnableMinimalGlobalFilter(number_K, lamda)

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        if not NO_RESIDUAL:
            x_out = x_out + x
        return x_out
