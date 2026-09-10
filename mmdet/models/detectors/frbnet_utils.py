import os
import torch.nn.functional as F
import torch.nn as nn
import torch


class DWT_2D(nn.Module):
    """Haar 2-D DWT via pixel_unshuffle. (B,C,H,W) -> (B,4C,H/2,W/2) [LL,LH,HL,HH]."""

    def forward(self, x):
        xu = F.pixel_unshuffle(x, 2)
        b, c4, h2, w2 = xu.shape
        c = c4 // 4
        xu = xu.view(b, c, 4, h2, w2)
        x_ee, x_eo, x_oe, x_oo = xu[:, :, 0], xu[:, :, 1], xu[:, :, 2], xu[:, :, 3]
        ll = (x_ee + x_eo + x_oe + x_oo) * 0.5
        lh = (x_ee - x_eo + x_oe - x_oo) * 0.5
        hl = (x_ee + x_eo - x_oe - x_oo) * 0.5
        hh = (x_ee - x_eo - x_oe + x_oo) * 0.5
        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT_2D(nn.Module):
    """Haar 2-D IDWT via pixel_shuffle. (B,4C,H/2,W/2) [LL,LH,HL,HH] -> (B,C,H,W)."""

    def forward(self, x):
        b, c4, h2, w2 = x.shape
        c = c4 // 4
        x = x.view(b, 4, c, h2, w2)
        ll, lh, hl, hh = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        x_ee = (ll + lh + hl + hh) * 0.5
        x_eo = (ll - lh + hl - hh) * 0.5
        x_oe = (ll + lh - hl - hh) * 0.5
        x_oo = (ll - lh - hl + hh) * 0.5
        x = torch.stack([x_ee, x_eo, x_oe, x_oo], dim=2)
        x = x.view(b, c * 4, h2, w2)
        return F.pixel_shuffle(x, 2)


class LearnableDWT3Filter(nn.Module):
    """V9: 3-level DWT channel-difference filter with three explicit pathways.

    Pathway mapping (10 subbands, disjoint):
      低频通路  LL3                 -> 去均值(挖DC/色偏) + a_LL*d + b_LL   (Wg[0,0]=0 + 白平衡)
      中频通路  LH3 HL3 HH3 LH2 HL2 -> gamma_mid * d                       (结构增强, ~Phi)
      高频通路  HH2 LH1 HL1 HH1     -> gamma_hi * d                        (细节/噪声)
      HH 收缩   HH3 HH2 HH1         -> 额外乘 eta(<1)                     (Wg 压高频噪声)

    Contract: (B,3,H,W) -> (B,3,H,W). Total params = 36.
    """

    def __init__(self, wave="haar"):
        super().__init__()
        assert wave == "haar"
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        # 低频通路
        self.a_LL = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))     # (3,)
        self.b_LL = nn.Parameter(torch.zeros(3))
        # 中频通路: LH3,HL3,HH3,LH2,HL2
        self.gamma_mid = nn.Parameter(torch.full((5, 3), 0.1))      # (5,3)
        # 高频通路: HH2,LH1,HL1,HH1
        self.gamma_hi = nn.Parameter(torch.full((4, 3), 1e-3))      # (4,3)
        # HH 收缩: HH1,HH2,HH3
        self.eta = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))      # (3,)

    @staticmethod
    def _diff(subband):
        r, g, bl = subband[:, 0:1], subband[:, 1:2], subband[:, 2:3]
        return torch.cat([r - g, g - bl, r - bl], dim=1)

    def forward(self, img):
        b, c, h, w = img.shape
        assert c == 3
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        x = torch.log(img.clamp(min=1e-6))
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        d1 = self.dwt(x); ll1, lh1, hl1, hh1 = d1.chunk(4, dim=1)
        d2 = self.dwt(ll1); ll2, lh2, hl2, hh2 = d2.chunk(4, dim=1)
        d3 = self.dwt(ll2); ll3, lh3, hl3, hh3 = d3.chunk(4, dim=1)

        # ---- 低频通路: LL3 ----
        ll3_c = ll3 - ll3.mean(dim=(2, 3), keepdim=True)
        d_ll = self._diff(ll3_c)
        ll3_o = d_ll * self.a_LL.view(1, 3, 1, 1) + self.b_LL.view(1, 3, 1, 1)

        # ---- 中频通路 ----
        lh3_o = self._diff(lh3) * self.gamma_mid[0].view(1, 3, 1, 1)
        hl3_o = self._diff(hl3) * self.gamma_mid[1].view(1, 3, 1, 1)
        hh3_o = self._diff(hh3) * self.gamma_mid[2].view(1, 3, 1, 1) * self.eta[2].view(1, 1, 1, 1)
        lh2_o = self._diff(lh2) * self.gamma_mid[3].view(1, 3, 1, 1)
        hl2_o = self._diff(hl2) * self.gamma_mid[4].view(1, 3, 1, 1)

        # ---- 高频通路 ----
        hh2_o = self._diff(hh2) * self.gamma_hi[0].view(1, 3, 1, 1) * self.eta[1].view(1, 1, 1, 1)
        lh1_o = self._diff(lh1) * self.gamma_hi[1].view(1, 3, 1, 1)
        hl1_o = self._diff(hl1) * self.gamma_hi[2].view(1, 3, 1, 1)
        hh1_o = self._diff(hh1) * self.gamma_hi[3].view(1, 3, 1, 1) * self.eta[0].view(1, 1, 1, 1)

        # ---- 逆变换 ----
        ll2_r = self.idwt(torch.cat([ll3_o, lh3_o, hl3_o, hh3_o], dim=1))
        ll1_r = self.idwt(torch.cat([ll2_r, lh2_o, hl2_o, hh2_o], dim=1))
        out = self.idwt(torch.cat([ll1_r, lh1_o, hl1_o, hh1_o], dim=1))

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class LearnableDWT3FilterOnlyLL(nn.Module):
    """V9-Depth: 深度对照版 OnlyLL —— 按层级截断分解深度（真·一级/二级/三级 DWT）。

    与 fii-dwt-v9-only-ll 的关系：本分支唯一改动 = DWT 分解深度跟随 FRBNET_V9_LL_LEVELS
    截断（例如 LEVELS='1' 时只做一级分解 + LL1 白平衡 + 一级重建，不再多余分解到三级）。
    由 Haar 完美重建保证：新路径与"三级分解但只处理对应层"数值等价（浮点误差级），
    —— 对照实验唯一变量是分解深度（修正场尺度 2×2 / 4×4 / 8×8）。

    语义（由 DWT/IDWT 线性性）：out = x + Σ_k (ΔLL_k 经逐级 IDWT 摊回整图的低频修正场)，
    因此 x 的全部细节内容被原样保留（可数值验证：DWT(out) 细节带 == DWT(x) 细节带）；
    模块输出只叠加平滑的低频/颜色修正。与 V9 三通路对比：无中/高频增益、无 HH 收缩。

    LL 处理（白平衡仿射，作用于逐级重建出的近似 LL 带）：
        ll_k 去均值(挖 DC/色偏) -> 通道差 _diff -> a_k·d + b_k     （每启用级 6 参）
    细节带全程 raw：每级 IDWT 的 LH/HL/HH 槽直接取原始子带系数。

    环境开关（train/test 必须同 env，见设计文档）：
        FRBNET_V9_LL_LEVELS          '3'(默认) | '12' | '123' | '1' | '2'   处理哪些层级
        FRBNET_V9_ONLYLL_ZERO_DETAIL '1' -> 已分解的细节槽置 0（对照臂，输出≈纯低通）

    参数：6 × #启用层级（默认仅 LL3 = 6 个；全三级 = 18 个）。初值 a=0.3（每通道差增益），b=0。
    不变量（训练前数值验证全过）：常数图->输出 0（b=0 初值）/ 乘性光照不变 / 去均值去色偏 /
    细节带原样保留；a=b=0 时不是恒等映射（LL 槽有界修正），这是设计使然。
    """

    def __init__(self):
        super().__init__()
        levels = os.environ.get('FRBNET_V9_LL_LEVELS', '3').strip()
        self.enable1 = '1' in levels
        self.enable2 = '2' in levels
        self.enable3 = '3' in levels
        if not (self.enable1 or self.enable2 or self.enable3):
            raise ValueError(
                'FRBNET_V9_LL_LEVELS must enable at least one of {1,2,3}, got: %r' % levels)
        self.zero_detail = os.environ.get('FRBNET_V9_ONLYLL_ZERO_DETAIL', '0') == '1'
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        if self.enable3:
            self.a3 = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))
            self.b3 = nn.Parameter(torch.zeros(3))
        if self.enable2:
            self.a2 = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))
            self.b2 = nn.Parameter(torch.zeros(3))
        if self.enable1:
            self.a1 = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))
            self.b1 = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _wb(sub, a, b):
        """白平衡仿射: 去均值(挖 DC/色偏) -> 通道差 -> a·d + b。返回 (B,3,H',W')。"""
        sub_c = sub - sub.mean(dim=(2, 3), keepdim=True)
        r, g, bl = sub_c[:, 0:1], sub_c[:, 1:2], sub_c[:, 2:3]
        d = torch.cat([r - g, g - bl, r - bl], dim=1)
        return d * a.view(1, 3, 1, 1) + b.view(1, 3, 1, 1)

    def forward(self, img):
        b, c, h, w = img.shape
        assert c == 3
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        x = torch.log(img.clamp(min=1e-6))
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # ---- 分解：深度 = 最高启用的层级（'1'→1级 / '2'·'12'→2级 / '3'·'123'→3级）----
        d1 = self.dwt(x); ll1, lh1, hl1, hh1 = d1.chunk(4, dim=1)
        if self.enable2 or self.enable3:
            d2 = self.dwt(ll1); ll2, lh2, hl2, hh2 = d2.chunk(4, dim=1)
        else:
            ll2 = lh2 = hl2 = hh2 = None
        if self.enable3:
            d3 = self.dwt(ll2); ll3, lh3, hl3, hh3 = d3.chunk(4, dim=1)
        else:
            ll3 = lh3 = hl3 = hh3 = None

        # ---- 对照臂: 已分解的细节槽置 0（等价 V9 设计文档 §5.4 NO_MID+NO_HI 置零口径）----
        if self.zero_detail:
            lh1 = torch.zeros_like(lh1); hl1 = torch.zeros_like(hl1); hh1 = torch.zeros_like(hh1)
            if self.enable2 or self.enable3:
                lh2 = torch.zeros_like(lh2); hl2 = torch.zeros_like(hl2); hh2 = torch.zeros_like(hh2)
            if self.enable3:
                lh3 = torch.zeros_like(lh3); hl3 = torch.zeros_like(hl3); hh3 = torch.zeros_like(hh3)

        # ---- LL 处理 + 逐级重建（细节带全程 raw 或置 0）----
        if self.enable3:
            ll3_o = self._wb(ll3, self.a3, self.b3)
            ll2_r = self.idwt(torch.cat([ll3_o, lh3, hl3, hh3], dim=1))
            ll2_o = self._wb(ll2_r, self.a2, self.b2) if self.enable2 else ll2_r
            ll1_r = self.idwt(torch.cat([ll2_o, lh2, hl2, hh2], dim=1))
        elif self.enable2:
            ll2_o = self._wb(ll2, self.a2, self.b2)
            ll1_r = self.idwt(torch.cat([ll2_o, lh2, hl2, hh2], dim=1))
        else:  # enable1 必真（__init__ 已校验至少启用一层）
            ll1_r = ll1

        ll1_o = self._wb(ll1_r, self.a1, self.b1) if self.enable1 else ll1_r
        out = self.idwt(torch.cat([ll1_o, lh1, hl1, hh1], dim=1))
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class ChannelDiffDC(nn.Module):
    """Three-channel DC branch used by the cascaded DWT-DC variants.

    The operation intentionally matches ``LearnableDWT3FilterOnlyLL._wb``:

        x -> x - spatial_mean(x) -> [R-G, G-B, R-B]
          -> weight * difference + bias

    ``weight`` and ``bias`` are per output difference channel.  A separate
    instance is created for every DWT level; the branches must not share
    parameters because they operate at different spatial scales.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((3,), 0.3))
        self.bias = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _channel_difference(subband):
        r, g, bl = subband[:, 0:1], subband[:, 1:2], subband[:, 2:3]
        return torch.cat([r - g, g - bl, r - bl], dim=1)

    def forward(self, subband):
        if subband.ndim != 4 or subband.shape[1] != 3:
            raise ValueError(
                'ChannelDiffDC expects a 4-D tensor with 3 channels, got '
                f'{tuple(subband.shape)}')
        centered = subband - subband.mean(dim=(2, 3), keepdim=True)
        diff = self._channel_difference(centered)
        return (diff * self.weight.view(1, 3, 1, 1) +
                self.bias.view(1, 3, 1, 1))


class LearnableDWTDCCascade(nn.Module):
    """Cascaded multi-level DWT-DC with raw detail branches.

    This is deliberately different from ``LearnableDWT3FilterOnlyLL``.  In
    the latter, a deeper DWT is computed from the *raw* previous LL and the
    DC/white-balance operation is applied only on the selected LL during
    reconstruction.  Here the data path is instead::

        LL1 --DC1--> 3-channel tensor --DWT--> LL2
             --DC2--> 3-channel tensor --DWT--> LL3 --DC3--> ...

    Only the requested number of levels is instantiated.  Each level has an
    independent ``ChannelDiffDC`` branch.  The LH/HL/HH coefficients from
    every decomposition are passed to the matching IDWT unchanged (``raw``).
    Therefore the output of a two-level instance is reconstructed as::

        IDWT1(IDWT2([DC2(LL2), raw_detail2]), raw_detail1)

    and the three-level instance adds the analogous third-level stage.

    Contract: ``(B, 3, H, W) -> (B, 3, H, W)``.  The input is interpreted in
    the same log domain as the existing V9-Depth OnlyLL implementation.
    There are 6 learnable parameters per enabled level.
    """

    _MAX_LEVELS = 3
    # Keep the same 8-divisor padding as the existing (1/2/3, raw) protocol.
    # This makes the new 2/3-level arms comparable to the already-run arms and
    # guarantees that every possible subsequent DWT receives even dimensions.
    _PAD_DIVISOR = 8

    def __init__(self, levels=3, wave='haar'):
        super().__init__()
        if wave != 'haar':
            raise ValueError(f'LearnableDWTDCCascade expects haar, got {wave}')
        if not isinstance(levels, int) or isinstance(levels, bool):
            raise TypeError(f'levels must be an integer in [1, 3], got {levels!r}')
        if not 1 <= levels <= self._MAX_LEVELS:
            raise ValueError(
                f'levels must be in [1, {self._MAX_LEVELS}], got {levels}')

        self.levels = levels
        self.zero_detail = os.environ.get(
            'FRBNET_V9_ONLYLL_ZERO_DETAIL', '0').strip() == '1'
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()

        # Separate modules are important: DC1/DC2/DC3 correspond to
        # different resolutions and must not share weight or bias.
        self.dc1 = ChannelDiffDC()
        if levels >= 2:
            self.dc2 = ChannelDiffDC()
        if levels >= 3:
            self.dc3 = ChannelDiffDC()

    @staticmethod
    def _zero_details(details):
        return tuple(torch.zeros_like(detail) for detail in details)

    @staticmethod
    def _parse_level(value):
        """Parse ``1``, ``2``, ``3``, ``12`` or ``123`` to max depth."""
        text = str(value).strip()
        enabled = [int(ch) for ch in text if ch in '123']
        if not enabled:
            raise ValueError(
                'DWT-DC cascade levels must contain at least one of 1/2/3, '
                f'got {value!r}')
        return max(enabled)

    def _prepare_input(self, img):
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(
                'LearnableDWTDCCascade expects (B, 3, H, W), got '
                f'{tuple(img.shape)}')
        _, _, h, w = img.shape
        pad_h = (-h) % self._PAD_DIVISOR
        pad_w = (-w) % self._PAD_DIVISOR
        x = torch.log(img.clamp(min=1e-6))
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w, pad_h, pad_w

    def forward(self, img):
        x, h, w, pad_h, pad_w = self._prepare_input(img)

        # Level 1: raw details are retained, but its LL is transformed before
        # it is allowed to enter the next decomposition.
        ll1, lh1, hl1, hh1 = self.dwt(x).chunk(4, dim=1)
        ll1_dc = self.dc1(ll1)
        details1 = (lh1, hl1, hh1)

        if self.levels == 1:
            if self.zero_detail:
                details1 = self._zero_details(details1)
            out = self.idwt(torch.cat([ll1_dc, *details1], dim=1))
        else:
            # Level 2 is decomposed from DC1's 3-channel output, not from the
            # raw LL1.  Its details remain raw relative to this branch.
            ll2, lh2, hl2, hh2 = self.dwt(ll1_dc).chunk(4, dim=1)
            ll2_dc = self.dc2(ll2)
            details2 = (lh2, hl2, hh2)

            if self.levels == 2:
                if self.zero_detail:
                    details1 = self._zero_details(details1)
                    details2 = self._zero_details(details2)
                ll1_r = self.idwt(torch.cat([ll2_dc, *details2], dim=1))
                out = self.idwt(torch.cat([ll1_r, *details1], dim=1))
            else:
                # Level 3 repeats the same rule: DWT3 sees DC2's 3-channel
                # output.  This is a separate DC3 branch, not shared DC2.
                ll3, lh3, hl3, hh3 = self.dwt(ll2_dc).chunk(4, dim=1)
                ll3_dc = self.dc3(ll3)
                details3 = (lh3, hl3, hh3)

                if self.zero_detail:
                    details1 = self._zero_details(details1)
                    details2 = self._zero_details(details2)
                    details3 = self._zero_details(details3)
                ll2_r = self.idwt(torch.cat([ll3_dc, *details3], dim=1))
                ll1_r = self.idwt(torch.cat([ll2_r, *details2], dim=1))
                out = self.idwt(torch.cat([ll1_r, *details1], dim=1))

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class LearnableDWT1FilterDC(LearnableDWTDCCascade):
    """Explicit one-level name for the cascaded DWT-DC family."""

    def __init__(self, wave='haar'):
        super().__init__(levels=1, wave=wave)


class LearnableDWT2FilterDC(LearnableDWTDCCascade):
    """Two-level DWT-DC: DC1 -> DWT2 -> DC2, all details raw."""

    def __init__(self, wave='haar'):
        super().__init__(levels=2, wave=wave)


class LearnableDWT3FilterDC(LearnableDWTDCCascade):
    """Three-level DWT-DC: DC1 -> DWT2 -> DC2 -> DWT3 -> DC3."""

    def __init__(self, wave='haar'):
        super().__init__(levels=3, wave=wave)


class RadialBasisFilter(nn.Module):
    """FFT radial-basis filter (kept for F0 reference)."""

    def __init__(self, n_coeff, lamda):
        super().__init__()
        self.n_coeff = n_coeff
        self.n_ang_freq = 1
        self.coeff_mag   = nn.Parameter(torch.zeros(n_coeff))
        self.coeff_phase = nn.Parameter(torch.zeros(n_coeff))
        self.lamda = lamda
        self.raw_gate_mag = nn.Parameter(torch.ones(n_coeff))
        self.raw_gate_phase = nn.Parameter(torch.ones(n_coeff))
        mu = torch.linspace(0.0, 1.0, steps=n_coeff)
        self.register_buffer('mu', mu)
        self.log_bwh = nn.Parameter(torch.tensor(0.0))

    def forward(self, H: int, W: int, device, dtype):
        fy = torch.fft.fftfreq(H, dtype=dtype, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, dtype=dtype, device=device)[None, :]
        r_hat = torch.sqrt(fx ** 2 + fy ** 2)
        r_hat = r_hat / r_hat.max()
        bwh = torch.exp(self.log_bwh) + 1e-6
        basis = torch.exp(-((r_hat.unsqueeze(0) - self.mu[:, None, None]) ** 2) / (2 * bwh ** 2))
        gate_mag = torch.sigmoid(self.raw_gate_mag)[:, None, None]
        gate_phase = torch.sigmoid(self.raw_gate_phase)[:, None, None]
        angular_mod = 0
        theta = torch.atan2(fy, fx + 1e-8)
        for n in range(1, self.n_ang_freq + 1):
            angular_mod += torch.cos(n * theta) + torch.sin(n * theta)
        angular_mod = angular_mod / (2 * self.n_ang_freq)
        angular_mod = 1 + 0.1 * angular_mod
        basis = basis * angular_mod.unsqueeze(0)
        diff_mag = (gate_mag * self.coeff_mag[:, None, None] * basis).sum(0, keepdim=True)
        diff_phase = (gate_phase * self.coeff_phase[:, None, None] * basis).sum(0, keepdim=True)
        return diff_mag, diff_phase


class LearnableFreFilter(nn.Module):
    """Original FFT FIM (kept for F0 reference)."""

    def __init__(self, number_K=10, lamda=0.1):
        super().__init__()
        self.init_sigma_ratio = 0.2
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        self.rad_filter = RadialBasisFilter(number_K, lamda)
        self._sigma_init = False

    def forward(self, img):
        B, C, H, W = img.shape
        dtype, device = img.dtype, img.device
        assert C == 3
        if not self._sigma_init:
            sigma_px = self.init_sigma_ratio * min(H, W)
            with torch.no_grad():
                self.log_sigma.copy_(torch.tensor(np.log(sigma_px), dtype=dtype, device=device))
            self._sigma_init = True
        diff_mag, diff_phase = self.rad_filter(H, W, device, dtype)
        D = diff_mag.to(dtype) * torch.exp(1j * diff_phase.to(dtype))
        x = torch.log(img.clamp(min=1e-6))
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        fft_r, fft_g, fft_b = (torch.fft.rfft2(ch, norm='ortho') for ch in (r, g, b))
        diff_rg = fft_r - fft_g
        diff_gb = fft_g - fft_b
        diff_rb = fft_r - fft_b
        fy = torch.fft.fftfreq(H, dtype=dtype, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, dtype=dtype, device=device)[None, :]
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)
        sigma = torch.exp(self.log_sigma)
        Wg = torch.exp(-(r_grid / sigma) ** 2)
        Wg = Wg.clone(); Wg[0, 0] = 0.0

        def _filt(diff_fft):
            diff_fft = diff_fft.squeeze(1)
            out = torch.fft.irfft2((D * Wg) * diff_fft, s=(H, W), norm='ortho')
            return out.unsqueeze(1)

        fccr_rg = _filt(diff_rg)
        fccr_gb = _filt(diff_gb)
        fccr_rb = _filt(diff_rb)
        return torch.cat([fccr_rg, fccr_gb, fccr_rb], dim=1)


class LearnableDWTFilter(nn.Module):
    """V1 light DWT filter (kept for reference)."""

    def __init__(self, wave="haar"):
        super().__init__()
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.band_gains = nn.Parameter(torch.zeros(3, 4, 1, 1))
        self.band_bias = nn.Parameter(torch.zeros(3, 4, 1, 1))

    def forward(self, img):
        b, c, h, w = img.shape
        assert c == 3
        pad_h = h % 2
        pad_w = w % 2
        x = torch.log(img.clamp(min=1e-6))
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        bands = self.dwt(x).chunk(4, dim=1)
        diff_coeffs = []
        for band_idx, band in enumerate(bands):
            r, g, bl = band[:, 0:1], band[:, 1:2], band[:, 2:3]
            band_diffs = torch.cat([r - g, g - bl, r - bl], dim=1)
            gain = self.band_gains[:, band_idx].view(1, 3, 1, 1)
            bias = self.band_bias[:, band_idx].view(1, 3, 1, 1)
            diff_coeffs.append(band_diffs * gain + bias)
        out = self.idwt(torch.cat(diff_coeffs, dim=1))
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class FIINet(nn.Module):
    """FIINet with the V9-Depth DWT front-end.

    The historical ``dwt_onlyll`` path remains the default.  Set
    ``FRBNET_FIM=dwt_dc`` (or ``dwt_dc_cascade``) to use the new cascaded
    DWT-DC path, and select its depth with ``FRBNET_DWT_DC_LEVELS=1``, ``2``
    or ``3``.  If that variable is omitted, the highest level in the existing
    ``FRBNET_V9_LL_LEVELS`` variable is used, so old experiment runners can
    be reused safely.
    """

    def __init__(self, number_K, lamda):
        super(FIINet, self).__init__()
        self.spatial_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                           nn.BatchNorm2d(24),
                                           nn.LeakyReLU()])
        self.spectral_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1),
                                            nn.BatchNorm2d(24),
                                            nn.LeakyReLU()])
        self.fuse_net = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(32, 3, 3, 1, 1, groups=1)])
        fim_name = os.environ.get('FRBNET_FIM', 'dwt_onlyll').strip().lower()
        cascade_flag = os.environ.get(
            'FRBNET_DWT_DC_CASCADE',
            os.environ.get('FRBNET_V9_DWT_DC_CASCADE', '0'))
        use_cascade = fim_name in {
            'dwt_dc', 'dwt-dc', 'dwt_dc_cascade', 'dwt-cascade',
            'dwt_cascade'
        } or cascade_flag.strip() == '1'
        if use_cascade:
            level_value = os.environ.get('FRBNET_DWT_DC_LEVELS')
            if level_value is None:
                level_value = os.environ.get('FRBNET_V9_DWT_DC_LEVELS')
            if level_value is None:
                level_value = os.environ.get('FRBNET_V9_LL_LEVELS', '3')
            levels = LearnableDWTDCCascade._parse_level(level_value)
            cascade_cls = {
                1: LearnableDWT1FilterDC,
                2: LearnableDWT2FilterDC,
                3: LearnableDWT3FilterDC,
            }[levels]
            self.fim = cascade_cls()
        else:
            self.fim = LearnableDWT3FilterOnlyLL()

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out
