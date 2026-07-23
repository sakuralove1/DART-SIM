import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange
import numbers


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), proposed in Jie Liu et al. "Residual feature aggregation network for image super-resolution", CVPR2020

    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes are deleted.
    """

    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = 16
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """
    proposed by ByteESR Team in NTIRE 2022 Challenge on Efficient Super-Resolution
    """

    def __init__(self, n_feats, bias=True):
        super(RLFB, self).__init__()
        self.RB = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, (3, 3), (1, 1), (1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(n_feats, n_feats, (3, 3), (1, 1), (1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(n_feats, n_feats, (3, 3), (1, 1), (1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
        )
        self.C = nn.Conv2d(n_feats, n_feats, (1, 1), (1, 1), (0, 0), bias=bias)
        self.ESA = ESA(n_feats)

    def forward(self, x):
        res = self.RB(x)
        res = res + x
        res = self.C(res)
        res = self.ESA(res)
        return res


def to_3d(x):  # from_4d_to_3d
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):  # from_3d_to_4d
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b head c (h w)] \times [b head (h w) c] = [b head c c]
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # [b head c c] \times [b head c (h w)] = [b head c (h w)]

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class DirectionalScan2D(nn.Module):
    """Dependency-free directional scan used by the MambaIRv2-style block."""

    def __init__(self, dim, kernel_size=7, bias=True):
        super(DirectionalScan2D, self).__init__()
        padding = kernel_size // 2
        self.scan_h = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=bias)
        self.scan_v = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # Horizontal and vertical scans approximate non-causal spatial state propagation.
        x_h = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        x_h = self.scan_h(x_h).reshape(b, h, c, w).permute(0, 2, 1, 3)

        x_v = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        x_v = self.scan_v(x_v).reshape(b, w, c, h).permute(0, 2, 3, 1)

        return x_h + x_v


class MambaIRv2StyleBlock(nn.Module):
    """A lightweight PyTorch-only substitute for testing MambaIRv2-style modeling.

    The environment does not include mamba_ssm, so this block keeps the useful
    ingredients for our ablation: local convolution, bidirectional spatial
    scanning, channel gating, and a residual feed-forward refinement.
    """

    def __init__(self, dim=48, expansion=2.0, bias=True, LayerNorm_type=WithBias_LayerNorm):
        super(MambaIRv2StyleBlock, self).__init__()
        hidden_dim = int(dim * expansion)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.in_proj = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.local_dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=bias)
        self.scan = DirectionalScan2D(hidden_dim, kernel_size=7, bias=bias)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GFeedForward(dim, 2.0, bias)

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)
        content, gate = self.in_proj(x_norm).chunk(2, dim=1)
        local_feat = self.local_dwconv(content)
        scan_feat = self.scan(local_feat)
        gate = torch.sigmoid(gate)
        channel_gate = self.channel_gate(scan_feat)
        x = shortcut + self.out_proj(scan_feat * gate * channel_gate)
        x = x + self.ffn(self.norm2(x))
        return x


class DetailPreservedMambaBlock(nn.Module):
    """Detail-preserved Mamba-style block for post-pattern-prior refinement.

    The block only scans high-frequency residual features and adds the result
    back with small learnable residual weights. This keeps pattern-prior details sharp
    while still allowing long-range filament continuity to be modeled.
    """

    def __init__(self, dim=48, expansion=1.0, bias=True, LayerNorm_type=WithBias_LayerNorm,
                 scan_kernel_size=5, gamma_init=0.1):
        super(DetailPreservedMambaBlock, self).__init__()
        hidden_dim = int(dim * expansion)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.in_proj = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.local_dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                                      groups=hidden_dim, bias=bias)
        self.scan = DirectionalScan2D(hidden_dim, kernel_size=scan_kernel_size, bias=bias)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.gamma_scan = nn.Parameter(torch.ones(1) * gamma_init)
        self.gamma_ffn = nn.Parameter(torch.ones(1) * gamma_init)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GFeedForward(dim, 1.0, bias)

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)

        blur = F.avg_pool2d(x_norm, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        detail = x_norm - blur

        content, gate = self.in_proj(detail).chunk(2, dim=1)
        local_feat = self.local_dwconv(content)
        scan_feat = 0.5 * self.scan(local_feat)
        scan_feat = scan_feat * torch.sigmoid(gate) * self.channel_gate(scan_feat)
        scan_feat = self.out_proj(scan_feat)

        fuse_gate = self.fuse_gate(torch.cat([x_norm, detail, scan_feat], dim=1))
        x = shortcut + self.gamma_scan * fuse_gate * scan_feat
        x = x + self.gamma_ffn * self.ffn(self.norm2(x))
        return x


## Gated-Dconv Feed-Forward Network (GDFN)
class GFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


