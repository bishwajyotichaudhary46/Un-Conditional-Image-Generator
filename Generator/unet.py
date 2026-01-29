# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange



# def norm_layer(channels: int, groups: int = 32):
#     return nn.GroupNorm(groups, channels)


# class SiLU(nn.Module):
#     def forward(self, x):
#         return F.silu(x)


# class FlashAttention(nn.Module):
#     """Simple self-attention with flash support (PyTorch >= 2.0)"""
#     def __init__(self, dim: int, heads: int = 8):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.heads = heads
#         inner_dim = dim // heads
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Linear(dim, dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (b, c, h, w) → (b, h, w, c)
#         x = rearrange(x, "b c h w -> b h w c")
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(
#             lambda t: rearrange(t, "b h w (heads d) -> b heads h w d", heads=self.heads),
#             qkv
#         )

#         # Use scaled dot-product attention (flash when possible)
#         out = F.scaled_dot_product_attention(
#             q, k, v,
#             attn_mask=None,
#             dropout_p=0.0 if not self.training else 0.1,
#             is_causal=False
#         )

#         out = rearrange(out, "b heads h w d -> b h w (heads d)")
#         out = self.to_out(out)
#         out = rearrange(out, "b h w c -> b c h w")
#         return out



# #  Residual Block 
# class ResBlock(nn.Module):
#     def __init__(
#         self,
#         in_ch: int,
#         out_ch: int,
#         groups: int = 32,
#         dropout: float = 0.1,
#         use_attn: bool = False,
#         heads: int = 8,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(in_ch, groups)
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

#         self.norm2 = norm_layer(out_ch, groups)
#         self.dropout = nn.Dropout(dropout)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

#         self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

#         self.attn = FlashAttention(out_ch, heads) if use_attn else None

#         self.act = SiLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.act(self.norm1(x))
#         h = self.conv1(h)

#         h = self.act(self.norm2(h))
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.attn is not None:
#             h = h + self.attn(h)

#         return self.skip(x) + h



# #   Down / Up sample
# class Downsample(nn.Module):
#     def __init__(self, ch: int):
#         super().__init__()
#         self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

#     def forward(self, x):
#         return self.conv(x)


# class Upsample(nn.Module):
#     def __init__(self, ch: int):
#         super().__init__()
#         self.conv = nn.Conv2d(ch, ch, 3, padding=1)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2.0, mode="nearest")
#         return self.conv(x)


# #   Pure image-input U-Net 
# class UNet(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 3,
#         base_channels: int = 128,
#         channel_mult: tuple = (1, 2, 4, 8),
#         num_res_blocks: int = 2,
#         attention_levels: tuple = (2, 3),     
#         dropout: float = 0.1,
#         use_flash_attn: bool = True,
#         out_channels: int = 3,
#     ):
#         super().__init__()

#         levels = len(channel_mult)

       
#         #   Input
#         self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

#         # -------------------------------------------------------------------------
#         #   Down blocks
#         # -------------------------------------------------------------------------
#         self.down_blocks = nn.ModuleList([])
#         ch = base_channels
#         for level, mult in enumerate(channel_mult):
#             out_ch = base_channels * mult
#             for _ in range(num_res_blocks):
#                 use_attn = level in attention_levels
#                 self.down_blocks.append(
#                     ResBlock(
#                         ch, out_ch,
#                         dropout=dropout,
#                         use_attn=use_attn,
#                         heads=8 if use_flash_attn else 8
#                     )
#                 )
#                 ch = out_ch
#             if level != levels - 1:
#                 self.down_blocks.append(Downsample(ch))

#         # -------------------------------------------------------------------------
#         #   Middle block
#         # -------------------------------------------------------------------------
#         self.mid_block1 = ResBlock(ch, ch, dropout=dropout, use_attn=True)
#         self.mid_block2 = ResBlock(ch, ch, dropout=dropout, use_attn=False)

#         # -------------------------------------------------------------------------
#         #   Up blocks
#         # -------------------------------------------------------------------------
#         self.up_blocks = nn.ModuleList([])
#         for level in reversed(range(levels)):
#             out_ch = base_channels * channel_mult[level]
#             for i in range(num_res_blocks + 1):  # +1 for skip concat
#                 in_ch = ch + out_ch if i == 0 else out_ch
#                 use_attn = level in attention_levels
#                 self.up_blocks.append(
#                     ResBlock(
#                         in_ch, out_ch,
#                         dropout=dropout,
#                         use_attn=use_attn
#                     )
#                 )
#                 ch = out_ch
#             if level != 0:
#                 self.up_blocks.append(Upsample(ch))

#         # -------------------------------------------------------------------------
#         #   Output
#         # -------------------------------------------------------------------------
#         self.norm_out = norm_layer(ch)
#         self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (b, 3, 256, 256)

#         h = self.input_conv(x)
#         skips = []

#         # Down
#         for block in self.down_blocks:
#             if isinstance(block, ResBlock):
#                 h = block(h)
#             else:
#                 h = block(h)
#             skips.append(h)

#         # Middle
#         h = self.mid_block1(h)
#         h = self.mid_block2(h)

#         # Up
#         for block in self.up_blocks:
#             if isinstance(block, ResBlock):
#                 skip = skips.pop()
#                 h = torch.cat([h, skip], dim=1)
#                 h = block(h)
#             else:
#                 h = block(h)

#         # Final
#         h = self.norm_out(h)
#         h = F.silu(h)
#         out = self.out_conv(h)

#         return out


# # -------------------------------------------------------------------------
# #   Example instantiation & forward
# # -------------------------------------------------------------------------

# model = ImageOnlyUNet(
#     in_channels=3,
#     base_channels=192,              # tune this (128–256 common)
#     channel_mult=(1, 2, 4, 8),
#     num_res_blocks=2,
#     attention_levels=(2, 3),
#     dropout=0.1,
#     out_channels=3
# ).cuda()

# dummy = torch.randn(8, 3, 256, 256).cuda()
# pred = model(dummy)               # → (8, 3, 256, 256)
# print(pred.shape)