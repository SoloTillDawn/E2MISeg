import torch.nn as nn
import torch
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers import get_act_layer, get_norm_layer

from e2miseg.network_architecture.dynunet_block import UnetResBlock
import math

class HFRBlock(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = TransformerBlock(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.conv31 = get_conv_layer(3, hidden_size, hidden_size, kernel_size=3, stride=1, conv_only=True,)
        self.lrelu31 = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.conv32 = get_conv_layer(3, hidden_size, hidden_size, kernel_size=3, stride=1, conv_only=True,)
        self.lrelu32 = get_act_layer(name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.conv9 = get_conv_layer(3, hidden_size, hidden_size, kernel_size=1, stride=1, dropout=0.15, bias=False,)

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        x_skip = x
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        conv_skip = x_skip
        conv_m = self.conv31(conv_skip)
        conv_m = self.lrelu31(conv_m)
        conv_m = self.conv32(conv_m)
        conv_m = self.lrelu32(conv_m)

        x = x + conv_m
        x = self.conv9(x) + x_skip

        return x


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_s, k_s, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_s = q_s.transpose(-2, -1)
        k_s = k_s.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
        k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_s, v_SA), (self.EF, self.EF)))

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)

        attn_CAB = (q_s @ k_s.transpose(-2, -1)) * self.temperature1
        attn_CAB = attn_CAB.softmax(dim=-1)
        attn_CAB = self.attn_drop(attn_CAB)
        x_CA = (attn_CAB @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SAB = (q_s.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SAB = attn_SAB.softmax(dim=-1)
        attn_SAB = self.attn_drop_2(attn_SAB)
        x_SA = (attn_SAB @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        return x_CA + x_SA

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
