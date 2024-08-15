import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import

from e2miseg.network_architecture.layers import LayerNorm
from e2miseg.network_architecture.tumor.transformerblock import TransformerBlock, HFRBlock
from e2miseg.network_architecture.dynunet_block import get_conv_layer, UnetResBlock, UnetOutBlock


einops, _ = optional_import("einops")

class MFGA(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_classes: int,
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            r: Union[Sequence[int], int],
    ) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.maskout = UnetOutBlock(spatial_dims=3,in_channels=out_channels,out_channels=num_classes)  # out_channels=9
        self.in_c = out_channels//2 + num_classes
        self.groupin_c = out_channels*2 + num_classes*4

        self.layernorm = nn.LayerNorm(self.in_c)
        self.layernorm1 = nn.LayerNorm(self.groupin_c)
        self.dilation_conv1 = nn.Conv3d(self.in_c, self.in_c, kernel_size=3,stride=1,padding=r[0],dilation=r[0],bias=True,groups=self.in_c)
        self.dilation_conv2 = nn.Conv3d(self.in_c, self.in_c, kernel_size=3,stride=1,padding=r[1],dilation=r[1],bias=True,groups=self.in_c)
        self.dilation_conv3 = nn.Conv3d(self.in_c, self.in_c, kernel_size=3,stride=1,padding=r[2],dilation=r[2],bias=True,groups=self.in_c)
        self.dilation_conv4 = nn.Conv3d(self.in_c, self.in_c, kernel_size=3,stride=1,padding=r[3],dilation=r[3],bias=True,groups=self.in_c)
        self.conv = get_conv_layer(spatial_dims=3,in_channels=self.groupin_c,out_channels=out_channels,kernel_size=1,conv_only=True,)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        mask = self.maskout(out)

        out_group = torch.chunk(out, chunks=4, dim=1)
        skip_group = torch.chunk(skip, chunks=4, dim=1)

        group0 = torch.cat((out_group[0], skip_group[0], mask), dim=1)
        B, C, H, W, D = group0.shape
        group0 = group0.reshape(B, C, H*W*D).permute(0, 2, 1)
        group0 = self.layernorm(group0)
        group0 = group0.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        group0 = self.dilation_conv1(group0)

        group1 = torch.cat((out_group[1], skip_group[1], mask), dim=1)
        group1 = group1.reshape(B, C, H*W*D).permute(0, 2, 1)
        group1 = self.layernorm(group1)
        group1 = group1.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        group1 = self.dilation_conv2(group1)

        group2 = torch.cat((out_group[2], skip_group[2], mask), dim=1)
        group2 = group2.reshape(B, C, H*W*D).permute(0, 2, 1)
        group2 = self.layernorm(group2)
        group2 = group2.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        group2 = self.dilation_conv3(group2)

        group3 = torch.cat((out_group[3], skip_group[3], mask), dim=1)
        group3 = group3.reshape(B, C, H*W*D).permute(0, 2, 1)
        group3 = self.layernorm(group3)
        group3 = group3.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        group3 = self.dilation_conv4(group3)

        x = torch.cat((group0,group1,group2,group3), dim=1)
        B1, C1, H1, W1, D1 = x.shape
        x = x.reshape(B1, C1, H1*W1*D1).permute(0, 2, 1)
        x = self.layernorm1(x)
        x = x.reshape(B1, H1, W1, D1, C1).permute(0, 4, 1, 2, 3)
        x = self.conv(x)

        return x


class E2MISEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=4,
                 dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(4, 4, 4), stride=(4, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(HFRBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        hidden_states.append(x)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class E2MISDecoder(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.decoder_block = nn.ModuleList()

        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(HFRBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out
