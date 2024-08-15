from torch import nn
from typing import Tuple, Union

from e2miseg.network_architecture.neural_network import SegmentationNetwork
from e2miseg.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from e2miseg.network_architecture.tumor.model_components import MFGA, E2MISEncoder, E2MISDecoder


class E2MISeg(SegmentationNetwork):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,
    ) -> None:

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (4, 4, 4,)
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = E2MISEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = E2MISDecoder(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8*8*8,
        )
        self.decoder4 = E2MISDecoder(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16*16*16,
        )
        self.decoder3 = E2MISDecoder(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32*32*32,
        )
        self.decoder2 = E2MISDecoder(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=128*128*128,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

        self.mfgam0 = MFGA(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            num_classes=out_channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            r=[1, 2, 6, 7]
        )
        self.mfgam1 = MFGA(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            num_classes=out_channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            r=[1, 2, 6, 7]
        )
        self.mfgam2 = MFGA(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            num_classes=out_channels,
            upsample_kernel_size=2,
            norm_name=norm_name,
            r=[1, 2, 6, 7]
        )
        self.mfgam3 = MFGA(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size * 1,
            num_classes=out_channels,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            r=[1, 2, 6, 7]
        )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder(x_in)

        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        mfga0 = self.mfgam0(dec4, enc3)
        dec3 = self.decoder5(dec4, mfga0)
        mfga1 = self.mfgam1(dec3, enc2)
        dec2 = self.decoder4(dec3, mfga1)
        mfga2 = self.mfgam2(dec2, enc1)
        dec1 = self.decoder3(dec2, mfga2)
        mfga3 = self.mfgam3(dec1, convBlock)
        out = self.decoder2(dec1, mfga3)

        if self.do_ds:
            l = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            l = self.out1(out)

        return l
