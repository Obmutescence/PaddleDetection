# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Uniform, Normal, XavierUniform
from paddle.regularizer import L2Decay

from ppdet.modeling.ops import batch_norm
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import (
    DeformableConvV2,
    ConvNormLayer,
    LiteConv,
    DCNv3_paddle,
)
from ..shape_spec import ShapeSpec
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from .ccanet import CrissCrossAttention, RCCAWrapper, RCCAModule
from ..backbones.cspresnet import RepVggBlock
from ..backbones.csp_darknet import BaseConv
from paddle.jit import to_static

__all__ = ["TTFFPN"]


class CSPRepLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
    ):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act
        )
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, act=act)
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels, out_channels, ksize=1, stride=1, bias=bias, act=act
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        ch_in,
        ch_out,
        filter_size=3,
        stride=1,
        groups=1,
        padding=0,
        norm_type="bn",
        norm_decay=0.0,
        act="leaky",
        freeze_norm=False,
        data_format="NCHW",
        name="",
    ):
        """
        conv + bn + activation layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 1
            groups (int): number of groups of conv layer, default 1
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            act (str): activation function type, default 'leaky', which means leaky_relu
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            data_format=data_format,
            bias_attr=False,
        )
        self.batch_norm = batch_norm(
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format,
        )
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == "leaky":
            out = F.leaky_relu(out, 0.1)
        else:
            out = getattr(F, self.act)(out)
        return out


class Upsample(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type="bn", use_dcn=True):
        super(Upsample, self).__init__()
        fan_in = ch_in * 3 * 3
        stdv = 1.0 / math.sqrt(fan_in)
        self.use_dcn = use_dcn
        if use_dcn:
            self.dcn = DeformableConvV2(
                ch_in,
                ch_out,
                kernel_size=3,
                weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                bias_attr=ParamAttr(
                    initializer=Constant(0), regularizer=L2Decay(0.0), learning_rate=2.0
                ),
                lr_scale=2.0,
                regularizer=L2Decay(0.0),
            )
        else:
            self.conv = nn.Conv2D(
                ch_in,
                ch_out,
                kernel_size=3,
                padding=1,
                weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
                bias_attr=ParamAttr(
                    initializer=Constant(0), regularizer=L2Decay(0.0), learning_rate=2.0
                ),
            )

        self.bn = batch_norm(ch_out, norm_type=norm_type, initializer=Constant(1.0))

    def forward(self, feat):
        if self.use_dcn:
            x = self.dcn(feat)
        else:
            x = self.conv(feat)
        bn = self.bn(x)
        relu = F.relu(bn)
        out = F.interpolate(relu, scale_factor=2.0, mode="bilinear")
        return out


class UpsampleDCNv3(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type="bn"):
        super(UpsampleDCNv3, self).__init__()
        # fan_in = ch_in * 3 * 3
        # stdv = 1.0 / math.sqrt(fan_in)
        self.dcn = DCNv3_paddle(
            ch_in,
            # ch_out,
            kernel_size=3,
            # weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            # bias_attr=ParamAttr(
            #     initializer=Constant(0), regularizer=L2Decay(0.0), learning_rate=2.0
            # ),
            # lr_scale=2.0,
            # regularizer=L2Decay(0.0),
        )
        self.bn = batch_norm(ch_in, norm_type=norm_type, initializer=Constant(1.0))
        self.conv = ConvNormLayer(
            ch_in, ch_out, filter_size=3, stride=1, norm_type="sync_bn"
        )

    def forward(self, feat):
        dcn = self.dcn(feat)
        dcn = dcn.transpose([0, 3, 1, 2])
        bn = self.bn(dcn)
        relu = F.relu(bn)
        out = F.interpolate(relu, scale_factor=2.0, mode="bilinear")
        out = self.conv(out)
        return out


class DeConv(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type="bn"):
        super(DeConv, self).__init__()
        self.deconv = nn.Sequential()
        conv1 = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            stride=1,
            filter_size=1,
            norm_type=norm_type,
            initializer=XavierUniform(),
        )
        conv2 = nn.Conv2DTranspose(
            in_channels=ch_out,
            out_channels=ch_out,
            kernel_size=4,
            padding=1,
            stride=2,
            groups=ch_out,
            weight_attr=ParamAttr(initializer=XavierUniform()),
            bias_attr=False,
        )
        bn = batch_norm(ch_out, norm_type=norm_type, norm_decay=0.0)
        conv3 = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            stride=1,
            filter_size=1,
            norm_type=norm_type,
            initializer=XavierUniform(),
        )

        self.deconv.add_sublayer("conv1", conv1)
        self.deconv.add_sublayer("relu6_1", nn.ReLU6())
        self.deconv.add_sublayer("conv2", conv2)
        self.deconv.add_sublayer("bn", bn)
        self.deconv.add_sublayer("relu6_2", nn.ReLU6())
        self.deconv.add_sublayer("conv3", conv3)
        self.deconv.add_sublayer("relu6_3", nn.ReLU6())

    def forward(self, inputs):
        return self.deconv(inputs)


class LiteUpsample(nn.Layer):
    def __init__(self, ch_in, ch_out, norm_type="bn"):
        super(LiteUpsample, self).__init__()
        self.deconv = DeConv(ch_in, ch_out, norm_type=norm_type)
        self.conv = LiteConv(ch_in, ch_out, norm_type=norm_type)

    def forward(self, inputs):
        deconv_up = self.deconv(inputs)
        conv = self.conv(inputs)
        interp_up = F.interpolate(conv, scale_factor=2.0, mode="bilinear")
        return deconv_up + interp_up


class ShortCut(nn.Layer):
    def __init__(
        self,
        layer_num,
        ch_in,
        ch_out,
        norm_type="bn",
        lite_neck=False,
        name=None,
        use_cca=False,
    ):
        super(ShortCut, self).__init__()
        self.use_cca = use_cca
        shortcut_conv = nn.Sequential()
        for i in range(layer_num):
            fan_out = 3 * 3 * ch_out
            std = math.sqrt(2.0 / fan_out)
            in_channels = ch_in if i == 0 else ch_out
            shortcut_name = name + ".conv.{}".format(i)
            if lite_neck:
                shortcut_conv.add_sublayer(
                    shortcut_name,
                    LiteConv(
                        in_channels=in_channels,
                        out_channels=ch_out,
                        with_act=i < layer_num - 1,
                        norm_type=norm_type,
                    ),
                )
            else:
                if self.use_cca:
                    conv_model = RCCAWrapper(in_channels, ch_out, 1)
                    # conv_model = CrissCrossAttention(in_channels)
                else:
                    conv_model = nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=ch_out,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(initializer=Normal(0, std)),
                        bias_attr=ParamAttr(
                            learning_rate=2.0, regularizer=L2Decay(0.0)
                        ),
                    )

                shortcut_conv.add_sublayer(
                    shortcut_name,
                    conv_model,
                )
                if i < layer_num - 1:
                    shortcut_conv.add_sublayer(shortcut_name + ".act", nn.ReLU())
        self.shortcut = self.add_sublayer("shortcut", shortcut_conv)

    def forward(self, feat):
        out = self.shortcut(feat)
        return out


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])


@register
class TTFTransformerLayer(nn.Layer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        rezero=False,
    ):
        super(TTFTransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

        self.rezero = rezero
        if not rezero:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.resweight = 1
        else:
            self.resweight = self.create_parameter(
                shape=(1,), attr=ParamAttr(initializer=Constant(0.0))
            )

        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    @to_static(full_graph=True)
    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if not self.rezero and self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src *= self.resweight

        src = residual + self.dropout1(src)
        if not self.rezero and not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if not self.rezero and self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src *= self.resweight

        src = residual + self.dropout2(src)
        if not self.rezero and not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
@serializable
class TTFFPN(nn.Layer):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [128,256,512,1024] by default, means the channels of DarkNet53
            backbone return_idx [1,2,3,4].
        planes (list): the number of output feature channels of FPN.
            [256, 128, 64] by default
        shortcut_num (list): the number of convolution layers in each shortcut.
            [3,2,1] by default, means DarkNet53 backbone return_idx_1 has 3 convs
            in its shortcut, return_idx_2 has 2 convs and return_idx_3 has 1 conv.
        norm_type (string): norm type, 'sync_bn', 'bn', 'gn' are optional.
            bn by default
        lite_neck (bool): whether to use lite conv in TTFNet FPN,
            False by default
        fusion_method (string): the method to fusion upsample and lateral layer.
            'add' and 'concat' are optional, add by default
    """

    __shared__ = ["norm_type"]
    __inject__ = ["encoder_layer"]

    def __init__(
        self,
        in_channels,
        planes=[256, 128, 64],
        shortcut_num=[3, 2, 1],
        norm_type="bn",
        lite_neck=False,
        dcnv3_neck=False,
        fusion_method="add",
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        encoder_layer=None,
        hidden_dim=256,
        pe_temperature=10000,
        eval_size=None,
        feat_strides=[4, 8, 16, 32],
        num_classes=80,  # COCO 多分类
        aux_multi_cls=False,
        pan=False,
        act="silu",
        use_cca=False,
        use_dcn_list=[True, True, True],
    ):
        super(TTFFPN, self).__init__()
        self.planes = planes
        self.shortcut_num = shortcut_num[::-1]
        self.shortcut_len = len(shortcut_num)
        self.ch_in = in_channels[::-1]
        self.fusion_method = fusion_method
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        # self.hidden_dim = hidden_dim
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size
        self.feat_strides = feat_strides

        self.pan = pan

        self.upsample_list = []
        self.shortcut_list = []
        self.upper_list = []
        for i, out_c in enumerate(self.planes):
            in_c = self.ch_in[i] if i == 0 else self.upper_list[-1]
            upsample_module = LiteUpsample if lite_neck else Upsample
            upsample_module = UpsampleDCNv3 if dcnv3_neck else upsample_module
            upsample = self.add_sublayer(
                "upsample." + str(i),
                upsample_module(
                    in_c, out_c, norm_type=norm_type, use_dcn=use_dcn_list[i]
                ),
            )
            self.upsample_list.append(upsample)
            if i < self.shortcut_len:
                shortcut = self.add_sublayer(
                    "shortcut." + str(i),
                    ShortCut(
                        self.shortcut_num[i],
                        self.ch_in[i + 1],
                        out_c,
                        norm_type=norm_type,
                        lite_neck=lite_neck,
                        name="shortcut." + str(i),
                        use_cca=use_cca,
                    ),
                )
                self.shortcut_list.append(shortcut)
                if self.fusion_method == "add":
                    upper_c = out_c
                elif self.fusion_method == "concat":
                    upper_c = out_c * 2
                else:
                    raise ValueError(
                        "Illegal fusion method. Expected add or\
                        concat, but received {}".format(
                            self.fusion_method
                        )
                    )
                self.upper_list.append(upper_c)

        # encoder transformer
        self.encoder = nn.LayerList(
            [
                TransformerEncoder(encoder_layer, num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        # 在这里添加辅助多分类器
        self.aux_multi_cls = aux_multi_cls
        if aux_multi_cls:
            self.conv1x1 = ConvBNLayer(self.ch_in[0], hidden_dim, filter_size=1)
            self.avg_pool = nn.AdaptiveAvgPool2D(1)
            self.linear = nn.Conv2D(hidden_dim, num_classes, kernel_size=1)

        if self.pan:
            # 路径聚合, 从大特征图向小特征图走, 只是feature map 不再减小
            self.downsample_convs = nn.LayerList()
            self.pan_blocks = nn.LayerList()
            for idx in range(len(in_channels) - 1):
                self.downsample_convs.append(
                    BaseConv(hidden_dim, hidden_dim, 3, stride=2, act=act)
                )
                self.pan_blocks.append(
                    CSPRepLayer(
                        hidden_dim * 2,
                        hidden_dim,
                        round(3 * depth_mult),
                        act=act,
                        expansion=expansion,
                    )
                )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride,
                    self.eval_size[0] // stride,
                    self.hidden_dim,  # 1024
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w),
                paddle.cos(out_w),
                paddle.sin(out_h),
                paddle.cos(out_h),
            ],
            axis=1,
        )[None, :, :]

    def forward(self, inputs):

        assert len(inputs) == len(self.ch_in)
        # get projection features
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(inputs)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                B, _, h, w = inputs[enc_ind].shape
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = inputs[enc_ind].flatten(2).transpose([0, 2, 1])
                if self.training or self.eval_size is None:
                    # self.hidden_dim = 1024
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, inputs[enc_ind].shape[1], self.pe_temperature
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                inputs[enc_ind] = memory.transpose([0, 2, 1]).reshape([B, -1, h, w])

        feat = inputs[-1]

        # 辅助多分类器
        if self.aux_multi_cls:
            aux_mul_cls = self.linear(self.avg_pool(self.conv1x1(feat))).squeeze_(
                [2, 3]
            )
        else:
            aux_mul_cls = None

        # feature map size 从 小 到 大
        inner_outs = []
        for i, out_c in enumerate(self.planes):
            feat = self.upsample_list[i](feat)
            if i < self.shortcut_len:
                shortcut = self.shortcut_list[i](inputs[-i - 2])
                if self.fusion_method == "add":
                    feat = feat + shortcut
                else:
                    feat = paddle.concat([feat, shortcut], axis=1)
            inner_outs.insert(0, feat)

        # TODO: Copy From RT-DETR
        # # feature map size 再从大 -> 小走一次, 只是size不减小
        # for idx in range(len(self.ch_in) - 1):

        #     feat_height = inner_outs[idx]
        #     downsample_feat = self.downsample_convs[idx](feat)
        #     out = self.pan_blocks[idx](
        #         paddle.concat([downsample_feat, feat_height], axis=1)
        #     )

        return feat, aux_mul_cls

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [i.channels for i in input_shape],
            "feat_strides": [i.stride for i in input_shape],
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.upper_list[-1],
            )
        ]
