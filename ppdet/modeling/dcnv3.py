# https://github.com/OpenGVLab/InternImage/blob/master/detection/ops_dcnv3/modules/dcnv3.py

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings

import paddle
from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant, XavierUniform

if __name__ == "__main__":
    from initializer import xavier_uniform_, constant_
else:
    from .initializer import xavier_uniform_, constant_

__all__ = ["DCNv3_paddle"]


class to_channels_first(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose([0, 3, 1, 2])


class to_channels_last(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose([0, 2, 3, 1])


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2D(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, epsilon=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Layer):
    def forward(
        self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias
    ):
        center_feature_scale = F.linear(
            query,
            weight=center_feature_scale_proj_weight.T,
            bias=center_feature_scale_proj_bias,
        )
        center_feature_scale = F.sigmoid(center_feature_scale)
        return center_feature_scale


def _get_reference_points(
    spatial_shapes,
    device,
    kernel_h,
    kernel_w,
    dilation_h,
    dilation_w,
    pad_h=0,
    pad_w=0,
    stride_h=1,
    stride_w=1,
):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = paddle.meshgrid(
        paddle.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype="float32",
        ),
        paddle.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype="float32",
        ),
    )
    ref_y = ref_y.flatten()[None] / H_
    ref_x = ref_x.flatten()[None] / W_

    ref = paddle.stack((ref_x, ref_y), -1).reshape([1, H_out, W_out, 1, 2])

    return ref


def _generate_dilation_grids(
    spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device
):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = paddle.meshgrid(
        paddle.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w,
            dtype="float32",
        ),
        paddle.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h,
            dtype="float32",
        ),
    )

    points_list.extend([x / W_, y / H_])
    grid = (
        paddle.stack(points_list, -1)
        .reshape([-1, 1, 2])
        .tile([1, group, 1])
        .transpose([1, 0, 2])
    )
    grid = grid.reshape([1, 1, 1, group * kernel_h * kernel_w, 2])

    return grid


def dcnv3_core_pytorch(
    _input,
    offset,
    mask,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    group,
    group_channels,
    offset_scale,
):
    # for debug and test only,
    # need to use cuda version instead
    input = F.pad(
        _input, pad=[0, 0, pad_w, pad_w, pad_h, pad_h, 0, 0], mode="constant", value=0
    )
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape,
        input.place,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.place
    )
    spatial_norm = (
        paddle.to_tensor([W_in, H_in])
        .reshape([1, 1, 1, 2])
        .tile([1, 1, 1, group * kernel_h * kernel_w])
    )

    sampling_locations = (ref + grid * offset_scale).tile([N_, 1, 1, 1, 1]).flatten(
        3, 4
    ) + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = (
        input.reshape([N_, H_in * W_in, group * group_channels])
        .transpose([0, 2, 1])
        .reshape([N_ * group, group_channels, H_in, W_in])
    )
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = (
        sampling_grids.reshape([N_, H_out * W_out, group, P_, 2])
        .transpose([0, 2, 1])
        .flatten(0, 1)
    )
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_,
        sampling_grid_,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = (
        mask.reshape([N_, H_out * W_out, group, P_])
        .transpose([0, 2, 1])
        .reshape([N_ * group, 1, H_out * W_out, P_])
    )
    output = (
        (sampling_input_ * mask)
        .sum(-1)
        .reshape([N_, group * group_channels, H_out * W_out])
    )

    return output.transpose([0, 2, 1]).reshape([N_, H_out, W_out, -1]).contiguous()


class DCNv3_paddle(nn.Layer):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer="GELU",
        norm_layer="LN",
        center_feature_scale=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {group}"
            )
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2D(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels,
            ),
            build_norm_layer(channels, norm_layer, "channels_first", "channels_last"),
            build_act_layer(act_layer),
        )
        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        if center_feature_scale:

            self.center_feature_scale_proj_weight = self.create_parameter(
                shape=[group, channels],
                attr=ParamAttr(initializer=Constant(0.0)),
                dtype="float32",
            )

            self.center_feature_scale_proj_bias = self.create_parameter(
                shape=[group],
                attr=ParamAttr(initializer=Constant(0.0)),
                dtype="float32",
            )
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """

        input = input.transpose([0, 2, 3, 1])
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x

        x1 = input.transpose([0, 3, 1, 2])
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape([N, H, W, self.group, -1])
        mask = F.softmax(mask, -1).reshape([N, H, W, -1])

        x = dcnv3_core_pytorch(
            x,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
        )
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1,
                self.center_feature_scale_proj_weight,
                self.center_feature_scale_proj_bias,
            )
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = (
                center_feature_scale[..., None]
                .tile([1, 1, 1, 1, self.channels // self.group])
                .flatten(-2)
            )
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


@paddle.no_grad()
def check_forward_equal_with_paddle_double():
    input = paddle.rand([N, H_in, W_in, M * D]) * 0.01
    offset = paddle.rand([N, H_out, W_out, M * P * 2]) * 10
    mask = paddle.rand([N, H_out, W_out, M, P]) + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape([N, H_out, W_out, M * P])

    output_pytorch = (
        dcnv3_core_pytorch(
            input.astype("float64"),
            offset.astype("float64"),
            mask.astype("float64"),
            Kh,
            Kw,
            stride,
            stride,
            Kh // 2,
            Kw // 2,
            dilation,
            dilation,
            M,
            D,
            offset_scale,
        )
        .detach()
        .cpu()
    )


@paddle.no_grad()
def check_forward_equal_with_paddle_float():
    input = paddle.rand([N, H_in, W_in, M * D]) * 0.01
    offset = paddle.rand([N, H_out, W_out, M * P * 2]) * 10
    mask = paddle.rand([N, H_out, W_out, M, P]) + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape([N, H_out, W_out, M * P])

    output_pytorch = (
        dcnv3_core_pytorch(
            input,
            offset,
            mask,
            Kh,
            Kw,
            stride,
            stride,
            Kh // 2,
            Kw // 2,
            dilation,
            dilation,
            M,
            D,
            offset_scale,
        )
        .detach()
        .cpu()
    )


def check_backward_equal_with_paddle_double(
    channels=4, grad_input=True, grad_offset=True, grad_mask=True
):
    # H_in, W_in = 4, 4
    N = 2
    M = 2
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    D = channels
    input0 = paddle.rand([N, H_in, W_in, M * D]) * 0.01
    offset0 = paddle.rand([N, H_out, W_out, M * P * 2]) * 10
    mask0 = paddle.rand([N, H_out, W_out, M, P]) + 1e-5
    mask0 /= mask0.sum(-1, keepdim=True)
    mask0 = mask0.reshape([N, H_out, W_out, M * P])
    input0.stop_gradient = not grad_input
    offset0.stop_gradient = not grad_offset
    mask0.stop_gradient = not grad_mask

    output_pytorch = dcnv3_core_pytorch(
        input0.astype("float64"),
        offset0.astype("float64"),
        mask0.astype("float64"),
        Kh,
        Kw,
        stride,
        stride,
        Kh // 2,
        Kw // 2,
        dilation,
        dilation,
        M,
        D,
        offset_scale,
    )
    output_pytorch.sum().backward()

    input1 = input0.detach()
    offset1 = offset0.detach()
    mask1 = mask0.detach()
    input1.stop_gradient = not grad_input
    offset1.stop_gradient = not grad_offset
    mask1.stop_gradient = not grad_mask

    # print(f">>> backward double: channels {D}")
    # bwdok = paddle.allclose(input0.grad, input1.grad, rtol=1e-2, atol=1e-3)
    # max_abs_err = (input0.grad - input1.grad).abs().max()
    # max_rel_err = ((input0.grad - input1.grad).abs() / input0.grad.abs()).max()
    # print(
    #     f"* {bwdok} input_grad check_backward_equal_with_paddle_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    # )

    # bwdok = paddle.allclose(offset0.grad, offset1.grad, rtol=1e-2, atol=1e-3)
    # max_abs_err = (offset0.grad - offset1.grad).abs().max()
    # max_rel_err = ((offset0.grad - offset1.grad).abs() / offset0.grad.abs()).max()
    # print(
    #     f"* {bwdok} offset_grad check_backward_equal_with_paddle_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    # )

    # bwdok = paddle.allclose(mask0.grad, mask1.grad, rtol=1e-2, atol=1e-3)
    # max_abs_err = (mask0.grad - mask1.grad).abs().max()
    # max_rel_err = ((mask0.grad - mask1.grad).abs() / mask0.grad.abs()).max()
    # print(
    #     f"* {bwdok} mask_grad check_backward_equal_with_paddle_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    # )


def check_backward_equal_with_paddle_float(
    channels=4, grad_input=True, grad_offset=True, grad_mask=True
):
    # H_in, W_in = 4, 4
    N = 2
    M = 2
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    D = channels
    input0 = paddle.rand([N, H_in, W_in, M * D]) * 0.01
    offset0 = paddle.rand([N, H_out, W_out, M * P * 2]) * 10
    mask0 = paddle.rand([N, H_out, W_out, M, P]) + 1e-5
    mask0 /= mask0.sum(-1, keepdim=True)
    mask0 = mask0.reshape([N, H_out, W_out, M * P])
    input0.stop_gradient = not grad_input
    offset0.stop_gradient = not grad_offset
    mask0.stop_gradient = not grad_mask

    output_pytorch = dcnv3_core_pytorch(
        input0,
        offset0,
        mask0,
        Kh,
        Kw,
        stride,
        stride,
        Kh // 2,
        Kw // 2,
        dilation,
        dilation,
        M,
        D,
        offset_scale,
    )
    output_pytorch.sum().backward()

    input1 = input0.detach()
    offset1 = offset0.detach()
    mask1 = mask0.detach()
    input1.stop_gradient = not grad_input
    offset1.stop_gradient = not grad_offset
    mask1.stop_gradient = not grad_mask

    # print(f">>> backward float: channels {D}")
    # bwdok = paddle.allclose(input0.grad, input1.grad, rtol=1e-2, atol=1e-3)
    # max_abs_err = (input0.grad - input1.grad).abs().max()
    # max_rel_err = ((input0.grad - input1.grad).abs() / input0.grad.abs()).max()
    # print(
    #     f"* {bwdok} input_grad check_backward_equal_with_paddle_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    # )

    # bwdok = paddle.allclose(offset0.grad, offset1.grad, rtol=1e-2, atol=1e-3)
    # max_abs_err = (offset0.grad - offset1.grad).abs().max()
    # max_rel_err = ((offset0.grad - offset1.grad).abs() / offset0.grad.abs()).max()
    # print(
    #     f"* {bwdok} offset_grad check_backward_equal_with_paddle_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    # )

    # bwdok = paddle.allclose(mask0.grad, mask1.grad, rtol=1e-2, atol=1e-3)
    # max_abs_err = (mask0.grad - mask1.grad).abs().max()
    # max_rel_err = ((mask0.grad - mask1.grad).abs() / mask0.grad.abs()).max()
    # print(
    #     f"* {bwdok} mask_grad check_backward_equal_with_paddle_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    # )


@paddle.no_grad()
def check_time_cost(im2col_step=128):
    N = 512
    H_in, W_in = 64, 64
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    input = paddle.rand([N, H_in, W_in, M * D]) * 0.01
    offset = paddle.rand([N, H_out, W_out, M * P * 2]) * 10
    mask = paddle.rand([N, H_out, W_out, M, P]) + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape([N, H_out, W_out, M * P])
    print(f">>> time cost: im2col_step {im2col_step}; input {input.shape}; points {P} ")


if __name__ == "__main__":

    H_in, W_in = 8, 8
    N, M, D = 2, 4, 16
    Kh, Kw = 3, 3
    P = Kh * Kw
    offset_scale = 2.0
    pad = 1
    dilation = 1
    stride = 1
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    check_forward_equal_with_paddle_double()
    check_forward_equal_with_paddle_float()
    for channels in [1, 16, 30, 32, 64, 71, 1025]:
        check_backward_equal_with_paddle_double(channels, True, True, True)
    for channels in [1, 16, 30, 32, 64, 71, 1025]:
        check_backward_equal_with_paddle_float(channels, True, True, True)
    for i in range(3):
        im2col_step = 128 * (2**i)
        check_time_cost(im2col_step)

    x = paddle.rand([4, 128, 256, 256])
    x = x.transpose([0, 2, 3, 1])  # 通道在最后一维
    model = DCNv3_paddle(128, center_feature_scale=True)
    y = model(x)
    print(y.shape)
