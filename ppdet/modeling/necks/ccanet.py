import os

import paddle
import paddle.nn as nn


class CrissCrossAttention(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.q_conv = nn.Conv2D(in_channels, in_channels // 8, kernel_size=1)
        self.k_conv = nn.Conv2D(in_channels, in_channels // 8, kernel_size=1)
        self.v_conv = nn.Conv2D(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(axis=3)
        self.gamma = self.create_parameter(
            shape=(1,), default_initializer=nn.initializer.Constant(0)
        )
        self.inf_tensor = paddle.full(shape=(1,), fill_value=float("inf"))

    def forward(self, x):
        b, c, h, w = paddle.shape(x)
        proj_q = self.q_conv(x)
        proj_q_h = (
            proj_q.transpose([0, 3, 1, 2]).reshape([b * w, -1, h]).transpose([0, 2, 1])
        )
        proj_q_w = (
            proj_q.transpose([0, 2, 1, 3]).reshape([b * h, -1, w]).transpose([0, 2, 1])
        )

        proj_k = self.k_conv(x)
        proj_k_h = proj_k.transpose([0, 3, 1, 2]).reshape([b * w, -1, h])
        proj_k_w = proj_k.transpose([0, 2, 1, 3]).reshape([b * h, -1, w])

        proj_v = self.v_conv(x)
        proj_v_h = proj_v.transpose([0, 3, 1, 2]).reshape([b * w, -1, h])
        proj_v_w = proj_v.transpose([0, 2, 1, 3]).reshape([b * h, -1, w])

        energy_h = (
            (paddle.bmm(proj_q_h, proj_k_h) + self.Inf(b, h, w))
            .reshape([b, w, h, h])
            .transpose([0, 2, 1, 3])
        )
        energy_w = paddle.bmm(proj_q_w, proj_k_w).reshape([b, h, w, w])
        concate = self.softmax(paddle.concat([energy_h, energy_w], axis=3))

        attn_h = concate[:, :, :, 0:h].transpose([0, 2, 1, 3]).reshape([b * w, h, h])
        attn_w = concate[:, :, :, h : h + w].reshape([b * h, w, w])
        out_h = (
            paddle.bmm(proj_v_h, attn_h.transpose([0, 2, 1]))
            .reshape([b, w, -1, h])
            .transpose([0, 2, 3, 1])
        )
        out_w = (
            paddle.bmm(proj_v_w, attn_w.transpose([0, 2, 1]))
            .reshape([b, h, -1, w])
            .transpose([0, 2, 1, 3])
        )
        return self.gamma * (out_h + out_w) + x

    def Inf(self, B, H, W):
        return -paddle.tile(
            paddle.diag(paddle.tile(self.inf_tensor, [H]), 0).unsqueeze(0),
            [B * W, 1, 1],
        )


class Activation(nn.Layer):
    """
    The wrapper of activations.

    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.

    Returns:
        A callable object of Activation.

    Raises:
        KeyError: When parameter `act` is not in the optional range.

    Examples:

        from paddleseg.models.common.activation import Activation

        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>

        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>

        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """

    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.layer.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("nn.layer.activation.{}()".format(act_name))
            else:
                raise KeyError(
                    "{} does not exist in the current {}".format(act, act_dict.keys())
                )

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if (
        paddle.get_device() == "cpu"
        or os.environ.get("PADDLESEG_EXPORT_STAGE")
        or "xpu" in paddle.get_device()
        or "npu" in paddle.get_device()
    ):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNLeakyReLU(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", **kwargs
    ):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

        if "data_format" in kwargs:
            data_format = kwargs["data_format"]
        else:
            data_format = "NCHW"
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = Activation("leakyrelu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvBNReLU(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", **kwargs
    ):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

        if "data_format" in kwargs:
            data_format = kwargs["data_format"]
        else:
            data_format = "NCHW"
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = Activation("relu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(
        self, in_channels, inter_channels, out_channels, dropout_prob=0.1, **kwargs
    ):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs,
        )

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class RCCAModule(nn.Layer):
    def __init__(
        self, in_channels, out_channels, num_classes, dropout_prob=0.1, recurrence=1
    ):
        super().__init__()
        inter_channels = in_channels // 4
        self.recurrence = recurrence
        self.conva = ConvBNLeakyReLU(
            in_channels, inter_channels, 3, padding=1, bias_attr=False
        )
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = ConvBNLeakyReLU(
            inter_channels, inter_channels, 3, padding=1, bias_attr=False
        )
        self.out = AuxLayer(
            in_channels + inter_channels,
            out_channels,
            num_classes,
            dropout_prob=dropout_prob,
        )

    def forward(self, x):
        feat = self.conva(x)
        for i in range(self.recurrence):
            feat = self.cca(feat)
        feat = self.convb(feat)
        output = self.out(paddle.concat([x, feat], axis=1))
        return output


if __name__ == "__main__":

    # test CCA
    x = paddle.rand([32, 64, 256, 256])
    m = CrissCrossAttention(64)
    y = m(x)

    print(y.shape)
