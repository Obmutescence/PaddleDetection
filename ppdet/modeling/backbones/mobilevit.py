# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://github.com/BR-IDL/PaddleViT/blob/develop/image_classification/MobileViT/mobilevit.py
# and https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
# reference: https://arxiv.org/abs/2110.02178

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform, TruncatedNormal, Constant
import math
from ppdet.core.workspace import register, serializable

import os
import sys
import os.path as osp
import shutil
import requests
import hashlib
import tarfile
import zipfile
import time
from collections import OrderedDict
from tqdm import tqdm

from ..shape_spec import ShapeSpec

__all__ = ["MobileViT"]
# from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

WEIGHTS_HOME = osp.expanduser("~/.paddleclas/weights")

DOWNLOAD_RETRY_LIMIT = 3


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    # logger.info("File {} md5 checking...".format(fullname))
    print("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        # logger.info(
        #     "File {} md5 check failed, {}(calc) != "
        #     "{}(base)".format(fullname, calc_md5sum, md5sum)
        # )
        print(
            "File {} md5 check failed, {}(calc) != "
            "{}(base)".format(fullname, calc_md5sum, md5sum)
        )
        return False
    return True


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        # logger.info("Downloading {} from {}".format(fname, url))
        print("Downloading {} from {}".format(fname, url))

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            # logger.info(
            #     "Downloading {} from {} failed {} times with exception {}".format(
            #         fname, url, retry_cnt + 1, str(e)
            #     )
            # )
            print(
                "Downloading {} from {} failed {} times with exception {}".format(
                    fname, url, retry_cnt + 1, str(e)
                )
            )
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError(
                "Downloading from {} failed with code "
                "{}!".format(url, req.status_code)
            )

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith("http://") or path.startswith("https://")


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if "/" in file_path:
            file_path = file_path.replace("/", os.sep)
        elif "\\" in file_path:
            file_path = file_path.replace("\\", os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
        return True
    return False


def _uncompress_file_tar(filepath, mode="r:*"):
    files = tarfile.open(filepath, mode)
    file_list = files.getnames()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)

        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _uncompress_file_zip(filepath):
    files = zipfile.ZipFile(filepath, "r")
    file_list = files.namelist()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)
        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    # logger.info("Decompressing {}...".format(fname))
    print("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    return uncompressed_path


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + ".pdparams")):
        raise ValueError(
            "Model pretrain path {}.pdparams does not " "exists.".format(path)
        )
    param_state_dict = paddle.load(path + ".pdparams")
    if isinstance(model, list):
        for m in model:
            if hasattr(m, "set_dict"):
                m.set_dict(param_state_dict)
    else:
        model.set_dict(param_state_dict)
    return


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True, decompress=True):
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    from paddle.distributed import ParallelEnv

    assert is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)
    # Mainly used to solve the problem of downloading data from different
    # machines in the case of multiple machines. Different nodes will download
    # data, and the same node will only download data once.
    rank_id_curr_node = int(os.environ.get("PADDLE_RANK_IN_NODE", 0))

    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        # logger.info("Found {}".format(fullpath))
        print("Found {}".format(fullpath))
    else:
        if rank_id_curr_node == 0:
            fullpath = _download(url, root_dir, md5sum)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)

    if rank_id_curr_node == 0:
        if decompress and (
            tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath)
        ):
            fullpath = _decompress(fullpath)

    return fullpath


def get_weights_path_from_url(url, md5sum=None):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.

    Args:
        url (str): download url
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded weights.

    Examples:
        .. code-block:: python

            from paddle.utils.download import get_weights_path_from_url

            resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
            local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)

    """
    path = get_path_from_url(url, WEIGHTS_HOME, md5sum)
    return path


def load_dygraph_pretrain_from_url(model, pretrained_url, use_ssld=False):
    if use_ssld:
        pretrained_url = pretrained_url.replace("_pretrained", "_ssld_pretrained")
    local_weight_path = get_weights_path_from_url(pretrained_url).replace(
        ".pdparams", ""
    )
    load_dygraph_pretrain(model, path=local_weight_path)
    return


MODEL_URLS = {
    "MobileViT_XXS": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViT_XXS_pretrained.pdparams",
    "MobileViT_XS": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViT_XS_pretrained.pdparams",
    "MobileViT_S": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViT_S_pretrained.pdparams",
}


def _init_weights_linear():
    weight_attr = ParamAttr(initializer=TruncatedNormal(std=0.02))
    bias_attr = ParamAttr(initializer=Constant(0.0))
    return weight_attr, bias_attr


def _init_weights_layernorm():
    weight_attr = ParamAttr(initializer=Constant(1.0))
    bias_attr = ParamAttr(initializer=Constant(0.0))
    return weight_attr, bias_attr


class ConvBnAct(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_attr=False,
        groups=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingUniform()),
            bias_attr=bias_attr,
        )
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = nn.Silu()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class Identity(nn.Layer):
    """Identity layer"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.1):
        super().__init__()
        w_attr_1, b_attr_1 = _init_weights_linear()
        self.fc1 = nn.Linear(
            embed_dim,
            int(embed_dim * mlp_ratio),
            weight_attr=w_attr_1,
            bias_attr=b_attr_1,
        )

        w_attr_2, b_attr_2 = _init_weights_linear()
        self.fc2 = nn.Linear(
            int(embed_dim * mlp_ratio),
            embed_dim,
            weight_attr=w_attr_2,
            bias_attr=b_attr_2,
        )

        self.act = nn.Silu()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class Attention(nn.Layer):
    def __init__(
        self, embed_dim, num_heads, qkv_bias=True, dropout=0.1, attention_dropout=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = int(embed_dim / self.num_heads)
        self.all_head_dim = self.attn_head_dim * self.num_heads

        w_attr_1, b_attr_1 = _init_weights_linear()
        self.qkv = nn.Linear(
            embed_dim,
            self.all_head_dim * 3,
            weight_attr=w_attr_1,
            bias_attr=b_attr_1 if qkv_bias else False,
        )

        self.scales = self.attn_head_dim**-0.5

        w_attr_2, b_attr_2 = _init_weights_linear()
        self.proj = nn.Linear(
            embed_dim, embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2
        )

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        B, P, N, d = x.shape
        x = x.reshape([B, P, N, self.num_heads, d // self.num_heads])
        x = x.transpose([0, 1, 3, 2, 4])
        return x

    def forward(self, x):
        b_sz, n_patches, in_channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            [b_sz, n_patches, 3, self.num_heads, qkv.shape[-1] // self.num_heads // 3]
        )
        qkv = qkv.transpose([0, 3, 2, 1, 4])
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        query = query * self.scales
        key = key.transpose([0, 1, 3, 2])
        # QK^T
        attn = paddle.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        # weighted sum
        out = paddle.matmul(attn, value)
        out = out.transpose([0, 2, 1, 3]).reshape(
            [b_sz, n_patches, out.shape[1] * out.shape[3]]
        )
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class EncoderLayer(nn.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads=4,
        qkv_bias=True,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_dropout=0.0,
        droppath=0.0,
    ):
        super().__init__()
        w_attr_1, b_attr_1 = _init_weights_layernorm()
        w_attr_2, b_attr_2 = _init_weights_layernorm()

        self.attn_norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1
        )
        self.attn = Attention(
            embed_dim, num_heads, qkv_bias, dropout, attention_dropout
        )
        self.drop_path = DropPath(droppath) if droppath > 0.0 else Identity()
        self.mlp_norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2
        )
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class Transformer(nn.Layer):
    """Transformer block for MobileViTBlock"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        depth,
        qkv_bias=True,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_dropout=0.0,
        droppath=0.0,
    ):
        super().__init__()
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(
                EncoderLayer(
                    embed_dim,
                    num_heads,
                    qkv_bias,
                    mlp_ratio,
                    dropout,
                    attention_dropout,
                    droppath,
                )
            )
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = _init_weights_layernorm()
        self.norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.norm(x)
        return out


class MobileV2Block(nn.Layer):
    """Mobilenet v2 InvertedResidual block"""

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            layers.append(ConvBnAct(inp, hidden_dim, kernel_size=1))

        layers.extend(
            [
                # dw
                ConvBnAct(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, padding=1
                ),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(oup),
            ]
        )

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileViTBlock(nn.Layer):
    """MobileViTBlock for MobileViT"""

    def __init__(
        self,
        dim,
        hidden_dim,
        depth,
        num_heads=4,
        qkv_bias=True,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_dropout=0.0,
        droppath=0.0,
        patch_size=(2, 2),
    ):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # local representations
        self.conv1 = ConvBnAct(dim, dim, padding=1)
        self.conv2 = nn.Conv2D(
            dim, hidden_dim, kernel_size=1, stride=1, bias_attr=False
        )
        # global representations
        self.transformer = Transformer(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            droppath=droppath,
        )

        # fusion
        self.conv3 = ConvBnAct(hidden_dim, dim, kernel_size=1)
        self.conv4 = ConvBnAct(2 * dim, dim, padding=1)

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.conv2(x)

        patch_h = self.patch_h
        patch_w = self.patch_w
        patch_area = int(patch_w * patch_h)
        _, in_channels, orig_h, orig_w = x.shape
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = False

        if new_w != orig_w or new_h != orig_h:
            x = F.interpolate(x, size=[new_h, new_w], mode="bilinear")
            interpolate = True

        num_patch_w, num_patch_h = new_w // patch_w, new_h // patch_h
        num_patches = num_patch_h * num_patch_w
        reshaped_x = x.reshape([-1, patch_h, num_patch_w, patch_w])
        transposed_x = reshaped_x.transpose([0, 2, 1, 3])
        reshaped_x = transposed_x.reshape([-1, in_channels, num_patches, patch_area])
        transposed_x = reshaped_x.transpose([0, 3, 2, 1])

        x = transposed_x.reshape([-1, num_patches, in_channels])
        x = self.transformer(x)
        x = x.reshape([-1, patch_h * patch_w, num_patches, in_channels])

        _, pixels, num_patches, channels = x.shape
        x = x.transpose([0, 3, 2, 1])
        x = x.reshape([-1, num_patch_w, patch_h, patch_w])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([-1, channels, num_patch_h * patch_h, num_patch_w * patch_w])

        if interpolate:
            x = F.interpolate(x, size=[orig_h, orig_w])
        x = self.conv3(x)
        x = paddle.concat((h, x), axis=1)
        x = self.conv4(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


@register
class MobileViTV1(nn.Layer):
    """MobileViT
    A PaddlePaddle impl of : `MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer`  -
      https://arxiv.org/abs/2110.02178
    """

    _out_channels_dict = {
        "MobileViT_XS": [48, 64, 80, 96],
        "MobileViT_XXS": [48, 64, 80, 96],
        "MobileViT_S": [48, 64, 80, 96],
    }

    _arch_config = dict(
        # Small
        MobileViT_XXS=dict(
            in_channels=3,
            dims=[16, 16, 24, 24, 24, 48, 64, 80, 320],
            hidden_dims=[64, 80, 96],
            mv2_expansion=2,
        ),
        # Extra Small
        MobileViT_XS=dict(
            in_channels=3,
            dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],
            hidden_dims=[96, 120, 144],
            mv2_expansion=4,
        ),
        # Extra Extra Small
        MobileViT_S=dict(
            in_channels=3,
            dims=[16, 32, 64, 64, 64, 96, 128, 160, 640],
            hidden_dims=[144, 192, 240],
            mv2_expansion=4,
        ),
        null=dict(),
    )

    def __init__(
        self,
        in_channels=3,
        dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],
        hidden_dims=[96, 120, 144],
        mv2_expansion=4,
        pretrained=False,
        use_ssld=False,
        arch_name=None,
    ):
        super().__init__()

        if arch_name in self._arch_config:
            in_channels = self._arch_config[arch_name]["in_channels"]
            dims = self._arch_config[arch_name]["dims"]
            hidden_dims = self._arch_config[arch_name]["hidden_dims"]
            mv2_expansion = self._arch_config[arch_name]["mv2_expansion"]

        self.conv3x3 = ConvBnAct(
            in_channels, dims[0], kernel_size=3, stride=2, padding=1
        )
        self.mv2_block_1 = MobileV2Block(dims[0], dims[1], expansion=mv2_expansion)
        self.mv2_block_2 = MobileV2Block(
            dims[1], dims[2], stride=2, expansion=mv2_expansion
        )
        self.mv2_block_3 = MobileV2Block(dims[2], dims[3], expansion=mv2_expansion)
        self.mv2_block_4 = MobileV2Block(dims[3], dims[4], expansion=mv2_expansion)

        self.mv2_block_5 = MobileV2Block(
            dims[4], dims[5], stride=2, expansion=mv2_expansion
        )
        self.mvit_block_1 = MobileViTBlock(dims[5], hidden_dims[0], depth=2)

        self.mv2_block_6 = MobileV2Block(
            dims[5], dims[6], stride=2, expansion=mv2_expansion
        )
        self.mvit_block_2 = MobileViTBlock(dims[6], hidden_dims[1], depth=4)

        self.mv2_block_7 = MobileV2Block(
            dims[6], dims[7], stride=2, expansion=mv2_expansion
        )
        self.mvit_block_3 = MobileViTBlock(dims[7], hidden_dims[2], depth=3)
        self.conv1x1 = ConvBnAct(dims[7], dims[8], kernel_size=1)

        if arch_name is not None:
            self._load_pretrained(pretrained, MODEL_URLS[arch_name], use_ssld=use_ssld)

        self._out_channels = self._out_channels_dict[arch_name]

    def forward(self, inputs):

        return_list = []

        x = self.conv3x3(inputs["image"])  # 1/2
        x = self.mv2_block_1(x)
        x = self.mv2_block_2(x)
        x = self.mv2_block_3(x)  # 1/2
        x = self.mv2_block_4(x)  # 128
        return_list.append(x)

        x = self.mv2_block_5(x)  # 64
        x = self.mvit_block_1(x)
        return_list.append(x)

        x = self.mv2_block_6(x)  # 32
        x = self.mvit_block_2(x)
        return_list.append(x)

        x = self.mv2_block_7(x)  # 16
        x = self.mvit_block_3(x)
        return_list.append(x)

        x = self.conv1x1(x)  # [16, 96, 16, 16]

        return return_list

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]

    def _load_pretrained(self, pretrained, model_url, use_ssld=False):
        if pretrained is False:
            pass
        elif pretrained is True:
            load_dygraph_pretrain_from_url(self, model_url, use_ssld=use_ssld)
        elif isinstance(pretrained, str):
            load_dygraph_pretrain(self, pretrained)
        else:
            raise RuntimeError(
                "pretrained type is not available. Please use `string` or `boolean` type."
            )


def MobileViT_XXS(pretrained=False, use_ssld=False, **kwargs):
    model = MobileViT(
        in_channels=3,
        dims=[16, 16, 24, 24, 24, 48, 64, 80, 320],
        hidden_dims=[64, 80, 96],
        mv2_expansion=2,
        **kwargs
    )

    _load_pretrained(pretrained, model, MODEL_URLS["MobileViT_XXS"], use_ssld=use_ssld)
    return model


def MobileViT_XS(pretrained=False, use_ssld=False, **kwargs):
    model = MobileViT(
        in_channels=3,
        dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],
        hidden_dims=[96, 120, 144],
        mv2_expansion=4,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["MobileViT_XS"], use_ssld=use_ssld)
    return model


def MobileViT_S(pretrained=False, use_ssld=False, **kwargs):
    model = MobileViT(
        in_channels=3,
        dims=[16, 32, 64, 64, 64, 96, 128, 160, 640],
        hidden_dims=[144, 192, 240],
        mv2_expansion=4,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["MobileViT_S"], use_ssld=use_ssld)
    return model
