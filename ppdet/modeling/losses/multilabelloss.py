# https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/ppcls/loss/multilabelloss.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

__all__ = ["MultiLabelLoss"]


def ratio2weight(targets, ratio):
    pos_weights = targets * (1.0 - ratio)
    neg_weights = (1.0 - targets) * ratio
    weights = paddle.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights = weights - weights * (targets > 1)

    return weights


@register
@serializable
class MultiLabelLoss(nn.Layer):
    """
    Multi-label loss
    """

    def __init__(
        self,
        epsilon=None,
        size_sum=False,
        weight_ratio=False,
        class_num=80,
        reduction="mean",
    ):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        self.weight_ratio = weight_ratio
        self.size_sum = size_sum
        self.class_num = class_num
        self.reduction = reduction

    def _labelsmoothing(self, target, class_num):
        if target.ndim == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def _binary_crossentropy(self, input, target, class_num):
        if self.weight_ratio:
            target, label_ratio = target[:, 0, :], target[:, 1, :]
        if self.epsilon is not None:
            target = self._labelsmoothing(target, class_num)
        cost = F.binary_cross_entropy_with_logits(
            logit=input, label=target, reduction=self.reduction
        )

        if self.weight_ratio:
            targets_mask = paddle.cast(target > 0.5, "float32")
            weight = ratio2weight(targets_mask, paddle.to_tensor(label_ratio))
            weight = weight * (target > -1)
            cost = cost * weight

        if self.size_sum:
            cost = cost.sum(1).mean() if self.size_sum else cost.mean()

        return cost

    def forward(self, x, target):
        if isinstance(x, dict):
            x = x["logits"]
        # class_num = x.shape[-1]
        loss = self._binary_crossentropy(x, target, self.class_num)
        # return {"MultiLabelLoss": loss}
        return loss
