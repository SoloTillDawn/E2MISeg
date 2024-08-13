from torch import nn


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        # 一个可选参数，用于指定每个输出损失的权重因子。如果不提供，则默认所有输出权重因子均为 1。
        self.weight_factors = weight_factors  # [0.57142857 0.28571429 0.14285714]
        self.loss = loss

    def forward(self, x, y):
        # x/y:list{3}=>tensor(2,14,64,128,128)\(2,14,32,32,32)\(2,14,16,16,16)
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors  # [0.57142857 0.28571429 0.14285714]

        # 计算第一个输出与其对应目标之间的损失
        l = weights[0] * self.loss(x[0], y[0])

        # 循环计算后续输出与目标之间的损失，按权重加总
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        # print(l)
        return l
