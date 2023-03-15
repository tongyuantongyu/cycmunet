from mindspore import nn
from mindspore.ops import functional as F


class RMSELoss(nn.LossBase):
    def __init__(self, epsilon=0.001):
        super(RMSELoss, self).__init__()
        self.epsilon = epsilon * epsilon
        self.MSELoss = nn.MSELoss()

    def construct(self, logits, label):
        rmse_loss = F.sqrt(self.MSELoss(logits, label) + self.epsilon)
        return rmse_loss
