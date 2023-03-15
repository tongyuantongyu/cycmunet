from mindspore import nn


class TrainModel(nn.Cell):
    def __init__(self, network, loss):
        super().__init__()
        self.net = network
        self.loss = loss

    def loss_frame(self, expected, actual):
        return self.loss(expected[0], actual[0]) + self.loss(expected[1], actual[1])

    def construct(self, *data):
        inputs = [data[6:8], data[10:12]]
        expected = [data[0:2], data[2:4], data[4:6], data[8:10]]
        actual = self.net(*inputs)
        loss = self.loss_frame(expected[0], actual[0]) * 0.5 + \
               self.loss_frame(expected[1], actual[1]) + \
               self.loss_frame(expected[2], actual[2]) * 0.5 + \
               self.loss_frame(expected[3], actual[3])
        return loss
