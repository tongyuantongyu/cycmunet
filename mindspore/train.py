import os
import time
from collections import namedtuple
import argparse

import numpy as np
import tqdm
import mindspore as ms
from mindspore import nn, ops

from model import CycMuNet, TrainModel
from util.rmse import RMSELoss
from util.normalize import Normalizer
from dataset.video import VideoFrameDataset

print("Initialized.")

dummyArg = namedtuple('dummyArg', (
    'nf', 'groups', 'upscale_factor', 'format', 'layers', 'cycle_count', 'batch_mode', 'all_frames', 'stop_at_conf'))

size = 128
args = dummyArg(nf=64, groups=8, upscale_factor=2, format='yuv420', layers=4, cycle_count=3, batch_mode='batch',
                all_frames=True, stop_at_conf=False)

epochs = 1
batch_size = 1
learning_rate = 0.001
save_prefix = 'monitor'

ds_path = r"/home/ma-user/work/cctv-scaled/"
pretrained = "/home/ma-user/work/cycmunet-ms/model-files/2x_yuv420_cycle3_layer4.ckpt"
save_path = "/home/ma-user/work/cycmunet-ms/checkpoints/"

# ds_path = r"./test-files/index-train.txt"
# pretrained = "./model-files/2x_yuv420_cycle3_layer4.ckpt"
# save_path = "./checkpoints/"

# ds_path = r"D:\Python\cycmunet-ms\test-files/"
# pretrained = ""

# parser = argparse.ArgumentParser(
#     prog='CycMuNet+ MindSpore Training')
#
# parser.add_argument('--dataset')
# parser.add_argument('--pretrained')
# parser.add_argument('--save')
#
# cmd_args = parser.parse_args()
#
# ds_path = cmd_args.dataset
# pretrained = cmd_args.pretrained
# save_path = cmd_args.save


save_prefix = f'{save_prefix}_{args.upscale_factor}x_l{args.layers}_c{args.cycle_count}'

network = CycMuNet(args)
if pretrained:
    ms.load_checkpoint(pretrained, network)


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

inp = (ms.Tensor(np.zeros((batch_size, 1, size, size), dtype=np.float32)),
       ms.Tensor(np.zeros((batch_size, 2, size // 2, size // 2), dtype=np.float32)))
network.compile(inp, inp)
inp = None


loss_fn = RMSELoss()

nm = Normalizer()
ds_train = VideoFrameDataset(ds_path + "index-train.txt", size, args.upscale_factor, True, nm)
ds_test = VideoFrameDataset(ds_path + "index-test.txt", size, args.upscale_factor, True, nm)

ds_train = ds_train.batch(batch_size)
ds_test = ds_test.batch(batch_size)

scheduler = nn.CosineDecayLR(min_lr=1e-7, max_lr=learning_rate, decay_steps=640000)
optimizer = nn.AdaMax(network.trainable_params(), learning_rate=learning_rate)


model = TrainModel(network, loss_fn)
model = ms.Model(model, optimizer=optimizer, eval_network=model, boost_level="O1")


def save_model(epoch):
    if epoch == -1:
        name = "snapshot"
    else:
        name = f"epoch_{epoch}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_path = save_path + f"{save_prefix}_{name}.ckpt"
    ms.save_checkpoint(network, str(output_path))
    print(f"Checkpoint saved to {output_path}")


print("Start train.")

profiler = ms.Profiler(output_path='./profiler_data')

for t in range(1, epochs + 1):
    try:
        print(f"Epoch {t}\n-------------------------------")
        model.train(t, ds_train, dataset_sink_mode=True)
        save_model(t)
    except KeyboardInterrupt:
        save_model(-1)

profiler.analyse()

print("Done.")
