from collections import namedtuple
import time

import numpy as np
import mindspore as ms

from model import CycMuNet


def export_rgb(checkpoint, size):
    dummyArg = namedtuple('dummyArg', (
        'nf', 'groups', 'upscale_factor', 'format', 'layers', 'cycle_count', 'batch_mode', 'all_frames',
        'stop_at_conf'))

    args = dummyArg(nf=64, groups=8, upscale_factor=4, format='rgb', layers=3, cycle_count=5, batch_mode='sequence',
                    all_frames=False, stop_at_conf=False)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    print('Init done')
    build_start = time.time()
    model = CycMuNet(args)
    ms.load_checkpoint(checkpoint, model)

    inp = ms.Tensor(np.ones((2, 3, *size), dtype=np.float32))
    inp = inp.astype(ms.float16)
    model = model.to_float(ms.float16)
    model.compile(inp)
    print(f'Load done in {time.time() - build_start}s')
    # verify
    model(inp)

    ms.export(model, inp, file_name=model, file_format='MINDIR')
    print('Export done')


def export_yuv(checkpoint, size):
    dummyArg = namedtuple('dummyArg', (
        'nf', 'groups', 'upscale_factor', 'format', 'layers', 'cycle_count', 'batch_mode', 'all_frames',
        'stop_at_conf'))

    args = dummyArg(nf=64, groups=8, upscale_factor=2, format='yuv420', layers=4, cycle_count=3, batch_mode='sequence',
                    all_frames=False, stop_at_conf=False)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    print('Init done')
    build_start = time.time()
    model = CycMuNet(args)
    ms.load_checkpoint(checkpoint, model)

    inp_y = ms.Tensor(np.zeros((2, 1, *size), dtype=np.float16))
    inp_uv = ms.Tensor(np.zeros((2, 2, size[0] // 2, size[1] // 2), dtype=np.float16))
    model = model.to_float(ms.float16)
    model.compile(inp_y, inp_uv)
    print(f'Load done in {time.time() - build_start}s')
    # verify
    model(inp_y, inp_uv)

    ms.export(model, inp_y, inp_uv, file_name=model, file_format='MINDIR')
    print('Export done')


if __name__ == '__main__':
    # export_rgb('model-files/2x_rgb_base.ckpt', 'model-files/cycmunet_2x_rgb', (64, 64))
    export_yuv('model-files/2x_yuv420_cycle3_layer4.ckpt', 'model-files/cycmunet_2x_yuv420_cycle3_layer4', (1920, 1088))
