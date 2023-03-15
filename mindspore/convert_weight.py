import re
from collections import namedtuple

import mindspore as ms
from mindspore import ops
import torch

import model


def transform_dcnpack(weights):
    result = {
        'dcn_weight': weights['dcn.weight'],
        'dcn_bias': weights['dcn.bias'],
        'conv_mask.weight': weights['conv_mask.weight'],
        'conv_mask.bias': weights['conv_mask.bias'],
    }

    w = weights['conv_offset.weight'].reshape(72, 2, 64, 3, 3)
    b = weights['conv_offset.bias'].reshape(72, 2)
    w = w[:, ::-1, ...].transpose(1, 0, 2, 3, 4).reshape(144, 64, 3, 3)
    b = b[:, ::-1, ...].transpose(1, 0).reshape(144)

    result['conv_offset.weight'] = w
    result['conv_offset.bias'] = b
    return result


if __name__ == '__main__':
    torch_source = 'checkpoints/2x_cycle3_yuv420_sparsity_epoch_20.pth'
    ms_normal = 'checkpoints/2x_yuv420_cycle3_layer4.ckpt'
    ms_sep = 'checkpoints/2x_sep_yuv420_cycle3_layer4.ckpt'

    dummyArg = namedtuple('dummyArg', (
        'nf', 'groups', 'upscale_factor', 'format', 'layers', 'cycle_count', 'batch_mode', 'all_frames',
        'stop_at_conf'))

    size = (64, 64)
    args = dummyArg(nf=64, groups=8, upscale_factor=2, format='yuv420', layers=4, cycle_count=3, batch_mode='sequence',
                    all_frames=False, stop_at_conf=False)

    print('Init done')

    rewrite_names = {
        ".Pro_align.conv1x1.": 3,
        ".Pro_align.conv1_3x3.": 2,
        ".offset_conv1.": 2,
        ".offset_conv2.": 2,
        ".fea_conv.": 2,
        "ff.fusion.": 2,
        "mu.conv.": 2 * args.cycle_count + 1
    }

    rewrite_names_re = {
        r"(merge1?\.(\d+)\.)": lambda match: int(match[2]) + 1,
    }

    # normal model
    model_normal = model.CycMuNet(args)

    source = torch.load(torch_source, map_location=torch.device('cpu'))
    source = {k: v for k, v in source.items() if '__weight_mma_mask' not in k}
    template = model_normal.parameters_dict()

    dest = dict()
    pending_dcn = dict()
    for k, v in source.items():
        if '.dcnpack.' in k:
            module, name = k.split('.dcnpack.')
            if module in pending_dcn:
                pending_dcn[module][name] = v.numpy()
            else:
                pending_dcn[module] = {name: v.numpy()}
            continue

        for name in rewrite_names:
            k = k.replace(name, name + 'conv.')
        for re_name in rewrite_names_re:
            k = re.sub(re_name, "\\1conv.", k)
        if k in template:
            dest[k] = ms.Parameter(v.numpy())
        else:
            print(f"Unknown parameter {k} ignored.")

    for m, ws in pending_dcn.items():
        for name, w in transform_dcnpack(ws).items():
            dest[f'{m}.dcnpack.{name}'] = ms.Parameter(w)

    print(ms.load_param_into_net(model_normal, dest, strict_load=True))
    ms.save_checkpoint(model_normal, ms_normal)

    print('Done normal model')

    # sep model: concat + conv is separated to multiple conv + add, to reduce memory footprint
    model_separate = model.cycmunet_sep(args)
    template = model_separate.parameters_dict()

    dest = dict()
    pending_dcn = dict()

    def filter_catconv(k, tensor):
        for name, n in rewrite_names.items():
            if name in k:
                t = ms.Parameter(tensor.numpy())
                if k.endswith('.weight'):
                    dest.update({k.replace(name, f'{name}convs.{i}.'): ms.Parameter(v) for i, v in
                                 enumerate(ops.split(t, axis=1, output_num=n))})
                elif k.endswith('.bias'):
                    dest[k.replace(name, name + 'convs.0.')] = t
                return True
        for name, get_n in rewrite_names_re.items():
            search_result = re.search(name, k)
            if not search_result:
                continue
            n = get_n(search_result)
            t = ms.Parameter(tensor.numpy())
            if k.endswith('.weight'):
                dest.update({re.sub(name, f'\\1convs.{i}.', k): ms.Parameter(v) for i, v in
                             enumerate(ops.split(t, axis=1, output_num=n))})
            elif k.endswith('.bias'):
                dest[re.sub(name, f'\\1convs.0.', k)] = t
            return True
        return False

    for k, v in source.items():
        if '.dcnpack.' in k:
            module, name = k.split('.dcnpack.')
            if module in pending_dcn:
                pending_dcn[module][name] = v.numpy()
            else:
                pending_dcn[module] = {name: v.numpy()}
            continue

        if filter_catconv(k, v):
            continue
        if k in template:
            dest[k] = ms.Parameter(v.numpy())
        else:
            print(f"Unknown parameter {k} ignored.")

    for m, ws in pending_dcn.items():
        for name, w in transform_dcnpack(ws).items():
            dest[f'{m}.dcnpack.{name}'] = ms.Parameter(w)

    print(ms.load_param_into_net(model_separate, dest, strict_load=True))
    ms.save_checkpoint(model_separate, ms_sep)

    print('Done separate model')
