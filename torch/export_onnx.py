import os
import pathlib
from collections import namedtuple

import torch
from torch.onnx import symbolic_helper
import onnx
import onnx.shape_inference
import onnxsim
import onnx_graphsurgeon as gs  # Can be installed from TensorRT SDK

from model import CycMuNet, use_fold_catconv

dummyArg = namedtuple('dummyArg', ('nf', 'groups', 'upscale_factor', 'format', 'layers', 'cycle_count'))

args = dummyArg(nf=64, groups=8, upscale_factor=2, format='yuv420', layers=4, cycle_count=3)
size = 64
checkpoint_file = 'checkpoints/monitor-ugly-sparsity_2x_l4_c3_epoch_2.pth'
output_path = 'onnx/monitor_omni'


size_in = (size, size)
size_out = tuple(i * args.upscale_factor for i in size_in)
output_path = pathlib.Path(output_path)
config_string = f"_{args.upscale_factor}x_l{args.layers}"
if args.format == 'yuv420':
    size_uv_in = tuple(i // 2 for i in size_in)
    config_string += '_yuv1-1'

fe_onnx = str(output_path / f'fe{config_string}.onnx')
ff_onnx = str(output_path / f'ff{config_string}.onnx')
os.makedirs(output_path, exist_ok=True)


# Placeholder to export DeformConv
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def symbolic_deform_conv2d_forward(g,
                                   input,
                                   weight,
                                   offset,
                                   mask,
                                   bias,
                                   stride_h,
                                   stride_w,
                                   pad_h,
                                   pad_w,
                                   dil_h,
                                   dil_w,
                                   n_weight_grps,
                                   n_offset_grps,
                                   use_mask):
    if n_weight_grps != 1 or not use_mask:
        raise NotImplementedError()
    return g.op("custom::DeformConv2d", input, offset, mask, weight, bias, stride_i=[stride_h, stride_w],
                padding_i=[pad_h, pad_w], dilation_i=[dil_h, dil_w], deformable_groups_i=n_offset_grps)


# Register custom symbolic function
torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", symbolic_deform_conv2d_forward, 13)


def clean_fp16_subnormal(t: torch.Tensor):
    threshold = 0.00006103515625
    mask = torch.logical_and(t > -threshold, t < threshold)
    t[mask] = 0
    return t


def as_module(func):
    class mod(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *x):
            return func(*x)

    return mod().eval()


def simplify(name):
    model, other = onnxsim.simplify(name)
    graph = gs.import_onnx(model)
    graph.fold_constants().cleanup()

    for n in graph.nodes:
        if n.op == 'DeformConv2d':
            if n.outputs[0].outputs[0].op == 'LeakyRelu':
                lrelu = n.outputs[0].outputs[0]
                n.attrs['activation_type'] = 3
                n.attrs['alpha'] = lrelu.attrs['alpha']
                n.attrs['beta'] = 0.0
                n.outputs = lrelu.outputs
                lrelu.inputs = []
                lrelu.outputs = []
            else:
                n.attrs['activation_type'] = -1
                n.attrs['alpha'] = 0.0
                n.attrs['beta'] = 0.0

    graph.cleanup().toposort()

    model = gs.export_onnx(graph)
    onnx.save_model(model, name)
    print(f'Simplify {name} done')


use_fold_catconv()
model = CycMuNet(args)
state_dict = torch.load(checkpoint_file, map_location='cpu')
state_dict = {k: clean_fp16_subnormal(v) for k, v in state_dict.items() if '__weight_mma_mask' not in k}
model.load_state_dict(state_dict)
model = model.eval()
for v in model.parameters(recurse=True):
    v.requires_grad = False


if __name__ == '__main__':
    with torch.no_grad():
        print("Exporting fe...")

        if args.format == 'rgb':
            fe_i = torch.zeros((2, 3, *size))
            dynamic_axes = {
                "x": {
                    0: "batch_size",
                    2: "input_height",
                    3: "input_width"
                },
            }
        elif args.format == 'yuv420':
            fe_i = tuple([torch.zeros((2, 1, *size_in)), torch.zeros((2, 2, *size_uv_in))])
            dynamic_axes = {
                "y": {
                    0: "batch_size",
                    2: "input_height",
                    3: "input_width"
                },
                "uv": {
                    0: "batch_size",
                    2: "input_height_uv",
                    3: "input_width_uv"
                },
            }
        else:
            raise NotImplementedError()
        input_names = list(dynamic_axes.keys())
        output_names = [f'l{i}' for i in range(args.layers)[::-1]]
        dynamic_axes.update({f'l{i}': {
            0: "batch_size",
            2: f"feature_height_{i}",
            3: f"feature_width_{i}"
        } for i in range(args.layers)[::-1]})

        @as_module
        def fe(*x_or_y_uv: torch.Tensor):
            if args.format == 'rgb':
                return model.head_fe(x_or_y_uv[0])
            else:
                return model.head_fe(x_or_y_uv)


        torch.onnx.export(fe, fe_i, fe_onnx, opset_version=13,
                          export_params=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print("Exporting ff...")

        ff_i = []
        input_axes = dict()
        cur_size = size_in
        for i in range(args.layers)[::-1]:
            ff_i.insert(0, torch.zeros(1, args.nf, *cur_size))
            cur_size = tuple(i // 2 for i in cur_size)

        for i in range(args.layers):
            axes = {
                0: "batch_size",
                2: f"feature_height_{i}",
                3: f"feature_width_{i}"
            }
            input_axes[f'f0l{i}'] = axes
            input_axes[f'f2l{i}'] = axes
        input_names = [f'f0l{i}' for i in range(args.layers)[::-1]] + [f'f2l{i}' for i in range(args.layers)[::-1]]
        output_names = ['f1']
        dynamic_axes = dict(input_axes)
        dynamic_axes[f'f1'] = {
            0: "batch_size",
            2: f"feature_height_{args.layers - 1}",
            3: f"feature_width_{args.layers - 1}"
        }

        if args.format == 'rgb':
            output_axes = {
                "h0": {
                    0: "batch_size",
                    2: "output_height",
                    3: "output_width"
                },
                "h1": {
                    0: "batch_size",
                    2: "output_height",
                    3: "output_width"
                },
            }
        elif args.format == 'yuv420':
            output_axes = {
                "h0_y": {
                    0: "batch_size",
                    2: "output_height",
                    3: "output_width"
                },
                "h0_uv": {
                    0: "batch_size",
                    2: "output_height_uv",
                    3: "output_width_uv"
                },
                "h1_y": {
                    0: "batch_size",
                    2: "output_height",
                    3: "output_width"
                },
                "h1_uv": {
                    0: "batch_size",
                    2: "output_height_uv",
                    3: "output_width_uv"
                },
            }
        else:
            raise NotImplementedError()
        output_names = list(output_axes.keys())
        dynamic_axes = dict(input_axes)
        dynamic_axes.update(output_axes)

        @as_module
        def ff(input1, input2):
            fea = [input1[-1], model.ff(input1, input2)[0], input2[-1]]
            outs = model.mu_fr_tail(fea, all_frames=False)
            return outs


        torch.onnx.export(ff, (ff_i, ff_i),
                          str(output_path / f'ff{config_string}.onnx'), opset_version=13,
                          export_params=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

    simplify(fe_onnx)
    simplify(ff_onnx)
