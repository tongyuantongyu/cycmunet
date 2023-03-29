import math
import logging
import os
import time

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cuda
import torch.backends.cudnn
import torchvision.utils
from pytorch_msssim import SSIM

from model import CycMuNet
from model.util import converter, normalizer
import dataset
from cycmunet.model import model_arg
from cycmunet.run import test_arg

model_args = model_arg(nf=64,
                       groups=8,
                       upscale_factor=2,
                       format='yuv420',
                       layers=4,
                       cycle_count=3
                       )

test_args = test_arg(
    size=(256, 256),
    checkpoint='checkpoints/monitor-ugly-sparsity_2x_l4_c3_epoch_2.pth',
    dataset_indexes=[
        "/root/videos/cctv-scaled/index-test-good.txt",
        "/root/videos/cctv-scaled/index-test-ugly.txt",
        "/root/videos/cctv-scaled/index-test-smooth.txt",
        "/root/videos/cctv-scaled/index-test-sharp.txt",
    ],
    preview_interval=100,
    seed=0,
    batch_size=4,
    fp16=True,
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

force_data_dtype = torch.float16 if test_args.fp16 else None

# --------------------------------------
# Start of code

preview_interval = 100 \
    if (len(test_args.dataset_indexes) == 1 or math.gcd(100, len(test_args.dataset_indexes)) == 1) \
    else 101

nrow = 1 if test_args.size[0] * 9 > test_args.size[1] * 16 else 3

torch.manual_seed(test_args.seed)
torch.cuda.manual_seed(test_args.seed)

formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

logger = logging.getLogger('test_progress')
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

logger_init = logging.getLogger('initialization')
logger_init.addHandler(ch)
logger_init.setLevel(logging.DEBUG)

cvt = converter()
norm = normalizer()

dataset_types = {
    'triplet': dataset.ImageSequenceDataset,
    'video': dataset.VideoFrameDataset
}
Dataset = dataset_types[test_args.dataset_type]
if len(test_args.dataset_indexes) == 1:
    ds_test = Dataset(test_args.dataset_indexes[0],
                      test_args.size,
                      model_args.upscale_factor,
                      augment=True,
                      seed=test_args.seed)
else:
    ds_test = dataset.InterleavedDataset(*[
        Dataset(dataset_index,
                test_args.size,
                model_args.upscale_factor,
                augment=True,
                seed=test_args.seed + i)
        for i, dataset_index in enumerate(test_args.dataset_indexes)])
ds_test = DataLoader(ds_test,
                     num_workers=1,
                     batch_size=test_args.batch_size,
                     shuffle=Dataset.want_shuffle,  # Video dataset friendly
                     drop_last=True)

model = CycMuNet(model_args)
model.eval()
num_params = 0
for param in model.parameters():
    num_params += param.numel()
logger_init.info(f"Model has {num_params} parameters.")

if not os.path.exists(test_args.checkpoint):
    logger_init.error(f"Checkpoint weight {test_args.checkpoint} not exist.")
    exit(1)
state_dict = torch.load(test_args.checkpoint, map_location=lambda storage, loc: storage)
load_result = model.load_state_dict(state_dict, strict=False)
if load_result.unexpected_keys:
    logger_init.warning(f"Unknown parameters ignored: {load_result.unexpected_keys}")
if load_result.missing_keys:
    logger_init.warning(f"Missing parameters not initialized: {load_result.missing_keys}")
logger_init.info("Checkpoint loaded.")

model = model.cuda()
if force_data_dtype:
    model = model.to(force_data_dtype)

epsilon = (1 / 255) ** 2


def rmse(a, b):
    return torch.mean(torch.sqrt((a - b) ** 2 + epsilon))


ssim_module = SSIM(data_range=1.0, nonnegative_ssim=True).cuda()


def ssim(a, b):
    return 1 - ssim_module(a, b)


def recursive_cuda(li, force_data_dtype):
    if isinstance(li, (list, tuple)):
        return tuple(recursive_cuda(i, force_data_dtype) for i in li)
    else:
        if force_data_dtype is not None:
            return li.cuda().to(force_data_dtype)
        else:
            return li.cuda()


if __name__ == '__main__':
    with torch.no_grad():
        total_loss = [0.0] * 4
        total_iter = len(ds_test)
        with tqdm.tqdm(total=total_iter, desc=f"Test") as progress:
            for it, data in enumerate(ds_test):
                (hf0, hf1, hf2), (lf0, lf1, lf2) = recursive_cuda(data, force_data_dtype)
                if Dataset.pix_type == 'yuv':
                    target = [cvt.yuv2rgb(*inp) for inp in (hf0, hf1, hf2, lf1)]
                else:
                    target = [hf0, hf1, hf2, lf1]

                if it % preview_interval == 0:
                    if Dataset.pix_type == 'yuv':
                        org = [F.interpolate(cvt.yuv2rgb(y[0:1], uv[0:1]),
                                             scale_factor=(model_args.upscale_factor, model_args.upscale_factor),
                                             mode='nearest').detach().float().cpu()
                               for y, uv in (lf0, lf1, lf2)]
                    else:
                        org = [F.interpolate(lf[0:1],
                                             scale_factor=(model_args.upscale_factor, model_args.upscale_factor),
                                             mode='nearest').detach().float().cpu()
                               for lf in (lf0, lf1, lf2)]

                if Dataset.pix_type == 'rgb':
                    lf0, lf2 = cvt.rgb2yuv(lf0), cvt.rgb2yuv(lf2)

                t0 = time.perf_counter()
                lf0, lf2 = norm.normalize_yuv_420(*lf0), norm.normalize_yuv_420(*lf2)
                outs = model(lf0, lf2, batch_mode='batch')

                t1 = time.perf_counter()
                t_forward = t1 - t0
                actual = [cvt.yuv2rgb(*norm.denormalize_yuv_420(*out)).float() for out in outs]

                if it % preview_interval == 0:
                    out = [i[0:1].detach().float().cpu() for i in actual[:3]]
                    ref = [i[0:1].detach().float().cpu() for i in target[:3]]

                    for idx, ts in enumerate(zip(org, out, ref)):
                        torchvision.utils.save_image(torch.concat(ts), f"./result/out{idx}.png",
                                                     value_range=(0, 1), nrow=nrow, padding=0)

                rmse_loss = [rmse(a, t).item() for a, t in zip(actual, target)]
                ssim_loss = [ssim(a, t).item() for a, t in zip(actual, target)]

                t2 = time.perf_counter()
                t_loss = t2 - t1

                rmse_h = sum(rmse_loss[:3]) / 3
                rmse_l = rmse_loss[3]
                ssim_h = sum(ssim_loss[:3]) / 3
                ssim_l = ssim_loss[3]

                total_loss[0] += rmse_h
                total_loss[1] += rmse_l
                total_loss[2] += ssim_h
                total_loss[3] += ssim_l

                progress.set_postfix(ordered_dict={
                    "rmse_h": f"{rmse_h:.4f}",
                    "rmse_l": f"{rmse_l:.4f}",
                    "ssim_h": f"{ssim_h:.4f}",
                    "ssim_l": f"{ssim_l:.4f}",
                    "f": f"{t_forward:.4f}s",
                    "l": f"{t_loss:.4f}s",
                })
                progress.update()

        logger.info(f"Test Complete: "
                    f"RMSE HQ: {total_loss[0] / total_iter:.4f} "
                    f"RMSE LQ: {total_loss[1] / total_iter:.4f} "
                    f"SSIM HQ: {total_loss[2] / total_iter:.4f} "
                    f"SSIM LQ: {total_loss[3] / total_iter:.4f}")
