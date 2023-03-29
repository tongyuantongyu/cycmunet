import math
import logging
import os
import pathlib
import time

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cuda
import torch.backends.cudnn
import torchvision.utils
from pytorch_msssim import SSIM

from model import CycMuNet
from model.util import converter, normalizer
import dataset
from cycmunet.model import model_arg
from cycmunet.run import train_arg

# ------------------------------------------
# Configs

model_args = model_arg(nf=64,
                       groups=8,
                       upscale_factor=2,
                       format='yuv420',
                       layers=4,
                       cycle_count=3
                       )

train_args = train_arg(
    size=(128, 128),
    pretrained="/root/cycmunet-new/checkpoints/monitor-ugly_2x_l4_c3_epoch_19.pth",
    # dataset_type="video",
    # dataset_indexes=[
    #     "/root/videos/cctv-scaled/index-train-good.txt",
    #     "/root/videos/cctv-scaled/index-train-ugly.txt",
    #     "/root/videos/cctv-scaled/index-train-smooth.txt",
    #     "/root/videos/cctv-scaled/index-train-sharp.txt",
    # ],
    dataset_type="triplet",
    dataset_indexes=[
        "/root/dataset/vimeo_triplet/tri_trainlist.txt"
    ],
    preview_interval=100,
    seed=0,
    lr=0.001,
    start_epoch=1,
    end_epoch=11,
    sparsity=True,
    batch_size=2,
    autocast=False,
    loss_type='rmse',
    save_path='checkpoints',
    save_prefix='triplet',
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# --------------------------------------
# Start of code

preview_interval = 100 \
    if (len(train_args.dataset_indexes) == 1 or math.gcd(100, len(train_args.dataset_indexes)) == 1) \
    else 101

save_prefix = f'{train_args.save_prefix}_{model_args.upscale_factor}x_l{model_args.layers}_c{model_args.cycle_count}'

save_path = pathlib.Path(train_args.save_path)

nrow = 1 if train_args.size[0] * 9 > train_args.size[1] * 16 else 3

torch.manual_seed(train_args.seed)
torch.cuda.manual_seed(train_args.seed)

formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s]: %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

logger = logging.getLogger('train_progress')
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
Dataset = dataset_types[train_args.dataset_type]
if len(train_args.dataset_indexes) == 1:
    ds_train = Dataset(train_args.dataset_indexes[0],
                       train_args.size,
                       model_args.upscale_factor,
                       augment=True,
                       seed=train_args.seed)
else:
    ds_train = dataset.InterleavedDataset(*[
        Dataset(dataset_index,
                train_args.size,
                model_args.upscale_factor,
                augment=True,
                seed=train_args.seed + i)
        for i, dataset_index in enumerate(train_args.dataset_indexes)])
ds_train = DataLoader(ds_train,
                      num_workers=1,
                      batch_size=train_args.batch_size,
                      shuffle=Dataset.want_shuffle,  # Video dataset friendly
                      drop_last=True)

model = CycMuNet(model_args)
model.train()
model_updated = False
num_params = 0
for param in model.parameters():
    num_params += param.numel()
logger_init.info(f"Model has {num_params} parameters.")

if train_args.pretrained:
    if not os.path.exists(train_args.pretrained):
        logger_init.warning(f"Pretrained weight {train_args.pretrained} not exist.")
    state_dict = torch.load(train_args.pretrained, map_location=lambda storage, loc: storage)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        logger_init.warning(f"Unknown parameters ignored: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        logger_init.warning(f"Missing parameters not initialized: {load_result.missing_keys}")
    logger_init.info("Pretrained weights loaded.")

model = model.cuda()
optimizer = optim.Adamax(model.parameters(), lr=train_args.lr, betas=(0.9, 0.999), eps=1e-8)
# Or, train only some parts
# optimizer = optim.Adamax(itertools.chain(
#         model.head.parameters(),
#         model.fe.parameters(),
#         model.fr.parameters(),
#         model.tail.parameters()
# ), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40000, eta_min=1e-7)

num_params_train = 0
for group in optimizer.param_groups:
    for params in group.get('params', []):
        num_params_train += params.numel()
logger_init.info(f"Model has {num_params} parameters to train.")

if train_args.sparsity:
    from apex.contrib.sparsity import ASP

    target_layers = []
    target_layers.extend('mu.' + name for name, _ in model.mu.named_modules())
    target_layers.extend('fr.' + name for name, _ in model.fr.named_modules())

    ASP.init_model_for_pruning(model,
                               mask_calculator="m4n2_1d",
                               allowed_layer_names=target_layers,
                               verbosity=2,
                               whitelist=[torch.nn.Linear, torch.nn.Conv2d],
                               allow_recompute_mask=False,
                               allow_permutation=False,
                               )
    ASP.init_optimizer_for_pruning(optimizer)

    # import torch.fx
    # original_symbolic_trace = torch.fx.symbolic_trace
    # torch.fx.symbolic_trace = functools.partial(original_symbolic_trace, concrete_args={
    #     'batch_mode': '_no_use_sparsity_pseudo',
    #     'stop_at_conf': False,
    #     'all_frames': True
    # })

    ASP.compute_sparse_masks()
    # torch.fx.symbolic_trace = original_symbolic_trace
    logger.info('Training with sparsity.')


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


def train(epoch):
    epoch_loss = 0
    total_iter = len(ds_train)
    loss_coeff = [1, 0.5, 1, 0.5]
    with tqdm.tqdm(total=total_iter, desc=f"Epoch {epoch}") as progress:
        for it, data in enumerate(ds_train):
            optimizer.zero_grad()

            def compute_loss(force_data_dtype=None):
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
                actual = [cvt.yuv2rgb(*norm.denormalize_yuv_420(*out)) for out in outs]

                if train_args.loss_type == 'rmse':
                    loss = [rmse(a, t) * c for a, t, c in zip(actual, target, loss_coeff)]
                elif train_args.loss_type == 'ssim':
                    loss = [ssim(a, t) * c for a, t, c in zip(actual, target, loss_coeff)]
                else:
                    raise ValueError("Unknown loss type: " + train_args.loss_type)

                assert not any(torch.any(torch.isnan(i)).item() for i in loss)

                t2 = time.perf_counter()

                if it % preview_interval == 0:
                    out = [i[0:1].detach().float().cpu() for i in actual[:3]]
                    ref = [i[0:1].detach().float().cpu() for i in target[:3]]

                    for idx, ts in enumerate(zip(org, out, ref)):
                        torchvision.utils.save_image(torch.concat(ts), f"./result/out{idx}.png",
                                                     value_range=(0, 1), nrow=nrow, padding=0)

                return loss, t1 - t0, t2 - t1

            if train_args.autocast:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, t_forward, t_loss = compute_loss(torch.float16)
            else:
                loss, t_forward, t_loss = compute_loss()

            total_loss = sum(loss)
            epoch_loss += total_loss.item()

            t3 = time.perf_counter()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            t_backward = time.perf_counter() - t3

            global model_updated
            model_updated = True

            progress.set_postfix(ordered_dict={
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6e}",
                "f": f"{t_forward:.4f}s",
                "l": f"{t_loss:.4f}s",
                "b": f"{t_backward:.4f}s",
            })
            progress.update()

    logger.info(f"Epoch {epoch} Complete: Avg. Loss: {epoch_loss / total_iter:.4f}")


def save_model(epoch):
    if epoch == -1:
        name = "snapshot"
    else:
        name = f"epoch_{epoch}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_path = save_path / f"{save_prefix}_{name}.pth"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Checkpoint saved to {output_path}")


if __name__ == '__main__':
    try:
        for epoch in range(train_args.start_epoch, train_args.end_epoch):
            # with torch.autograd.detect_anomaly():
            #     train(epoch)
            train(epoch)
            save_model(epoch)
    except KeyboardInterrupt:
        if model_updated:
            save_model(-1)
