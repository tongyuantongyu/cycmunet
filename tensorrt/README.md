# TensorRT implementation of CycMuNet+

This is the TensorRT implementation of CycMuNet+ capable of running on NVIDIA GPU.

## Installation

We provide precompiled binary of VapourSynth plugin for Windows and
Linux x64 platforms on Anaconda, and recommend installing via conda:

```bash
conda create -n cycmunet -c conda-forge -c nvidia -c tongyuantongyu vapoursynth-cycmunet
```

We recommend enable CUDA lazy loading by running the following command:

```bash
conda env config vars set CUDA_MODULE_LOADING=LAZY -n cycmunet
```

## Build

Building requires CUDAToolkit and TensorRT installed. As usual, you can set 
`CUDAToolkit_ROOT` and `TensorRT_ROOT` if they are not installed in default location.

To build VapourSynth plugin you need to provide VapourSynth headers as well.

```bash
mkdir build && cd build
cmake -G Ninja -DVAPOURSYNTH_PLUGIN=ON ..
ninja
```

## Usage

### `cycmunet_y4m`

Standalone inference runner accepts y4m input and produces y4m output.

Read input from stdin, and write output to stdout:

```bash
cycmunet_y4m
```

Read input from stdin, and write output to `output.y4m`:

```bash
cycmunet_y4m output.y4m
```

Read input from `input.y4m`, and write output to `output.y4m`:

```bash
cycmunet_y4m input.y4m output.y4m
```

### `vs-cycmunet`

VapourSynth plugin. Supports RGB and YUV inputs.
See [VapourSynth documentation](http://vapoursynth.com/doc) for the usage of
VapourSynth.

```python
def core.cycmunet.CycMuNet(clip: vs.VideoNode,
                           scale: float,
                           batch_size: int,
                           batch_size_fusion: int,
                           use_fp16: bool,
                           extraction_layers: int,
                           norm_mean: List[float, float, float],
                           norm_std: List[float, float, float],
                           raw_norm: bool,
                           model: str,
                           model_path: str,
                           low_mem: bool):
    """
    Run CycMuNet+ Spatio-Temporal Super Resolution on the input clip.

    :param clip: Input clip
    :param scale: Spatial scale ratio
    :param batch_size: Batch size for feature extract phase. Default: 1
    :param batch_size_fusion: Batch size for feature fusion phase. Default: batch_size
    :param use_fp16: Use FP16 (half precision) data format during inference.
     Half memory consumption, and runs faster. Requires at least Volta architecture. Default: False
    :param extraction_layers: Model extraction layers number. Default 4
    :param norm_mean: Mean value of each channel for normalization. Default: [0.485, 0.456, 0.406]
    :param norm_std: Standard derivation of each channel for normalization. Default: [0.229, 0.224, 0.225]
    :param raw_norm: Whether passed norm_mean and norm_std are values for input. If False, then mean and std are
     values of RGB in range [0, 1], and will be internally converted to values for input based on video properties.
     Default: False
    :param model: Model name used for inference. Default: "."
    :param model_path: Model storage location. Default is in "dev.tyty.aim.cycmunet" folder next to plugin file.
    :param low_mem: Enable tweaks to reduce memory consumption. Default: False
    :return: Output clip
    """
    pass
```

See demo.vpy for a basic example of how to use the plugin in VapourSynth script.

The following command can be used to get output video:
```bash
vspipe demo.vpy -c y4m - | ffmpeg -i - -c:v libx264 output.mp4
```

The plugin reads `_SceneChangePrev` property on frame to handle scene change.
You can use other plugins to add scene change information.