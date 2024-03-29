import vapoursynth as vs

core = vs.core

# Load example.mp4 as input video
# The ffms2 plugin can be installed by running
# $ conda install -c conda-forge -c tongyuantongyu vapoursynth-ffms2
clip = core.ffms2.Source(source="example.mp4")

clip = core.cycmunet.CycMuNet(clip,
                              scale_factor=2,
                              batch_size=1,
                              batch_size_fusion=1,
                              use_fp16=True,
                              low_mem=True,
                              model_path=r"./models",
                              model="2x_vimeo"
                              )

clip.set_output()

# - `scale_factor_h`: float. The scale factor of height of the network in use.
# This is default tobe the same as `scale_factor`. (*)
# - `batch_size_extract`: int. The batch size of Extract model. Default automatically
# selected depending on other parameters.
# - `batch_size_fusion`: int. The batch size of Fusion model. Default to 1.
# - `input_count`: int. The number of input frames network needed. Default to 1. (*)
# - `feature_count`: int. The "feature" (`C` channel) size. Default to 64. (*)
# - `extraction_layers`: int. The number of layers Extract model outputs. Default to 1. (*)
# - `interpolation`: bool. If the network is doing frame interpolation
# (i.e. output clip) will have double framerate Default to False.
# - `extra_frame`: bool. If network need 1 more input frame than consumed.
# Default to False. (*)
# - `double_frame`: bool. If network outputs 2 times of frames than input.
# Default to False. (*)
# - `use_fp16`: bool. Use half precision during inference. On supported GPUs
# (starting from Volta), this is usually ~2x faster and consumes half
# amount of GPU memory, but may cause numeric instability for some
#     networks.
# - `low_mem`: bool. Tweak TensorRT configurations to reduce memory usage.
# May cause performance degradation and effectiveness varies depending on
# actual model. Default to False.
# - `norm_mean` and `norm_std`: float[3]. Normalization mean and std applied
# to inputs and output. The interpretation of these values depending on the
# following option. Defaults to [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] (*)
# - `raw_norm`; bool. If True, `norm_mean` and `norm_std` are applied directly
# to the input and output frame pixel value of each channel.
# If False, `norm_mean` and `norm_std` are values of RGB channels from
# 0-1 range. The actual value used for normalization is inferred automatically
# from colorspace information of input clip. Default to False. (*)
# - `model`: str. The name of the model to be used. Default to ".". (*)
# - `model_path`: str. The path that stores the model files.
# Default to `dev.tyty.aim.nnvisr` folder under the folder of plugin
# DLL.


