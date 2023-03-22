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
