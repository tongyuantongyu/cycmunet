from collections import namedtuple

model_arg = namedtuple('model_arg', ('nf',  # number of feature channel
                                     'groups',  # number of deformable convolution group
                                     'upscale_factor',  # model upscale factor
                                     'format',  # model I/O format (rgb, yuv420)
                                     'layers',  # feature fusion pyramid layers
                                     'cycle_count'  # mutual cycle count
                                     ))
