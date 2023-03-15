# MindSpore implementation of CycMuNet+

This is the MindSpore implementation of CycMuNet+ capable of running on CPU and
Huawei Ascend AI Processor.

## Installation

We recommend using a prebuilt image provided by Huawei  with necessary Ascend
and MindSpore dependencies installed. Run following command to install extra
necessary dependencies:

```bash
conda env update -f environment.yml
```

`environment-full.yml` contains a full list of libraries in the environment.
Note that you may not be able to recreate the environment using this file.
Instead see [https://mindspore.cn/install](https://mindspore.cn/install) for
instructions to manually install MindSpore if you prefer.

## Contents

### `train.py`

Train the network.

### `inference.py`

Do network inference.

### `convert_weight.py`

Import checkpoints from PyTorch to MindSpore.

### `build_mindir.py`

Build MindIR file.
