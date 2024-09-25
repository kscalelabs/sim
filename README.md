<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/onshape/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/sim/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/sim/actions/workflows/test.yml)
[![Update Stompy S3 Model](https://github.com/kscalelabs/sim/actions/workflows/update_stompy_s3.yml/badge.svg)](https://github.com/kscalelabs/sim/actions/workflows/update_stompy_s3.yml)

</div>

# K-Scale Sim Library

A library for simulating Stompy in Isaac Gym. This library is built on top of
the Isaac Gym library and Humanoid-gym and provides a simple interface for
running experiments with Stompy. For a start, we have defined two tasks:
getting up and walking.

We will be adding more tasks and simulator environments in upcoming weeks.

The walking task works reliably with upper body being fixed.
The getting up task is still an open challenge!


## Getting Started

This repository requires Python 3.8 due to compatibility issues with underlying libraries. We hope to support more recent Python versions in the future.

1. Clone this repository:
```bash
git clone https://github.com/kscalelabs/sim.git
cd sim
```

2. Create a new conda environment and install the package:
```bash
conda create --name kscale-sim-library python=3.8.19
conda activate kscale-sim-library
make install-dev
```

3. Configure environment variables:
```bash
conda env config vars set MODEL_DIR=sim/resources
```

4. Install third-party dependencies:

Manually download `IsaacGym_Preview_4_Package.tar.gz` from https://developer.nvidia.com/isaac-gym, and run:
```bash
tar -xvf IsaacGym_Preview_4_Package.tar.gz
conda env config vars set ISAACGYM_PATH=`pwd`/isaacgym
conda deactivate
conda activate kscale-sim-library
make install-third-party-external
```

### Running experiments
1. Run walking training with the following command:
```bash
python sim/train.py --task=stompymini --num_envs=4096 --headless
```
or standing policy:
```
python sim/train.py --task=stompymini_standing --num_envs=4096 --headless
```
2. Run evaluation with the following command:
```bash
python sim/play.py --task stompymini --sim_device cpu
```

### Contributing
See the [contributing guide](CONTRIBUTING.md) to get started.

### Errors
After cloning Isaac Gym, sometimes the bindings mysteriously disappear.
To fix this, update the submodule:
```bash
git submodule update --init --recursive
```

If you observe errors with libpython3.8.so.1.0, you can try the following:
```bash
export LD_LIBRARY_PATH=PATH_TO_YOUR_ENV/lib:$LD_LIBRARY_PATH
```

If you still see segmentation faults, you can try the following:
```bash
sudo apt-get vulkan1
```

### Appreciation
- [Humanoid-gym](https://sites.google.com/view/humanoid-gym/)
- KScale Labs community for bugspotting and feedback

### Discord
- [Discord](https://discord.com/invite/rhCy6UdBRD)
