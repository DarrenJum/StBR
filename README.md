# StBR: Structural Behavior Regularization

[![Paper](https://img.shields.io/badge/NeurIPS_2026-arXiv:2610.38238-b31b1b.svg)](https://arxiv.org/abs/2610.38238)
[![Python 3.9+](https://img.shields.io/badge/Python-3.8.5-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official PyTorch implementation of the NeurIPS 2026 paper: **[Variational Consequence-Driven Offline Reinforcement Learning](https://arxiv.org/abs/2610.38238)**  

## 💡 Overview & Key Innovations

Conventional offline reinforcement learning (RL) methods mitigate distribution shift by imposing divergence constraints on **policy induced state-consequence distributions**.

* **State Belief Gaussian Fitting (`vib.py`)**: Regularizing state-consequence distributions within a Gaussian latent space.
* **Offline RL training (`StBR.py`)**: Supporting both deterministic policies and generative stochastic policies (`--flow_policy`), as well as seamless offline-to-online fine-tuning(`main_finetune.py`).


## Environment

1. Install [MuJoCo version 2.1](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0) at ~/.mujoco/mujoco210
2. Create a conda environment
```
conda env create -f conda_env.yml
conda activate stbr
```
3. Install [D4RL](https://github.com/Farama-Foundation/D4RL/tree/4aff6f8c46f62f9a57f79caa9287efefa45b6688)

## Usage

### Offline RL

Run the following command to train offline RL on D4RL.

```
python main.py --config configs/offline/hopper-medium.yml
python main.py --config configs/offline/antmaze-large-diverse.yml
```

You can also specify the random seed:

```
python main.py --config configs/offline/hopper-medium.yml --seed <seed>
```

#### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```

### Online Fine-tuning

Run the following command to online fine-tune on AntMaze with pretrained offline agents.

```
python main_finetune.py --config configs/online_finetune/antmaze-large-diverse.yml
```

You can also specify the random seed offline models:

```
python main_finetune.py --config configs/online_finetune/antmaze-large-diverse.yml --seed <seed> --pretrain_model <pretrain_model/>
```

## Citation

If you find this code useful for your research, please cite our paper as:

```
@inproceedings{jiang2026variational,
  title={Variational Consequence-Driven Offline Reinforcement Learning},
  author={Jiang, Wen and Jiang, Ke and Wang, Jin and Tan, Xiaoyang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgement

This repo borrows heavily from [SPOT](https://github.com/thuml/SPOT) and [ATAC](https://github.com/microsoft/ATAC).
