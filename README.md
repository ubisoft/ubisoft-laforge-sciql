<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <h1>Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment</h1>
  <br>
  <h2>
    <a href="https://arxiv.org/pdf/2601.22823">Paper</a> |
    <a href="https://mathieu-petitbois.github.io/projects/sciql/">Website</a>
  </h2>
</div>

</div>

## Overview
This repository is the official implementation of the **Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment** paper published at ICML 2026 (Spotlight).

## Installation
1) Create and activate conda environment: 
```
conda create -n sciql python=3.10 -y && conda activate sciql
```
2) Install  pip dependencies:
```
python -m pip install --no-cache-dir -r requirements.txt
```
3) Setup the environment variables:
```
conda env config vars set PYTHONPATH="$PYTHONPATH:$PWD"
conda env config vars set MUJOCO_GL=egl
conda deactivate && conda activate sciql
```
4) Download the datasets from <a href="https://sciql-iclr-2026.github.io/">our website</a>. Organize them in a datasets/ folder at the root of the project:
```
datasets/
└── diverse_mujoco/
    └── mujoco_halfcheetah-fix/
        └── fix-val.npz
        └── fix.npz
    └── mujoco_halfcheetah-stitch/
        └── stitch-val.npz
        └── stitch.npz
    └── mujoco_halfcheetah-vary/
        └── vary-val.npz
        └── vary.npz
└── traj2d
    └── random_circles-inplace-v0/
    └── random_circles-navigate-v0/
```

## Usage
To reproduce the results, launch the experiments with the following commands:
```
# For Halfcheetah

python experiments/control/launch.py -cp yamls/diverse_mujoco/bc/jax -cn mujoco_halfcheetah

python experiments/control/launch.py -cp yamls/diverse_mujoco/cbc/jax -cn mujoco_halfcheetah

python experiments/control/launch.py -cp yamls/diverse_mujoco/bcpmi_joint/jax -cn mujoco_halfcheetah

python experiments/control/launch.py -cp yamls/diverse_mujoco/scbc/jax -cn mujoco_halfcheetah

python experiments/control/launch.py -cp yamls/diverse_mujoco/sciql_joint/jax -cn mujoco_halfcheetah

python experiments/control/launch.py -cp yamls/diverse_mujoco/sorl/jax -cn mujoco_halfcheetah

# For Circle2D

python experiments/control/launch.py -cp yamls/traj2d/bc/jax -cn random_circles-v0

python experiments/control/launch.py -cp yamls/traj2d/cbc/jax -cn random_circles-v0

python experiments/control/launch.py -cp yamls/traj2d/bcpmi_joint/jax -cn random_circles-v0

python experiments/control/launch.py -cp yamls/traj2d/scbc/jax -cn random_circles-v0

python experiments/control/launch.py -cp yamls/traj2d/sciql_joint/jax -cn random_circles-v0

python experiments/control/launch.py -cp yamls/traj2d/sorl/jax -cn random_circles-v0
```

## Aknownledgments
This codebase takes inspiration from the [jax_corl](https://github.com/nissymori/JAX-CORL) library.

## Citation
Accepted at ICML 2026. Proceedings citation will be updated once available.
```
@inproceedings{petitbois2026offline,
  title     = {Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment},
  author    = {Petitbois, Mathieu and Portelas, R{\'e}my and Lamprier, Sylvain},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

© [2026] Ubisoft Entertainment. All Rights Reserved.
