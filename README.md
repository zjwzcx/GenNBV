<br>
<p align="center">
<h1 align="center"><strong>GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction</strong></h1>
  <p align="center">
  	<strong>CVPR 2024</strong>
	<br>
    <a href='https://xiao-chen.tech/' target='_blank'>Xiao Chen</a>&emsp;
	<a href='https://quanyili.github.io/' target='_blank'>Quanyi Li</a>&emsp;
	<a href='https://tai-wang.github.io/' target='_blank'>Tai Wang</a>&emsp;
    <a href='https://tianfan.info/' target='_blank'>Tianfan Xue</a>&emsp;
	<a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a>&emsp;
    <br>
    Shanghai AI Laboratory&emsp;The Chinese University of Hong Kong
    <br>
  </p>
</p>


<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2402.16174-blue)](https://arxiv.org/abs/2402.16174)
[![](https://img.shields.io/badge/Paper-%F0%9F%93%96-blue)](./assets/2024_CVPR_GenNBV.pdf)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-blue)](https://gennbv.tech/)

</div>

## 📋 Contents (Coming Soon)
1. [About](#-about)
2. [Getting Started](#-getting-started)
3. [Model and Benchmark](#-Model-and-Benchmark)
4. [TODO List](#-todo-list)
5. [Citation](#-citation)
6. [License](#-license)
7. [Acknowledgements](#-acknowledgements)

## 🏠 About
<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/Fig_Teaser.png" alt="Dialogue_Teaser" width=100% >
</div>
While recent advances in neural radiance field enable realistic digitization for large-scale scenes, the image-capturing process is still time-consuming and labor-intensive. Previous works attempt to automate this process using the Next-Best-View (NBV) policy for active 3D reconstruction. However, the existing NBV policies heavily rely on hand-crafted criteria, limited action space, or per-scene optimized representations. These constraints limit their cross-dataset generalizability. To overcome them, we propose <b>GenNBV</b>, an end-to-end generalizable NBV policy. Our policy adopts a reinforcement learning (RL)-based framework and extends typical limited action space to 5D free space. It empowers our agent drone to scan from any viewpoint, and even interact with unseen geometries during training. To boost the cross-dataset generalizability, we also propose a novel multi-source state embedding, including geometric, semantic, and action representations. We establish a benchmark using the Isaac Gym simulator with the Houses3K and OmniObject3D datasets to evaluate this NBV policy. Experiments demonstrate that our policy achieves a 98.26% and 97.12% coverage ratio on unseen building-scale objects from these datasets, respectively, outperforming prior solutions.

## 📚 Getting Started
### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 545.29.02
- CUDA 11.3
- Python 3.8.12
- PyTorch 1.11.0+cu113
- PyTorch3D 0.7.5

1. Clone this repository.

```bash
git clone https://github.com/zjwzcx/GenNBV
cd GenNBV
```

2. Create an environment and install PyTorch.

```bash
conda create -n gennbv python=3.8 -y  # pytorch3d needs python>3.7
conda activate gennbv
# Install PyTorch, for example, install PyTorch 1.11.0 for CUDA 11.3
# For more information, please refer to https://pytorch.org/get-started/locally/
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Install GenNBV.

```bash
# TODO
```

### Data Preparation

Please refer to the [guide](data/README.md) for downloading and organization.


## 📦 Model and Benchmark
### Model Overview

<p align="center">
  <img src="assets/Fig_Method.png" align="center" width="100%">
</p>

### Benchmark Overview

<p align="center">
  <img src="assets/exp_main_table.png" align="center" width="100%">
</p>

## 📝 TODO List

## 🔗 Citation
If you find our work helpful, please cite:

```bibtex
@inproceedings{chen2024gennbv,
  title={GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction},
  author={Chen, Xiao and Li, Quanyi and Wang, Tai and Xue, Tianfan and Pang, Jiangmiao},
  year={2024}
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements