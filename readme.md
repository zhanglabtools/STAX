# STAX

## Overview

The rapid advancement of spatial omics technology has revolutionized biomedical research, unveiling unprecedented 
insights into the molecular and cellular architecture of biological systems. However, despite its transformative 
potential, the analysis of spatial omics data presents significant computational and analytical challenges. To address 
these hurdles, we introduce STAX—an innovative, lightweight, and multi-task Graph Attnetion Network (GAT) framework 
designed to unify diverse analytical tasks within a single computational platform. STAX excels across numerous critical 
applications, including spatial domain identification, spatial slice integration, cohort-level spatial analysis, spatial 
spot completion, cell-gene co-embedding, expression profile denoising, and 3D spatial multi-slice simulation. Extensive 
evaluations demonstrate that STAX consistently delivers superior performance, robustness, and biologically meaningful 
interpretations, establishing it as an indispensable tool for spatial omics research. As a versatile and computationally 
efficient framework, STAX effectively overcomes multiple analytical challenges in spatial omics, empowering researchers 
with a powerful platform to accelerate biomedical discoveries and deepen our understanding of complex biological systems. 

![](./Figure_main.jpg)

## Doc

TODO

## Prerequisites

### Data

The data can be download in [google driver](https://drive.google.com/drive/folders/18tcl-PRdK9j-W59GUPdsKgy_04IJvz05)

The URL is: https://drive.google.com/drive/folders/18tcl-PRdK9j-W59GUPdsKgy_04IJvz05

### Base environment

#### Declaration: All of our experiments were implemented in Windows 10 and python 3.11 environment. We also provide a linux ubuntu 22.04 installation pipeline, but the linux results are not exactly consistent with the Windows 10 results. One variation of result can be seen in Epigenomics mouse_brain_dataset_1 in tutorial1. This may be due to differences in numpy's underlying math libraries or compilers used across the two systems.

It is recommended to use a Python version  `3.11`.

* Set up conda environment for STAX:
```
conda create -n STAX python==3.11
```
* Activate STAX environment:
```
conda activate STAX
```
* Pytorch and DGL are 2 key libraries for STAX. When configuring a server with various CUDA versions, conda offers more flexible environment management. On a personal computer, however, pip often provides a quicker installation. The official conda and pip command and other versions of pytorch and dgl can be found from
[torch](https://pytorch.org/) and [dgl](https://www.dgl.ai/pages/start.html). You need to choose the appropriate dependency pytorch and dgl for your own environment, and we recommend the 
  pytorch==2.1.2+cu118 and dgl==2.2.1+cu118. 
* For torch:
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
* For dgl:
```
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
```
In addition. we uploaded dgl-2.2.1+cu118 whl file to https://drive.google.com/drive/folders/18tcl-PRdK9j-W59GUPdsKgy_04IJvz05.
Once the whl file is downloaded, 'cd' to the directory and pip install it. For example, 
```
For windows
pip install dgl-2.2.1+cu118-cp311-cp311-win_amd64.whl
For linux
pip install dgl-2.2.1+cu118-cp311-cp311-manylinux1_x86_64.whl
```

* We also tried other version combinations of pytorch and DGL, and all of them produced similar results including.
* pytorch==2.2.0+cu118 and dgl==2.4.0+cu118
```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
```
* pytorch==2.3.0+cu121 and dgl==2.4.0+cu121
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```
* pytorch==2.4.0+cu124 and dgl==2.4.0+cu124
```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```


#### If you want to use the R clustering algorithm mclust in python.
* Install R first. 
```
conda install -c conda-forge r-base==4.2.0
```
* Then, install r-mclust
```
conda install -c conda-forge r-mclust
```
* Finally, install rpy2
```
conda install rpy2 
```

## Installation STAX

* You can install STAX as follows:

```
git clone https://github.com/zhanglabtools/STAX.git
cd STAX
python setup.py bdist_wheel sdist
cd dist
pip install stax-0.0.1-py3-none-any.whl
```

Notably, if you have existed python environment, you can directly download the STAX package and try to import STAX in your code.
It may work, but we can't guarantee it.


## Set up jupyter lab

Once you have configured your STAX environment using conda. Add STAX to jupyter's core. 

* Specifically, we have installed ipykernel automatically through STAX, if it does not install successfully, 
please enter the following command.
```
pip install ipykernel
```
* Then enter the following command:
```
python -m ipykernel install --user --name=STAX --display-name STAX
```

## Tutorials

The following are detailed tutorials. All tutorials were carried out on a notebook with a 11800H cpu and a 3070 8G gpu.

1. [1_STAX outperforms state-of-the-art (SOTA) methods in benchmark datasets for single-omics spatial domain identification](./Github_Tutorials/1_STAX_single_slice.ipynb)
2. [2_STAX accurately identifies spatial domains across different techniques, resolutions, and scale](./Github_Tutorials/2_STAX_multi_slice.ipynb)
3. [3_STAX enables integration of spatial omics cohort across hundreds of individuals](./Github_Tutorials/3_STAX_cohort.ipynb)
4. [4_STAX achieves spot completion while deciphering the complex tumor microenvironment](./Github_Tutorials/4_STAX_spot_completion.ipynb)
5. [5_STAX pinpointed cell type, domain type, and disease-related specific genes in cell and gene co-embedding](./Github_Tutorials/5_STAX_cell_gene_coembedding.ipynb)
6. [6_STAX generates high-resolution 3D point cloud of the mouse embryo brain and simulates coronal, sagittal, and transverse slices](./Github_Tutorials/6_STAX_generate_high_resolution_3D_point_cloud.ipynb)


## BTW
If you have any problems or suggestions, please let us know in the issues section.


## Citations
If you find this method useful for your research, please consider citing the following paper:
```
STAX: Exploring spatial omics data with multi-task graph attention networks
```