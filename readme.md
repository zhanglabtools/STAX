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

### Environment

It is recommended to use a Python version  `3.11`.

* Set up conda environment for STAX:

```
conda create -n STAX python==3.11
```

* Install STAX:

```
conda activate STAX
```

* You need to choose the appropriate dependency pytorch and dgl for your own environment, and we recommend the following
  pytorch==2.1.2+cu118 and dgl==2.2.1+cu118:

```
Conda:
For torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
For dgl in linux
conda install -c dglteam/label/th22_cu118 dgl
For dgl in windows
conda install -c dglteam/label/cu118 dgl

Pip:
For torch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
For dgl in linux
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
For dgl in windows
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

The official command and other versions of pytorch and dgl can be found from
[torch](https://pytorch.org/) and [dgl](https://www.dgl.ai/pages/start.html).

In addition, we need scanpy package.
```
pip install scanpy==1.10.3
```
Scanpy can be found [scanpy](https://scanpy.readthedocs.io/en/stable/)

## Installation

You can install STAX as follows:

```
git clone https://github.com/zhanglabtools/STAX.git
cd STAX
python setup.py bdist_wheel sdist
cd dist
pip install STAX-0.0.1.tar.gz
```

If you have existed python environment, you can directly download the STAX package and try to import STAX in your code.
It may work, but we can't guarantee it

## Tutorials

The following are detailed tutorials. All tutorials were carried out on a notebook with a 11800H cpu and a 3070 8G gpu.

1. [1_STAX outperforms state-of-the-art (SOTA) methods in benchmark datasets for single-omics spatial domain identification](./Github_Tutorials/1_STAX_single_slice.ipynb)
2. [2_STAX accurately identifies spatial domains across different techniques, resolutions, and scale](./Github_Tutorials/2_STAX_multi_slice.ipynb)
3. [3_STAX enables integration of spatial omics cohort across hundreds of individuals](./Github_Tutorials/3_STAX_cohort.ipynb)
4. [4_STAX achieves spot completion while deciphering the complex tumor microenvironment](./Github_Tutorials/4_STAX_spot_completion.ipynb)
5. [5_STAX pinpointed cell type, domain type, and disease-related specific genes in cell and gene co-embedding](./Github_Tutorials/5_STAX_cell_gene_coembedding.ipynb)
6. [6_STAX generates high-resolution 3D point cloud of the mouse embryo brain and simulates coronal, sagittal, and transverse slices](./Github_Tutorials/6_STAX_generate_high_resolution_3D_point_cloud.ipynb)

