# PPDL: Physics-Prior Dynamic Learning for Human Pose Estimation from mmWave Radar



## Innovation Points
## PPDHF Module
<p align="center">
  <img src="./assets/figure_3.png" width="100%" /> 
</p>

## DSTG Module
<p align="center">
  <img src="./assets/figure_4.png" width="100%" /> 
</p>

## Introduction

Current research on millimeter-wave (mmWave) radar-based 3D human pose estimation faces notable challenges in effectively modeling cross-modal physical correlations and fine-grained local geometric structures. Under complex scenarios, specular reflection inevitably causes limb signal loss, while cross-modal physical constraints are often neglected, severely hindering effective multimodal fusion and dynamic skeleton-graph relation learning. To address these limitations, we propose Physical Prior Dynamic Learning (PPDL), a novel framework for radar-based human pose estimation. For effective multimodal fusion, we propose a Physical Prior-Driven Holographic Fusion (PPDHF) module, which constructs cross-modal pseudo-representations through physics-guided transformations and performs consistency fusion in the complex plane, thereby substantially enhancing the coupling and complementary utilization of multimodal information. However, feature representations within individual modalities may still amplify background clutter and non-human noise. To mitigate this issue, we further propose a Dynamic Spatio-Temporal Graph (DSTG) module, which reshapes spatial structures by integrating a learnable dynamic matrix, positional bias, and edge bias, thereby weakening spurious multipath effects and alleviating geometric misalignment. Meanwhile, a spatio-temporal coherence mechanism is employed to suppress asynchronous noise and improve the continuity of inter-frame geometric representations.

### Framework

![framework](./assets/figure_2.png)


### Visualizations

![visualization](./assets/vispicture.png)

## Code
Preparation:
We evaluate our model on the public **MVDoppler-Pose** and **HuPR** datasets. 
Please download the processed MVDoppler-Pose dataset from [Google Drive](<your_google_drive_link_here>) and extract it into the `data/MVDoppler-Pose` directory.

Installation:
Our development and execution environments are fully encapsulated inside a Docker container.

1. Build the Docker image using the provided Dockerfile:
```bash
docker build -t ppdl:latest .
