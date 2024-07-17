# WalkTheDog: Cross-Morphology Motion Alignment via Phase Manifolds

![Python](https://img.shields.io/badge/Python->=3.11-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=2.1-Red?logo=pytorch)


This repository provides the implementation for our vector-quantized periodic autoencoder. It learns a disconnected 1D phase manifold that aligns motion regardless of morphologies while requiring no paired data or any joint correspondence. It is based on our work [WalkTheDog: Cross-Morphology Motion Alignment via Phase Manifolds](https://peizhuoli.github.io/walkthedog/index.html) that is published in SIGGRAPH 2024.

For the Unity project for visualization, a separate repository is provided [here](https://github.com/PeizhuoLi/walk-the-dog-unity).


## Prerequisites

This code has been tested under Ubuntu 20.04. Before starting, please configure your Anaconda environment by

```bash
conda env create -f environment.yml
conda activate walk-the-dog
```

Alternatively, you may install the following packages (and their dependencies) manually:

- pytorch == 2.1.1
- onnx == 1.16.0
- numpy == 1.26.4
- scikit-learn == 1.4.2
- matplotlib == 3.8.4
- tqdm 


## Quick Start

We provide pre-trained models for the Human-Loco dataset and the Dog dataset. To run the demo, please download the pre-trained models and processed dataset [here](https://drive.google.com/file/d/13jreiLSl94Ff-ncsaK5ccT0C5hBVznm1/view?usp=sharing) and extract it under `./Datasets` directory from the root folder of the repository.

To run the demo, please execute the following command:

```bash
python test_vq.py --save=./pre-trained/human-dog
```

The learned phase manifolds, the average pose prediction neural networks, and the codebook will be exported as `Manifold_*_final.npz` file, `.onnx` file, and `VQ.npz` respectively. 

## Training from Scratch

To learn more about how to process data with Unity, please refer to our Unity repository [here](https://github.com/PeizhuoLi/walk-the-dog-unity/tree/main?tab=readme-ov-file#export-data-for-training).

After obtaining the pre-processed data, you can train the model by executing the following command:

```bash
python train_vq.py --load=dataset1path,dataset2path,dataset3path --save=./path-to-save
```

The datasets should be separated by commas and stored in the `./Datasets` folder. Note that the path should exclude `./Datasets` prefix in the command. You can put as many datasets as you want. The trained model will be saved in the `./path-to-save` folder.

After training, you can generate the files needed for the Unity side with the `test_vq.py` script.

## Motion Matching

Coming soon.


## Acknowledgments

The code is adapted from the [DeepPhase](https://github.com/sebastianstarke/AI4Animation/tree/master?tab=readme-ov-file#siggraph-2022deepphase-periodic-autoencoders-for-learning-motion-phase-manifoldssebastian-starkeian-masontaku-komuraacm-trans-graph-41-4-article-136) project under [AI4Animation](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2022/PyTorch) by [@sebastianstarke](https://github.com/sebastianstarke).

The code for the class `VectorQuantizer` is adapted from [CVQ-VAE](https://github.com/lyndonzheng/CVQ-VAE) by [@lyndonzheng](https://github.com/lyndonzheng).