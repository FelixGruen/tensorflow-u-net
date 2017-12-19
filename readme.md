This repository contains an implementation of the [U-Net](https://arxiv.org/abs/1505.04597) and of a more parameter efficient variant of the U-Net as well as the code needed to train a network on the [LiTS lesion dataset](http://lits-challenge.com/). The entire code is written in Python 2 and uses the [TensorFlow](http://tensorflow.org/) library for the implementation and training of the models and the [Data Processing Pipeline (dpp)](https://github.com/FelixGruen/data-processing-pipeline) library for the implementation of the pre-processing pipeline.

The parameter efficient version of the U-Net uses concatenated ReLUs, full pre-activation identity connections, batch normalization, and depth-wise separable convolutions to reduce the parameter count by over 60 % while obtaining Dice scores comparable to the original U-Net.

## General Overview

The network architectures are defined in **networks.py** in the **architecture** folder. They make use of the layers and building blocks defined in **layers.py** in the same folder.

The **utils** folder contains functions needed for training. The file **measurements.py** implements the SegmentationRecorder, which is used during training to record all the values needed to compute and save performance indicators, most notably the Dice score. The files **lesion_preprocessing.py** and **liver_preprocessing.py** define the pre-processing pipelines for the lesion and liver segmentation respectively.

The file **training.py** contains all the code needed to initialize or load a network and train it on the LiTS dataset. It will save different performance indicators and images during training, which can be used within TensorBoard to get an overview of the current training progress. It is called from the command line and help is available through

    python experiment.py --help

The file **lesion_generate_predictions.py** and **liver_generate_predictions.py** take a trained network and generate prediction volumes for all the data in the specified directory. They are also called from the command line and help is likewise available through

    python lesion_generate_predictions.py --help

## Installation

Simply clone the repository into the directory of your choice. Make sure the dpp library is available in your Python path. If it is not, you can use

    export PYTHONPATH=$PYTHONPATH:/path/to/the/data-processing-pipeline/

to add it.

## See also

The code implements the Cascaded-FCN approach presented in

---

Christ, P. F., et al. "Automatic liver and lesion segmentation in CT using cascaded fully convolutional neural networks and 3D conditional random fields". *International Conference on Medical Image Computing and Computer-Assisted Intervention.* 2016. URL: <https://arxiv.org/abs/1610.02177>

---

and used in

---

Christ, P. F., et al. "Automatic Liver and Tumor Segmentation of CT and MRI Volumes using Cascaded Fully Convolutional Neural Networks". *arXiv preprint arXiv:1702.05970.* 2017. URL: <https://arxiv.org/abs/1702.05970>

---

and

---

Christ, P. F., et al. "SurvivalNet: Predicting patient survival from diffusion weighted magnetic resonance images using cascaded fully convolutional and 3D convolutional neural networks". *IEEE International Symposium on Biomedical Imaging.* 2017. URL: <https://arxiv.org/abs/1702.05941>

---

For more code related to medical imaging, please see the repository of the IBBM research group at <https://github.com/IBBM/>.