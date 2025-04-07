# HOP-Heterogeneous-Topology-based-Multimodal-Entanglement-for-Co-Speech-Gesture-Generation
Hongye Cheng,  [Tianyu Wang](https://star-uu-wang.github.io/),  [Guangsi Shi](https://au.linkedin.com/in/guangsi-shi-040432126/en),  Zexing Zhao,  [Yanwei Fu](https://yanweifu.github.io/)
## [Project](https://star-uu-wang.github.io/HOP/) | [Paper](https://arxiv.org/abs/2503.01175)
Co-speech gestures are crucial non-verbal cues that enhance speech clarity and expressiveness in human communication, which have attracted increasing attention in multimodal research. While the existing methods have made strides in gesture accuracy, challenges remain in generating diverse and coherent gestures, as most approaches assume independence among multimodal inputs and lack explicit modeling of their interactions. In this work, we propose a novel multimodal learning method named HOP for co-speech gesture generation that captures the heterogeneous entanglement between gesture motion, audio rhythm, and text semantics, enabling the generation of coordinated gestures. By leveraging spatiotemporal graph modeling, we achieve the alignment of audio and action. Moreover, to enhance modality coherence, we build the audio-text semantic representation based on a reprogramming module, which is beneficial for cross-modality adaptation. Our approach enables the trimodal system to learn each other's features and represent them in the form of topological entanglement. Extensive experiments demonstrate that HOP achieves state-of-the-art performance, offering more natural and expressive co-speech gesture generation. More information, codes.

![](https://github.com/Chenghyyy/HOP-Heterogeneous-Topology-based-Multimodal-Entanglement-for-Co-Speech-Gesture-Generation/blob/main/Figures/framework.png)

## Environment
This project is developed and tested on Ubuntu 20.04, Python 3.7.16, PyTorch 1.13.1 and CUDA version 10.1. 

## Installation
1.Install required python packages:
    `pip install -r requirements.txt`
2.Download the trained autoencoder model for FGD,please refer mainly [here](https://github.com/alvinliu0/HA2G?tab=readme-ov-file) for more information related to the model.
3.Download pretrianed fasttext model from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip)

## Dataset
The dataset used is mainly from [HA2G](https://github.com/alvinliu0/HA2G?tab=readme-ov-file). Please refer mainly [here](https://github.com/alvinliu0/HA2G?tab=readme-ov-file) for more information related to the dataset.

## Training
Train the proposed HOP model on TED Gesture Dataset:
`python run_ted.py`
Train the proposed HOP model on TED Expressive Gesture Dataset:
`python run_expressive.py`
![](https://github.com/Chenghyyy/HOP-Heterogeneous-Topology-based-Multimodal-Entanglement-for-Co-Speech-Gesture-Generation/blob/main/Figures/result.png)

## Citation
If you are interested in our work or use the code in your research, please cite the following article:

@article{cheng2025hop,
  title     = {HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation},
  author    = {Cheng, Hongye and Wang, Tianyu and Shi, Guangsi and Zhao, Zexing and Fu, Yanwei},
  journal   = {arXiv preprint arXiv:2503.01175},
  year      = {2025}
}

## Acknowledgement
The codebase is developed based on [Gesture Generation from Trimodal Context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context) of Yoon et all.
