# Bidirectional-SemiSupervised-Dual-branch-CNN
This is an implementation for [Bidirectional Semi-supervised Dual-branch CNN for Robust 3D Reconstruction of Stereo Endoscopic Images via Adaptive Cross and Parallel Supervisions](https://arxiv.org/abs/2210.08291), which is a substantial extension of our conference work [Semi-supervised Learning via Improved Teacher-Student Network for Robust 3D Reconstruction of Stereo Endoscopic Image](https://dl.acm.org/doi/10.1145/3474085.3475527) (ACM MM oral presentation).

## Install
The packages and their corresponding version we used in this repository are listed in below.

- python 3.7
- PyTorch >= 1.7.0
- Numpy == 1.20

## Data Preparation

Download KITTI 2012, SERV-CT, SCARED datasets first, and organize it according to the txt files in the filenames folder.

## Inference

run the script `./scripts/kitti_save.sh`, `./scripts/servct_save.sh"`, and `./scripts/scared_save.sh"` and  to save png predictions on the KITTI2012, SERV-CT, and SCARED datasets.

## Models

[KITTI Model](https://drive.google.com/file/d/1OZSMEuOLP9SVIuUCsVhoZrV5uCUnUQxo/view?usp=sharing)
You can use this checkpoint to reproduce the result we reported in the KITTI 2012.

[SERV-CT Model](https://drive.google.com/file/d/1Q_xxm_eFWzNFJoeDA8zC0nn_RSY2wyVK/view?usp=sharing) [SCARED Model](https://drive.google.com/file/d/1YjvURzjfkvvA3UdZ00l2HNrxZ8MsChbP/view?usp=sharing)
You can use this checkpoint to reproduce the result we reported in the main paper.

## To do

Currently, we have released the inference code and trained models of the dual-branch model, which can be used to reproduce the results in our paper. We will continue to update the training code and pre-trained models in the future.

## Citation
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```
@article{shi2022bidirectional,
  title={Bidirectional Semi-supervised Dual-branch CNN for Robust 3D Reconstruction of Stereo Endoscopic Images via Adaptive Cross and Parallel Supervisions},
  author={Shi, Hongkuan and Wang, Zhiwei and Zhou, Ying and Li, Dun and Yang, Xin and Li, Qiang},
  journal={arXiv preprint arXiv:2210.08291},
  year={2022}
}

@inproceedings{shi2021semi,
  title={Semi-supervised Learning via Improved Teacher-Student Network for Robust 3D Reconstruction of Stereo Endoscopic Image},
  author={Shi, Hongkuan and Wang, Zhiwei and Lv, Jinxin and Wang, Yilang and Zhang, Peng and Zhu, Fei and Li, Qiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4661--4669},
  year={2021}
}
```

## Acknowledgment


Thanks to the excellent work GwcNet and ACFNet. Our work is inspired by these work and part of codes are migrated from [GwcNet](https://github.com/xy-guo/GwcNet) and [ACFNet](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark).

