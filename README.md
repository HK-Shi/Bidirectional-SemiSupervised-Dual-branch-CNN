# Bidirectional-SemiSupervised-Dual-branch-CNN
This is an implementation for [Bidirectional Semi-supervised Dual-branch CNN for Robust 3D Reconstruction of Stereo Endoscopic Images via Adaptive Cross and Parallel Supervisions](https://arxiv.org/abs/2210.08291).

## Install
The packages and their corresponding version we used in this repository are listed in below.

- PyTorch==1.8.0
- Torchvision==0.9.0
- scikit-learn==1.0.2
- apex==0.1

## Training
After configuring the environment, please use this command to train the model.

```sh
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 train.py \
    --logdir ./checkpoints/scared/ --loadckpt ./checkpoints/scared/checkpoint_scared.ckpt \
    --port 12345 --lr 0.001 --fold_num 1 --CV

```

## Testing
Use this command to obtain the testing results.
```sh

python test_singlebranch_scared.py --CV --fold_num 0 --loadckpt ./checkpoints/scared/checkpoint_scared.ckpt

```


## Citation
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```
@article{shi2022bidirectional,
  title={Bidirectional Semi-supervised Dual-branch CNN for Robust 3D Reconstruction of Stereo Endoscopic Images via Adaptive Cross and Parallel Supervisions},
  author={Shi, Hongkuan and Wang, Zhiwei and Zhou, Ying and Li, Dun and Yang, Xin and Li, Qiang},
  journal={arXiv preprint arXiv:2210.08291},
  year={2022}
}
```

## Acknowledgment

Some codes are modified from [GwcNet](https://github.com/xy-guo/GwcNet) and [ACFNet](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark).
Thanks a lot for their great contribution.
