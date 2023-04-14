#!/usr/bin/env bash
set -x

python test.py --loadckpt ./checkpoint_DualBranch_kitti.ckpt --dataset kitti \
        --filelist ./filenames/kitti12_test.txt --datapath /path/to/kitti2012/ \
        --savepath ./
