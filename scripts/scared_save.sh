#!/usr/bin/env bash
set -x

python test.py --CV --fold_num 0 --loadckpt ./checkpoint_DualBranch_scared_Loo1.ckpt --dataset scared \
        --filelist ./filenames/scared.txt --datapath /path/to/scared/ \
        --savepath ./
