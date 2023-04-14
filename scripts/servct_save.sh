#!/usr/bin/env bash
set -x

python test.py --CV --fold_num 1 --loadckpt ./checkpoint_DualBranch_servct_Loo1.ckpt --dataset servct \
        --filelist ./filenames/servct.txt --datapath /path/to/SERVCT \
        --savepath ./
