#!/usr/bin/env bash

DATA_BIN=/projects/tir3/users/pmichel1/data-bin/wmt14.en-fr.joined-dict.newstest2009-13
MODEL=/projects/tir3/users/pmichel1/checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt
EXTRA_OPTIONS=$1

python fairseq/prune.py \
     \
    -s en \
    -t fr \
    --restore-file  \
    -a transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings \
    --normalize-by-layer \
    --reset-optimizer \
    --batch-size 16 \
    --reset-optimizer \
    --beam 5 --lenpen 1 --remove-bpe "@@ " --raw-text $EXTRA_OPTIONS \
    --no-progress-bar

