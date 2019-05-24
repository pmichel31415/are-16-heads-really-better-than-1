#!/usr/bin/env bash
DATA_BIN=data-bin/iwslt14.tokenized.de-en
MODEL=$1
ARCH=${2:-"transformer_iwslt_de_en_8head_before"}
EXTRA_OPTIONS=$3

python fairseq/prune.py \
    $DATA_BIN \
    -s de \
    -t en \
    --restore-file $MODEL \
    --arch $ARCH \
    --normalize-by-layer \
    --reset-optimizer \
    --batch-size 64 \
    --reset-optimizer \
    --beam 5 --lenpen 1 --remove-bpe "@@ " --raw-text $EXTRA_OPTIONS \
    --no-progress-bar

