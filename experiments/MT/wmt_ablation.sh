#!/usr/bin/env bash

source env/bin/activate

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_BIN=/projects/tir3/users/pmichel1/data-bin/wmt14.en-fr.joined-dict.newstest2013
MODEL=/projects/tir3/users/pmichel1/checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt
MOSES_SCRIPTS="~/wd/mosesdecoder/scripts"
OUT_DIR=output
SRC_FILE=${"dev/newstest2013.bpe.en":-1}
REF_FILE=${"dev/newstest2013.fr":-2}
OUT_PREFIX=${"newstest2013":-3}
EXTRA_OPTIONS=${"":-3}
# Use the following instead for ablating all but one head in a layer
# OUT_PREFIX=newstest2013.allbut
# EXTRA_OPTIONS="--transformer-mask-all-but-one-head"

mkdir -p $OUT_DIR
# Compute base BLEU

cat $SRC_FILE | python fairseq/interactive.py \
    $DATA_BIN \
    --path $MODEL \
    "--beam 5 --lenpen 1.0 --buffer-size 100 --batch-size=64" |\
    grep "^H" | sed -r 's/(@@ )|(@@ ?$)//g' |\
    perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl -q -l fr | cut -f3 |\
    > $OUT_DIR/${OUT_PREFIX}.out.fr
base_bleu=$(cat $OUT_DIR/${OUT_PREFIX}.out.fr | sacrebleu $REF_FILE | cut -d" " -f3)
echo $base_bleu

# Iterate over the 3 "parts" of the model, Enc-Enc (E), Enc-Dec (A) and Dec-Dec (D)
for part in "E" "A" "D"
do
    # Ablate all but one head in each layer iteratively
    echo $part
    for layer in `seq 1 6`
    do
        echo -n "$layer"
        for head in `seq 1 16`
        do
            mask_str="${part}:${layer}:${head}"
            cat $SRC_FILE | python fairseq/interactive.py \
                $DATA_BIN \
                --path $MODEL \
                "--beam 5 --lenpen 1.0 --buffer-size 100 --batch-size=64 --transformer-mask-heads $mask_str $EXTRA_OPTIONS" |\
                grep "^H" | sed -r 's/(@@ )|(@@ ?$)//g' |\
                perl $MOSES_SCRIPTS/tokenizer/detokenizer.perl -q -l fr | cut -f3 |\
                > $OUT_DIR/${OUT_PREFIX}.${mask_str}.out.fr
            bleu=$(cat $OUT_DIR/${OUT_PREFIX}.${mask_str}.out.fr | sacrebleu $REF_FILE | cut -d" " -f3)
            printf "\t%.2f" $(echo "$bleu - $base_bleu" | bc )
        done
        echo ""
    done
done
