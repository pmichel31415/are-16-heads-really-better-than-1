#!/usr/bin/env bash

TASK=$1
EVAL=${EVAL:-"1"}
TRAIN_OPTIONS="${@:2}"
FEATURE_MODE=${FEATURE_MODE:-0}
NODROP_MODE=${NODROP_MODE:-0}

prefix="$TASK"
if [ "$FEATURE_MODE" -eq "1" ]
then
    prefix="${TASK}-feature"
    TRAIN_OPTIONS="$TRAIN_OPTIONS --feature_mode"
fi
if [ "$NODROP_MODE" -eq "1" ]
then
    prefix="${TASK}-nodrop"
    TRAIN_OPTIONS="$TRAIN_OPTIONS --attn_dropout 0.0"
fi

mkdir -p models
model_dir=models/$prefix
mkdir -p $model_dir

function run_train () {
    python pytorch-pretrained-BERT/examples/run_classifier.py $TRAIN_OPTIONS \
    --task_name $TASK \
    --do_train \
    --do_lower_case \
    --data_dir glue_data/$TASK/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $model_dir 2>&1
}

function run_eval () {
    python pytorch-pretrained-BERT/examples/run_classifier.py \
    --task_name $TASK \
    --do_eval \
    --do_lower_case \
    $1 \
    --data_dir glue_data/$TASK/ \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --eval_batch_size 32 \
    --output_dir $model_dir 2>&1
}

if [ ! -e $model_dir/pytorch_model.bin ]
then
    run_train
fi

metric="eval_accuracy"
if [ $TASK == "CoLA" ]
then
    metric="Matthew"
elif [ $TASK == "MRPC" ]
then
    metric="F-1"
fi

if [ "$EVAL" = "1" ]
then
    run_eval ""
    base_acc=$(run_eval "" | grep $metric | rev | cut -d" " -f1 | rev)
fi
