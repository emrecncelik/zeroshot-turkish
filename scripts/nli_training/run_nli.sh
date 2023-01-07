#!/bin/bash
source $1

declare -a MODELS=("dbmdz/convbert-base-turkish-cased" )
declare -a DATASETS=("allnli_tr" )

for MODEL in ${MODELS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        python finetune-nli.py --model $MODEL \
                            --dataset $DATASET \
                            --batch_size 16 \
                            --output_dir /home/emrecan/tez/zeroshot-turkish/models \
                            --max_train_examples 50 \
                            --max_eval_examples 50
    done
done
