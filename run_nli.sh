#!/bin/bash
source $1

declare -a MODELS=("dbmdz/distilbert-base-turkish-cased" "dbmdz/bert-base-turkish-cased" "dbmdz/convbert-base-turkish-mc4-cased" )
declare -a DATASETS=("snli_tr" "multinli_tr" )

for MODEL in ${MODELS[@]}
do
    for DATASET in ${DATASETS[@]}
    do
        python finetune-nli.py --model $MODEL \
                            --dataset $DATASET \
                            --batch_size 32 \
                            --output_dir /home/emrecan/tez/zeroshot-turkish/models
    done
done
