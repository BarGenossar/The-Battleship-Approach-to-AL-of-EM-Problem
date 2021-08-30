#!/bin/bash

# The size of training samples batch
K=$1;
# Number of training batches
Iterations=$2;

Seed=$3;


TaskName="WDC/wdc_shoes_title_small"
SourceTask="WDC/wdc_computers_title_small"
Mode="top_k"
InputPath="data/wdc/shoes/title/"
OutputPath="output/wdc/shoes/title/"
LM="roberta"
training_type="active_learning"
criterion_type="pagerank"

declare -i Intent=0
declare -i MaxLen=512
declare -i Batch=16
declare -i N_Epochs=3





for (( iter=0; iter<=Iterations; iter++ ))
do
    echo "iteration: " $iter;

    python training_samples_selection.py \
            --task=${TaskName} \
            --source_task=${SourceTask} \
            --intent=${Intent} \
            --k_size=${K} \
            --iter_num=$iter \
            --mode=${Mode} \
            --seed=${Seed} \
            --criterion=${criterion_type}

    python train_ditto.py \
            --task=${TaskName} \
            --batch=${Batch} \
            --max_len=${MaxLen} \
            --n_epochs=${N_Epochs} \
            --training_type=${training_type} \
            --finetuning  \
            --lm=${LM} \
            --seed=${Seed} \
            --save_model

    python matcher.py \
            --task=${TaskName} \
            --input_path=${InputPath}  \
            --output_path=${OutputPath}  \
            --lm=${LM} \
            --max_len=${MaxLen} \
            --training_type=${training_type} \
            --seed=${Seed} \
            --iter_num=$iter


done