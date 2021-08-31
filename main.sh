#!/bin/bash

# The size of training samples batch
K=$1;
# Number of training batches
Iterations=$2;

Seeds=$3;


TaskName="Structured/Amazon-Google"
SourceTask="Structured/Walmart-Amazon"
Mode="top_k"
InputPath="data/er_magellan/Structured/Amazon-Google/"
OutputPath="output/Structured/Amazon-Google/"
LM="roberta"
training_type="active_learning"
criterion_type="pagerank"

declare -i Intent=0
declare -i MaxLen=512
declare -i Batch=16
declare -i N_Epochs=20



for (( seed=1; seed<=Seeds; seed++ ))
  do
    for (( iter=0; iter<=Iterations; iter++ ))
    do
        echo "seed: $seed, iteration: $iter";
        python training_samples_selection.py \
                --task=${TaskName} \
                --source_task=${SourceTask} \
                --intent=${Intent} \
                --k_size=${K} \
                --iter_num=$iter \
                --mode=${Mode} \
                --seed=$seed \
                --criterion=${criterion_type}

        python train_ditto.py \
                --task=${TaskName} \
                --batch=${Batch} \
                --max_len=${MaxLen} \
                --n_epochs=${N_Epochs} \
                --training_type=${training_type} \
                --finetuning  \
                --lm=${LM} \
                --seed=$seed \
                --save_model

        python matcher.py \
                --task=${TaskName} \
                --input_path=${InputPath}  \
                --output_path=${OutputPath}  \
                --lm=${LM} \
                --max_len=${MaxLen} \
                --training_type=${training_type} \
                --seed=$seed \
                --iter_num=$iter
    done
  done