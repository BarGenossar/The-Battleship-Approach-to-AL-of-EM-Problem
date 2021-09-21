#!/bin/bash

# The size of training samples batch
K=$1;
# Number of training batches
Iterations=$2;

Seeds=$3;

TaskName="Structured/Amazon-Google"
SourceTask="Structured/Walmart-Amazon"

# Possible modes: "random", "top_k_threshold", "top_k_cliques", "all_D"
#

# More options:

# (1) "*mode*/only_selected". In this option the model is trained only with the final set of selected samples,
#    ignoring ones from D'. It can be used only if we already trained before a model with "*mode*",
#    for example "top_k_threshold/only_selected"
# (2) "*mode*/D_rep". In each iteration samples from (D') are removed (determined according to
#     replace_param in TopKSelection)

# For "all_D" and "*mode*/only_selected" set Iterations = 0

Mode="top_k_threshold/D_rep"

InputPath="data/er_magellan/Structured/Amazon-Google/"
OutputPath="output/er_magellan/Structured/Amazon-Google/Walmart-Amazon/"
LM="roberta"
training_type="active_learning"
criterion_type="pagerank"

#TaskName="Structured/Amazon-Google"
#SourceTask="Structured/Walmart-Amazon"
#Mode="top_k"
#InputPath="data/wdc/shoes/title/"
#OutputPath="output/wdc/shoes/title/computers/"
#LM="roberta"
#training_type="active_learning"
#criterion_type="pagerank"

declare -i Intent=0
declare -i MaxLen=512
declare -i Batch=12
declare -i N_Epochs=15



for (( seed=1; seed<=Seeds; seed++ ))
  do
    for (( iter=0; iter<=Iterations; iter++ ))
    do
#        python email_sender.py \
#                --message="Demo Iteration Started. seed = ${seed} / ${Seeds}, iteration = ${iter} / ${Iterations}"
        echo Iteration: $iter

        python training_samples_selection.py \
                --task=${TaskName} \
                --source_task=${SourceTask} \
                --intent=${Intent} \
                --k_size=${K} \
                --iter_num=$iter \
                --mode=${Mode} \
                --seed=$seed \
                --criterion=${criterion_type} \
                --output_path=${OutputPath}

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
                --iter_num=$iter \
                --mode=$Mode
    done
  done

#python email_sender.py \
#        --message="Run ${TaskName} Finished"