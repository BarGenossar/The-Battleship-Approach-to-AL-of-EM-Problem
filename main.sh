#!/bin/bash

# The size of training samples batch
K=$1;
# Number of training batches
Iterations=$2;

Seeds=$3;

TaskName="WDC/wdc_shoes_title_medium"
SourceTask="WDC/wdc_cameras_title_medium"
#TaskName="WDC/wdc_cameras_title_medium"
#SourceTask="WDC/wdc_shoes_title_medium"
#TaskName="Structured/Amazon-Google"
#SourceTask="Structured/Walmart-Amazon"
#TaskName="Structured/Walmart-Amazon"
#SourceTask="Structured/Amazon-Google"


# Possible modes: "random", "top_k_threshold", "battleships", "all_D", "top_k_kasai", "top_k_kasai_without"
# top_k_kasai_without is the same as top_k_kasai without the weak supervision

# More options:

# (1)   "*mode*/only_selected". In this option the model is trained only with the final set of selected samples,
#       ignoring ones from D'. It can be used only if we already trained before a model with "*mode*",
#       for example "top_k_threshold/only_selected"
# (2)   "*mode*/D_rep". In each iteration samples from (D') are removed (determined according to
#       replace_param in TopKSelection)

# For "all_D" and "*mode*/only_selected" set Iterations = 0

# Mode="top_k_cliques"
# Modes=("battleships_ws_k" "battleships_ws_b" "battleships" "top_k_Kasai" "random" "all_D")
Modes=("battleships_ws_b_alpha=0.1" "battleships_ws_b_alpha=0.3" "battleships_ws_b_alpha=0.5" "battleships_ws_b_alpha=0.7" \
"battleships_ws_b_alpha=0.9" "dummy")
#Mode="all_D"
#InputPath="data/er_magellan/Structured/Amazon-Google/"
#OutputPath="output/er_magellan/Structured/Amazon-Google/Walmart-Amazon/"
InputPath="data/wdc/shoes/title/"
OutputPath="output/wdc/shoes/title/cameras/"
#InputPath="data/wdc/cameras/title/"
#OutputPath="output/wdc/cameras/title/shoes/"


LM="roberta"
training_type="active_learning"
criterion_type="pagerank"

declare -i Intent=0
declare -i MaxLen=512
declare -i Batch=12
declare -i N_Epochs=15


for Mode in ${Modes[*]}:
do
  for (( seed=1; seed<=Seeds; seed++ ))
  do
    for (( iter=0; iter<=Iterations; iter++ ))
    do

        echo Iteration: $iter
        echo Mode: $Mode
        echo Seed: $seed

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
done
