#!/bin/bash

K=$1;
Iterations=$2;
Seeds=$3;
export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1

TaskName="WDC/wdc_shoes_title_medium"
SourceTask="WDC/wdc_shoes_title_medium"

#TaskName="WDC/wdc_cameras_title_medium"
#SourceTask="WDC/wdc_cameras_title_medium"

#TaskName="Structured/Amazon-Google"
#SourceTask="Structured/Amazon-Google"

#SourceTask="Structured/Walmart-Amazon"
#TaskName="Structured/Walmart-Amazon"

#TaskName="Structured/DBLP-GoogleScholar"
#SourceTask="Structured/DBLP-GoogleScholar"

# Possible modes: "random", "top_k_threshold", "battleships", "all_D", "top_k_DTAL", "top_k_DTAL_without"
# top_k_DTAL_without is the same as top_k_DTAL without the weak supervision

# More options:

# (1)   "*mode*/only_selected". In this option the model is trained only with the final set of selected samples,
#       ignoring ones from D'. It can be used only if we already trained before a model with "*mode*",
#       for example "top_k_threshold/only_selected"
# (2)   "*mode*/D_rep". In each iteration samples from (D') are removed (determined according to
#       replace_param in TopKSelection)

# For "all_D" and "*mode*/only_selected" set Iterations = 0

# Mode="top_k_cliques"
# Modes=("battleships_ws_k" "battleships_ws_b" "battleships" "top_k_DTAL" "random" "all_D")

Modes=("battleships_ws_b_alpha=0.0" "battleships_ws_b_alpha=0.25" \
"battleships_ws_b_alpha=0.5" "battleships_ws_b_alpha=0.75" "battleships_ws_b_alpha=1.0" \
"dummy")



#InputPath="data/er_magellan/Structured/Amazon-Google/"
#OutputPath="output/er_magellan/Structured/Amazon-Google/Amazon-Google/"

#InputPath="data/er_magellan/Structured/Walmart-Amazon/"
#OutputPath="output/er_magellan/Structured/Walmart-Amazon/Walmart-Amazon/"

InputPath="data/wdc/shoes/title/"
OutputPath="output/wdc/shoes/title/shoes/"

#InputPath="data/wdc/cameras/title/"
#OutputPath="output/wdc/cameras/title/cameras/"

#InputPath="data/er_magellan/Structured/DBLP-GoogleScholar/"
#OutputPath="output/er_magellan/Structured/DBLP-GoogleScholar/DBLP-GoogleScholar/"


LM="roberta"
training_type="active_learning"
criterion_type="pagerank"

declare -i Intent=0
declare -i MaxLen=512
declare -i Batch=12
declare -i N_Epochs=12


for (( seed=1; seed<=Seeds; seed++ ))
do
  for Mode in ${Modes[*]}:
  do
    if [ $Mode = "dummy" ]
    then
        break
    fi

    for (( iter=0; iter<=Iterations; iter++ ))
    do
        if [ $iter -ge 1 ] && [ $Mode = "all_D" ]
        then
          echo exit
          break
        fi

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
                --save_model \
                --mode=$Mode

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
