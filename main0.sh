#!/bin/bash

K=$1;
Iterations=$2;
Seeds=$3;
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

#TaskName="WDC/wdc_shoes_title_medium"
#SourceTask="WDC/wdc_shoes_title_medium"

#TaskName="WDC/wdc_cameras_title_medium"
#SourceTask="WDC/wdc_cameras_title_medium"

#TaskName="Structured/Amazon-Google"
#SourceTask="Structured/Amazon-Google"

SourceTask="Structured/Walmart-Amazon"
TaskName="Structured/Walmart-Amazon"

#TaskName="Structured/DBLP-ACM"
#SourceTask="Structured/DBLP-ACM"

#TaskName="Textual/Abt-Buy"
#SourceTask="Textual/Abt-Buy"

# Possible modes: "battleships_ws_b_alpha=?" (where ? is the value of alpha), "random",  "all_D", "top_k_DTAL",


# More options:

# (1)   "*mode*/only_selected". In this option the model is trained only with the final set of selected samples,
#       ignoring ones from D'. It can be used only if we already trained before a model with "*mode*",
#       for example "top_k_threshold/only_selected"
# (2)   "*mode*/D_rep". In each iteration samples from (D') are removed (determined according to
#       replace_param in TopKSelection)

# For "all_D" and "*mode*/only_selected" set Iterations = 0


Modes=("battleships_ws_k_alpha=1.0_beta=1.0" \
"battleships_ws_k_alpha=0.5_beta=0.5" "dummy")







#InputPath="data/er_magellan/Structured/Amazon-Google/"
#OutputPath="output/er_magellan/Structured/Amazon-Google/Amazon-Google/"

InputPath="data/er_magellan/Structured/Walmart-Amazon/"
OutputPath="output/er_magellan/Structured/Walmart-Amazon/Walmart-Amazon/"

#InputPath="data/wdc/shoes/title/"
#OutputPath="output/wdc/shoes/title/shoes/"

#InputPath="data/wdc/cameras/title/"
#OutputPath="output/wdc/cameras/title/shoes/"

#InputPath="data/er_magellan/Structured/DBLP-ACM/"
#OutputPath="output/er_magellan/Structured/DBLP-ACM/DBLP-ACM/"

#InputPath="data/er_magellan/Textual/Abt-Buy/"
#OutputPath="output/er_magellan/Textual/Abt-Buy/Abt-Buy/"


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
