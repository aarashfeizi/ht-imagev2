#!/bin/bash

read -p "gpu: " GPU
read -p "epochs: " EPOCHS
read -a LRS -p "lrs: "
read -a LRR -p "lrr: "
read -a LPK -p "lpk: "
read -a BHK -p "bhk: "

en="${EPOCHS}-nf"

for bhk in ${BHK[@]}
do
	for lr in ${LRS[@]}
	do
		for bblr in ${LRR[@]}
		do
			python3 main.py --cuda \
				--dataset new_hotels_small \
				--backbone resnet50 \
				--gpu_ids $GPU  \
				--workers 10 \
				--pin_memory \
				--batch_size 60 \
				--epochs $EPOCHS \
				--learning_rate $lr \
				--bb_learning_rate $bblr \
				--num_inst_per_class $bhk \
				--extra_name $en \
				--loss pnpp \
				--project_path ./ \
				--temperature 1 \
        --emb_size 512 \

				-es 20 \
				-ls 4 \

				--metric cosine \


				-pool spoc \
				-soft \
				-sr \
				-katn \

				--link_prediction_k 0 \
				--no_final_network \

		done
	done
done