#!/bin/bash

read -p "gpu: " GPU
read -p "epochs: " EPOCHS
read -a LRS -p "lr: "
read -a BBLRS -p "backbone lr: "
read -a KS -p "Ks: "

en="${EPOCHS}v2"

for ks in ${KS[@]}
do
	for lr in ${LRS[@]}
	do
		for bblr in ${BBLRS[@]}
		do
			python3 main.py --cuda \
				--dataset hotels_small \
				--backbone resnet50 \
				--gpu_ids $GPU  \
				--workers 10 \
				--pin_memory \
				--batch_size 60 \
				--epochs $EPOCHS \
				--learning_rate $lr \
				--bb_learning_rate $bblr \
				--num_inst_per_class $ks \
				--extra_name $en \
				--loss pnpp \
				--project_path ./ \
				--temperature 1 \
        --emb_size 512 \
        #--
		done
	done
done