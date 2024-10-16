#!/bin/bash


read -r -p "dataset yaml dir, a directoyu path to the dataset.yaml / train / val / test : " dataset_yaml

read -r -p "Output dir: " output_dir

read -r -p "model path, a directory path to the .pt model: " model_path



docker run -it --rm --gpus 0 \
	--mount type=bind,source=${dataset_yaml},target=/pruner/dataset_yaml \
	--mount type=bind,source=${output_dir},target=/pruner/output \
	--mount type=bind,source=${model_path},target=/pruner/models \
	hw_max_ira \
    run_pruner -ds /pruner/dataset_yaml -m /pruner/models -o /pruner/output