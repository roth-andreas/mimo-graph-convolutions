#!/usr/bin/zsh

# Run the grid search for all dataset and all methods
for dataset in {texas,cornell,wisconsin,film,chameleon,squirrel}; do
  for conv_type in {gatv2,gin,fagcn,lmgc,acm}; do
    args="--dataset $dataset --conv_type $conv_type"
    python run_GNN.py ${args}
  done
done