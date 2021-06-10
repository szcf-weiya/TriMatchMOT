#!/bin/bash

# https://stackoverflow.com/questions/687780/documenting-shell-scripts-parameters
if [ $# == 0 ]; then
  cat <<HELP_USAGE

  $0 param1 param2

  param1 number of repetitions
  param2 node label, can be stat or chpc
  param3 number of cores
HELP_USAGE
  exit 0
fi

curr_timestamp=$(date -Iseconds)
parent_folder=truth_coverage_${curr_timestamp}
mkdir $parent_folder

for i in $(seq 1 1 $1); do
  # echo $i
  
  sbatch -N 1 -c $3 -p $2 -q $2 --export=parent_folder=${parent_folder},nrep=${i},np=$3 multi-experiments.job
done
