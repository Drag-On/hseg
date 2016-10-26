#!/bin/bash

exec >  >(tee -ia out.log)      # Log stdout to out.log
exec 2> >(tee -ia err.log >&2)  # Log stderr to err.log

startIter=$1

if [ -z "${1+x}" ]; then
  # $1 was unset
  annotate-output echo "Starting from iteration 0."
  startIter=0
else
  annotate-output echo "Starting from iteration $1."
  startIter=$1
fi

for m in $(seq "$startIter" 500); do
  annotate-output echo "Iteration $m"

  annotate-output ./schedule.sh
  if [ $? -ne 0 ]; then
    annotate-output >&2 echo -e "Something has been wrong in the schedule: $m"
    exit
  fi

  annotate-output ./hseg_train_dist_merge -t "$m"
  if [[ $? -ne 0 ]]; then
    annotate-output >&2 echo -e "Merge was unsuccessful."
    exit
  fi

  mv Ready/* Queue/
  rm -r /remwork/atcremers65/moellerj/training/results/*
  mkdir /remwork/atcremers65/moellerj/training/results/labeling
  mkdir /remwork/atcremers65/moellerj/training/results/sp

  touch iter
  echo "$m" > iter
done
