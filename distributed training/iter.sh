#!/bin/bash

exec >  >(tee -ia out.log)      # Log stdout to out.log
exec 2> >(tee -ia err.log >&2)  # Log stderr to err.log

startIter=$1
./setup_hosts.sh

if [ -z "${1+x}" ]; then
  # $1 was unset
  echo "Starting from iteration 0."
  startIter=0
else
  echo "Starting from iteration $1."
  startIter=$1
fi

for m in $(seq "$startIter" 500); do
  echo "Iteration $m"

  mergeAttempts=0
  while [[ $mergeAttempts -lt 3 ]]; do

    # Schedule jobs
    scheduleAttempts=0
    while [[ $scheduleAttempts -lt 3 ]]; do
      scheduleAttempts=$((scheduleAttempts+1))
      python -m scoop --hostfile=hosts.txt --path=/work/moellerj/training_temp/ distribute.py
      err=$?
      if [ $err -ne 0 ]; then
        echo -e "Scheduling failed in iteration $m with error code $err (attempt $scheduleAttempts)."
      else
        echo -e "Scheduling successful!"
        break
      fi
    done

    # Merge job results
    mergeAttempts=$((mergeAttempts+1))
    ./hseg_train_dist_merge -t "$m"
    err=$?
    if [[ $err -ne 0 ]]; then
      echo -e "Merge failed in iteration $m with error code $err (attempt $mergeAttempts)."
    else
      echo -e "Merge successful!"
      break
    fi
  done

  if [[ $err -ne 0 ]]; then
    echo -e "Iteration $m failed ultimately with error code $err."
    exit 1
  fi

  # Send new weights to the hosts
  while IFS='' read -r line || [[ -n "$line" ]]; do
    host=${line%% *}
    scp "$PWD/weights.dat" "$host:/work/moellerj/training_temp/"
  done < "hosts.txt"

  rm -r /remwork/atcremers65/moellerj/training_new/results/*
  mkdir /remwork/atcremers65/moellerj/training_new/results/labeling
  mkdir /remwork/atcremers65/moellerj/training_new/results/sp
  mkdir /remwork/atcremers65/moellerj/training_new/results/sp_gt

  touch iter
  echo "$m" > iter
done
