#!/bin/bash

exec >  >(tee -ia out.log)      # Log stdout to out.log
exec 2> >(tee -ia err.log >&2)  # Log stderr to err.log

startIter=$1
endIter=5000

cp -f hosts.all.txt hosts.txt
./setup_hosts.sh

if [ -z "${1+x}" ]; then
  # $1 was unset
  echo "Starting from iteration 0."
  startIter=0
else
  echo "Starting from iteration $1."
  startIter=$1
fi

for m in $(seq "$startIter" "$endIter"); do
  echo "Iteration $m"

  mergeAttempts=0
  while [[ $mergeAttempts -lt 3 ]]; do

    # Schedule jobs
    scheduleAttempts=0
    while [[ $scheduleAttempts -lt 3 ]]; do
      scheduleAttempts=$((scheduleAttempts+1))
      hostfileVer=$(./pickHostFile.sh)
      cp -f "hosts.$hostfileVer.txt" hosts.txt
      echo "Using hostfile \"$hostfileVer\""
      ./setup_hosts.sh
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

  # Empty the results folder to make space for new iteration results
  rm -r /remwork/atcremers65/moellerj/training_new_noinit_400/results/*
  mkdir /remwork/atcremers65/moellerj/training_new_noinit_400/results/labeling
  mkdir /remwork/atcremers65/moellerj/training_new_noinit_400/results/sp
  mkdir /remwork/atcremers65/moellerj/training_new_noinit_400/results/sp_gt

  touch iter
  echo "$m" > iter
done
