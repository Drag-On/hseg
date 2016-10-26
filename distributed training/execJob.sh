#!/bin/bash

# Parameters:
# 1: working directorry
# 2: ID of the current job
# 3: ID of the current machine

exec >  >(tee -ia "$1/out.log")      # Log stdout to out.log
exec 2> >(tee -ia "$1/err.log" >&2)  # Log stderr to err.log

# run the current job
# check whether the job was successful
tries=0
if nice -n 12 annotate-output bash "$1/Scheduled/$2"; then
  # mark the given job as ready
  touch "$1/Ready/$2"
  while ! mv -f "$1/Scheduled/$2" "$1/Ready/"
  do
  	echo -e "Couldn't move $1/Scheduled/$2 to $1/Ready/. Trying again in 5 seconds..."
  	sleep 5
  	((tries++))
	if [[ $tries -eq 5 ]]; then
	  echo -e "Moving $1/Scheduled/$2 to $1/Ready/ failed."
	  exit 2
	fi
  done
  annotate-output echo "$2 ready on machine $3: $? From $1/Scheduled/$2 to $1/Ready/"
else
  # put back the given job to the queue
  touch "$1/Queue/$2"
  while ! mv -f "$1/Scheduled/$2" "$1/Queue/"
  do
  	echo -e "Couldn't move $1/Scheduled/$2 to $1/Queue/. Trying again in 5 seconds..."
  	sleep 5
  	((tries++))
	if [[ $tries -eq 5 ]]; then
	  echo -e "Moving $1/Scheduled/$2 to $1/Ready/ failed."
	  exit 2
	fi
  done
  annotate-output echo "$2 back to queue, was on machine $3: $? From $1/Scheduled/$2 to $1/Queue/"
fi

# release the current machine
annotate-output mv "$1/Occupied/$3" "$1/Available/"
annotate-output echo "released $3, finished job $2"
