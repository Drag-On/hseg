#!/bin/bash

#------------------------------------------------------------------------------
# Distribute jobs to all available machines until we have either no more
# available machine or no more jobs to schedule
#
function Distribute()
{
  # schedule jobs at all the available machines
  for i in Available/*; do  

    echo "Looking for a new job for machine $i"

    # check whether there are still any job to schedule
    if [ ! "$(ls -A Queue)" ]; then
      echo "No more jobs to schedule."
      break
    fi

    # Remove the directory before the filename
    i=${i##*/}

    MACHINE=${i%.*}
    QUEUE=${i#*.}

    # check whether the current machine can be reached
    if ! ssh -o ConnectTimeout=1 -o ConnectionAttempts=1 "$MACHINE" exit; then
      echo "Machine $MACHINE could not be reached."
      continue
    fi

    # select the first job in the queue
    JOBID=$(find Queue/ -maxdepth 1 -type f -name '*' | head -1)
    JOBID=${JOBID##*/}

    echo "Job to schedule: $JOBID"
	
    # mark the current machine as occupied
    rm -f "Available/$i"
    echo "$JOBID" > "Occupied/$i"
	
    # mark the given job as scheduled
    tries=0
    while ! mv "Queue/$JOBID" Scheduled/
    do
      echo -e "Couldn't move Queue/$JOBID to Scheduled/. Trying again in 5 seconds..."
      sleep 5
      ((tries++))
      if [[ $tries -eq 5 ]]; then
        echo -e "Moving Queue/$JOBID to Scheduled/ failed."
        exit 2
      fi
    done
    
    # execute the given job
    ssh -o StrictHostkeyChecking=no "$MACHINE" screen -d -m "$PWD/execJob.sh $PWD $JOBID $i" > /dev/null 2>&1
    
    echo -e "SCHEDULE: Job $JOBID is scheduled at $MACHINE in queue $QUEUE"
  done
}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Check whether the distributed jobs are still alive, i.e. they are running
#
function Check()
{
  numFilesQueue=$(find Queue/ -maxdepth 1 -type f -printf "." | wc -c)
  numFilesScheduled=$(find Scheduled/ -maxdepth 1 -type f -printf "." | wc -c)
  if [[ $numFilesQueue -eq 0 && $numFilesScheduled -eq 0 && $start -eq 0 ]]; then
    start=$SECONDS
  fi
  elapsed=$((SECONDS - start))
  if [[ $elapsed -ge 1200 ]]; then
    # More than 20 minutes have passed since Queue/ and Scheduled/ are empty
    start=0
    echo -e "TIMEOUT"
    exit 1
  fi
}
#------------------------------------------------------------------------------

shopt -s nullglob

start=0

# scheduling
while : ; do

  if [[ "$(ls -A Queue)" ]]; then
    Distribute
  fi

  sleep 1

  #Check

  # check whether there are still any job to execute
  if [ ! "$(ls -A Queue)" ] && 
     [ ! "$(ls -A Scheduled)" ]; then
    break
  fi

done

shopt -u nullglob
