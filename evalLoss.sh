#! /bin/bash

infer_exec="build/hseg_infer_batch"
accy_exec="build/hseg_accy"
list="train.txt"
weightsDir="out/iterations"
inferenceDir="out/inference"
maxIter=1000
logfile="out/loss.log"

{
	printf "========================================\n"
	printf "Loss \"%s\"\n" "$list"
	printf "weightsDir: \"%s\"\n" "$weightsDir"
	printf "inferenceDir: \"%s\"\n" "$inferenceDir"
	printf "maxIter: \"%s\"\n" "$maxIter"
	printf "========================================\n"
} > "$logfile"


for (( i = 0; i < maxIter; i++ )); do
	# Wait until weights file exists
	while [[ ! -f $weightsDir/$i.dat ]]; do
		sleep 60
	done

	# Do inference
	yes | eval "$infer_exec -l $list -w $weightsDir/$i.dat --out $inferenceDir/" &> /dev/null
	errCode=$?
	if [[ $errCode != 0 ]]; then
		echo -e "Inference failed with error code $errCode."
		exit 1
	fi

	# Compute loss
	loss=$(eval "$accy_exec -l $list -w $weightsDir/$i.dat --in $inferenceDir/labeling/" | grep "Loss:")
	errCode=$?
	if [[ $errCode != 0 ]]; then
		echo -e "Accuracy computation failed with error code $errCode."
		exit 2
	fi
	loss=${loss#"Loss: "}

	# Show it and write it to file
	printf "$i:\t%s\n" "$loss" | tee -a "$logfile"
done
