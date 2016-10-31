#!/bin/bash
basePath="/remwork/atcremers65/moellerj"
imgPath="$basePath/VOC2012/JPEGImages"
gtPath="$basePath/groundTruthFixed"
gtSpPath="$basePath/groundTruthSp$"
unaryPath="$basePath/unaries"
outPath="$basePath/training/results"
weightsPath="$basePath/training"
featureWeights="$basePath/featureWeights.txt"


# Re-create all jobs
i=0
n=0
while read m; do
	touch "$i";
	echo "$basePath/training/hseg_train_dist_pred -i $imgPath/$m.jpg -g $gtPath/$m.png -gsp $gtSpPath/$m.png -u ${unaryPath}/${m}_prob.dat -o $outPath/ -w $weightsPath/weights.dat -fw $featureWeights" >> "$i"
	n=$((n+1))
	if [[ $n -eq 5 ]]; then
		i=$((i+1))
		n=0
	fi
done <"../train.txt"
