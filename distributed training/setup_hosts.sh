#! /bin/bash

while IFS='' read -r line || [[ -n "$line" ]]; do
    host=${line%% *}
    echo "Set up host \"$host\""
    ssh -o StrictHostkeyChecking=no -x "$host" bash << EOF
    	mkdir -p /work/moellerj/training_temp/ ;
    	rm -r /work/moellerj/training_temp/* ;
    	mkdir -p /work/moellerj/training_temp/results/labeling ;
    	mkdir -p /work/moellerj/training_temp/results/sp ;
    	mkdir -p /work/moellerj/training_temp/results/sp_gt ;
EOF
	scp "$PWD/hseg_train_dist_pred" "$host:/work/moellerj/training_temp/"
	if [ -f "$PWD/weights.dat" ] ; then
		scp "$PWD/weights.dat" "$host:/work/moellerj/training_temp/"
	fi
	scp "$PWD/distribute.py" "$host:/work/moellerj/training_temp/"
	scp "$PWD/train.txt" "$host:/work/moellerj/training_temp/"
done < "hosts.txt"