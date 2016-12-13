#! /bin/bash

while IFS='' read -r line || [[ -n "$line" ]]; do
    host=${line%% *}
    echo "Set up host \"$host\""
    ssh -o StrictHostkeyChecking=no -x "$host" bash << EOF
    	mkdir -p /work/moellerj/training_temp/ ;
    	rm -r /work/moellerj/training_temp/* ;
EOF
    scp "$PWD/hseg_train_dist_pred" "$PWD/distribute.py" "$PWD/train.txt" "$host:/work/moellerj/training_temp/"
done < "hosts.txt"