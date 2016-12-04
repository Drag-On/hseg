#! /bin/bash

while IFS='' read -r line || [[ -n "$line" ]]; do
    host=${line%% *}
    echo -n "Checking host \"$host\": "
    ssh -o StrictHostkeyChecking=no -x "$host" bash << EOF
    	mkdir -p /work/moellerj/training_check/ &> /dev/null ;
EOF
	scp "$PWD/hseg_train_dist_pred" "$host:/work/moellerj/training_check/" &> /dev/null
    ssh -o StrictHostkeyChecking=no -x "$host" bash << EOF
        /work/moellerj/training_check/hseg_train_dist_pred &> /dev/null ;
        echo $? ;
        rm -r /work/moellerj/training_check/  &> /dev/null ;
EOF
done < "$1"