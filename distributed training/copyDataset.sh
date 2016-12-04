#! /bin/bash

while IFS='' read -r line || [[ -n "$line" ]]; do
    host=${line%% *}
    echo "Copy dataset to host \"$host\""
	scp -r "/work/moellerj/dataset_small/" "$host:/work/moellerj/"
done < "$1"