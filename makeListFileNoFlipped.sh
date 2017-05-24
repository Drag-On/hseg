#! /bin/bash

inFile=$1
outFile=$2

while IFS='' read -r line || [[ -n "$line" ]]; do
	if [[ $line != *"FLIP"* ]]; then
		echo "$line" >> "$outFile"
	fi
done < "$inFile"