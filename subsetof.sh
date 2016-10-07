#!/bin/bash

# Arguments:
# 1: input file
# 2: output file:
# 3: If this is 2, then every other line is removed, if it is 3, then every third line is removed, etc...

rm "$2"

i=0
while IFS='' read -r line || [[ -n "$line" ]]; do
	if [[ $i -eq $3 ]]; then
		i=0
		continue;
	fi
    echo "$line" >> "$2"
    ((i++))
done < "$1"