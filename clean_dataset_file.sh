#! /bin/bash

rm "$2" >& /dev/null

while read p; do 
    filename=${p%%;*}
    echo "$filename" | tee -a "$2"
done < "$1"