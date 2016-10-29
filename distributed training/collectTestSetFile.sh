#! /bin/bash

imageDir="/home/jan/Downloads/Pascal VOC/data/VOC2012/JPEGImages/"
trainvalFile="/home/jan/Downloads/Pascal VOC/data/VOC2012/ImageSets/Segmentation/trainval.txt"

for i in "$imageDir"*; do
	filename=$(basename "$i")
	extension="${filename##*.}"
	filename="${filename%.*}"

	if ! grep -q "$filename" "$trainvalFile"; then
		echo "$filename" >> test.txt
	fi
done