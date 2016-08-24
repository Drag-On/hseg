#!/bin/bash

path="."
if [ "$1" != '' ]
	then
	path=$1
fi

mkdir $path/frames

convert \( $path/unary.png -background Orange label:'unary' -gravity Center -append \) \( $path/rgb.png -background Khaki label:'rgb' -gravity Center -append \) +append $path/frames/frame0.png

for filename in $path/labeling/*
do
  filename=`basename $filename`
  convert \( $path/labeling/$filename -background Orange label:$filename -gravity Center -append \) \( $path/sp/$filename -background Khaki  label:$filename -gravity Center -append \) +append $path/frames/frame$filename
done
convert -layers OptimizePlus -delay 200 $path/frames/frame* -loop 0 -quality 100 $path/frames.gif

rm -rf $path/frames