#! /bin/bash

# $1: Folder of cityscapes dataset
# $2: Output folder
# $3: Listfile output relative to $2

rm "$2/$3" >& /dev/null

outRGBFolderName="rgb"
outGTFolderName="gt"

imageFolder="images"
labelsFolder="labels"

imgExt="png"
imgFilenameRem="_leftImg8bit"

gtExt="png"
gtFilenameRem="_gtFine_labelIds"

for f in "$1/$imageFolder"/*"$imgFilenameRem"."$imgExt"; do
	filename="$(basename "$f")"
	filename=${filename%"$imgFilenameRem"."$imgExt"}
	if [[ -f "$1/$labelsFolder/$filename$gtFilenameRem.$gtExt" ]]; then
		echo "$filename is valid."
		cp "$1/$imageFolder/$filename$imgFilenameRem.$imgExt" "$2/$outRGBFolderName/$filename.$imgExt"
		cp "$1/$labelsFolder/$filename$gtFilenameRem.$gtExt" "$2/$outGTFolderName/$filename.$gtExt"
		echo "$filename" >> "$2/$3"
	else
		echo "$filename is invalid. $1/$labelsFolder/$filename$gtFilenameRem.$gtExt doesnt exist. Skipping."
	fi
done