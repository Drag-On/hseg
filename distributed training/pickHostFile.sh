#! /bin/bash

curDate=$(date -R)
weekday=${curDate%%,*}
hour=${curDate%%:*}
hour=${hour##* }

if [[ $weekday == "Sun" || $weekday == "Sat" ]]; then
	echo "weekend"
elif [[ $((10#$hour)) -ge 22 || $((10#$hour)) -le 5 ]]; then
	echo "night"
else
	echo "workhours"
fi