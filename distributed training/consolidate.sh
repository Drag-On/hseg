#!/bin/bash

rm Queue/*
rm Scheduled/*
rm Ready/*
#rm Occupied/*
#rm Available/*
mv -f Occupied/* Available

cd Queue
../createJobs.sh
#cd ../Available
#../createMachines.sh
cd ..
