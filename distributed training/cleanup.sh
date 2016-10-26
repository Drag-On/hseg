#!/bin/bash
mv Occupied/* Available/
rm Ready/*
rm Scheduled/*
rm Queue/*
rm -r results/*
mkdir results/labeling/
mkdir results/sp
rm success.txt
rm out.log
rm err.log
rm weights.dat
rm iter
rm training_energy.txt
rm -r weights/*

cd Queue/
../createJobs.sh
cd ..