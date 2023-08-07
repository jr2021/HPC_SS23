#!/bin/bash -x

for NP in 1 4 9 16 25; 
    do 
    sbatch run.sh $NP
done