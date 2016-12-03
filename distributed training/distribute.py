# Launch with: python -m scoop --hostfile=filename distribute.py
from __future__ import print_function
from scoop import futures
from functools import partial
import subprocess
import sys
import getopt
import os
import shutil
import time

execPred="./hseg_train_dist_pred"
imgPath="/remwork/atcremers65/moellerj/dataset_small/color/"
gtPath="/remwork/atcremers65/moellerj/dataset_small/gt/"
unPath="/remwork/atcremers65/moellerj/dataset_small/unaries/"
weightsFile="/remwork/atcremers65/moellerj/training_new/weights.dat"
featureWeightsFile="/remwork/atcremers65/moellerj/training_new/featureWeights.txt"
outPath="/remwork/atcremers65/moellerj/training_new/results/"
numClusters=200
trainingFile="train.txt"

def runWorker(filename):
    #print("Run worker for", filename)
    #if not os.path.exists(outPath + "labeling/"):
    #    os.makedirs(outPath + "labeling/")
    #if not os.path.exists(outPath + "sp/"):
    #    os.makedirs(outPath + "sp/")
    #if not os.path.exists(outPath + "sp_gt/"):
    #    os.makedirs(outPath + "sp_gt/")
    #errcode = subprocess.call(['touch', filename + ".txt"])
    errcode = 128
    attempts = 0
    while errcode != 0 and attempts < 10:
        attempts = attempts + 1
        with open(os.devnull, 'w') as FNULL:
            errcode = subprocess.call([execPred, "-i", imgPath+filename+".jpg",
                "-g", gtPath+filename+".png", "-u", unPath+filename+"_prob.dat", "-o",
                outPath, "-w", weightsFile, "-fw", featureWeightsFile, "-c",
                str(numClusters)], stdout=FNULL, stderr=subprocess.STDOUT)
        if errcode != 0:
            time.sleep(1)
    #subprocess.call(['scp', outPath + "labeling/" + filename + ".png", resultsHost + ":" + resultsPath])
    #subprocess.call(['scp', outPath + "sp/" + filename + ".png", resultsHost + ":" + resultsPath])
    #subprocess.call(['scp', outPath + "sp_gt/" + filename + ".png", resultsHost + ":" + resultsPath])
    return errcode, filename, attempts

if __name__ == '__main__':
    startIter = 0
    endIter = 100
    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:i:g:u:o:w:f:n:t:")
    except getopt.GetoptError:
        print('Invalid arguments.')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-e"):
            execPred = arg
        elif opt in ("-i"):
            imgPath = arg
        elif opt in ("-g"):
            gtPath = arg
        elif opt in ("-u"):
            unPath = arg
        elif opt in ("-o"):
            outPath = arg
        elif opt in ("-w"):
            weightsFile = arg
        elif opt in ("-f"):
            featureWeightsFile = arg
        elif opt in ("-n"):
            numClusters = arg
        elif opt in ("-t"):
            trainingFile = arg

    with open(trainingFile) as f:
        lines = f.read().splitlines()

    print("{}".format(lines))

    for errcode, filename, attempts in futures.map_as_completed(runWorker, lines):
        print("Finished {} with return code {} after {} attempts.".format(filename, errcode, attempts))
