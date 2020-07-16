import os
import sys
import glob
import argparse
import math

import cv2

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input1', dest='input_folder1', type=str)
parser.add_argument('--input2', dest='input_folder2', type=str)
args = parser.parse_args()

input_folder1 = args.input_folder1
input_folder2 = args.input_folder2

input_list1 = glob.glob(input_folder1+'/**/*.*', recursive=True)
input_list1.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
input_list2 = glob.glob(input_folder2+'/**/*.*', recursive=True)
input_list2.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
sum = 0
for i, img in enumerate(input_list1):
    img = cv2.imread(img)
    original = img
    convert = cv2.imread(input_list2[i])
    h, w, c = img.shape

    # original = np.zeros((h, int(w/2)))
    # convert = np.zeros((h, int(w/2)))
    # original = img[:, :int(w/2)]
    # convert = img[:, int(w/2):w]
    # w/=2


    original=original/255
    convert=convert/255
    dist = math.sqrt(np.sum((original-convert)**2) / (h*w)) *100    
    print(dist)
    sum+=dist
sum/=len(input_list1)    
print(sum)
# 0.6096891418097973
# 4.059025405956506
