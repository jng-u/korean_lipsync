import os
import glob
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

vidlist = glob.glob('{}/**/*.*'.format(input_folder), recursive=True)
for vid in vidlist:
    cap = cv2.VideoCapture(vid)

    print('start %s' %vid)
    cnt=0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        w_path = '{}/{}/'.format(output_folder, vid[len(input_folder)+1:])
        os.makedirs(w_path, exist_ok=True) 
        cv2.imwrite(w_path+'{}.jpg'.format(cnt), frame)
        cnt+=1