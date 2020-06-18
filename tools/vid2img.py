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

WANTED_FPS = 15

vidlist = glob.glob('{}/**/*.*'.format(input_folder), recursive=True)
for vid in vidlist:
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f = int(round(fps/WANTED_FPS))
    print(fps)
    print(f)
    print('start %s' %vid)
    cnt=0
    while True:
        ret, frame = cap.read()
        if cnt%f == 0:
            if frame is None:
                break
            w_path = '{}/'.format(output_folder)
            os.makedirs(w_path, exist_ok=True) 
            cv2.imwrite(w_path+'{}.jpg'.format(int(cnt/f)), frame)
        cnt+=1