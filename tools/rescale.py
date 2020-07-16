import os
import glob
import cv2
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

file_list = glob.glob(input_folder+'/**/*.*', recursive=True)
file_list.sort()
total = len(file_list)
# scale = 1/2
for i, file in enumerate(file_list):
    # print("reading file: %s" % file)
    if (i+1)%50 == 0:
        print('{}/{}'.format(i+1, total))
    img = cv2.imread(file)

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # img = cv2.resize(img, None, fx=scale, fy=scale)

    wpath = output_folder+file[len(input_folder):]
    os.makedirs(os.path.dirname(wpath), exist_ok=True)
    cv2.imwrite(wpath, img)