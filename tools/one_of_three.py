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
for i, file in enumerate(file_list):
    print("reading file: %s" % file)
    img = cv2.imread(file)

    h,w = img.shape[:2]
    img = img[:, int(w/3):2*int(w/3)]

    # wpath = output_folder+file[len(input_folder):]
    # os.makedirs(os.path.dirname(wpath), exist_ok=True)
    cv2.imwrite(output_folder+os.path.basename(file), img)