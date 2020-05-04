import os
import sys
import cv2
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_folder+'/test.avi', fourcc, 20.0, (1280, 720))

flist = glob.glob(input_folder+'/**/*.*', recursive=True)
flist.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
for file in flist:
    print("reading file: %s" % file)
    img = cv2.imread(file)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out.write(img)
    # cv2.imwrite(output_folder+'/{}'.format(os.path.basename(file)) , dst)
    # cv2.imshow('source', source)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', dst)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
out.release()
cv2.destroyAllWindows()