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

cnt=0
file_list = glob.glob(input_folder+'/**/*.*', recursive=True)
file_list.sort()
for file in file_list:
    print("reading file: %s" % file)
    img = cv2.imread(file)
    h = img.shape[0]
    if h > 360:
        ratio = 360/h
        img = cv2.resize(img, None, fx=ratio, fy=ratio)

    # flip = cv2.flip(img, 1)

    # (h, w) = img.shape[:2]
    # rotate_matrix = cv2.getRotationMatrix2D((w/2, h/2), 15, 1)
    # rotate = cv2.warpAffine(img, rotate_matrix, tuple(np.flip(img.shape[:2])))

    # rotate_flip = cv2.flip(rotate, 1)

    # cv2.imshow('aa', rotate)
    # cv2.imshow('aas', flip)
    # cv2.imshow('aasss', rotate_flip)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
    
    # cv2.imwrite(output_folder+'/{}.jpg'.format(cnt*4) , img)
    # cv2.imwrite(output_folder+'/{}.jpg'.format(cnt*4+1) , flip)
    # cv2.imwrite(output_folder+'/{}.jpg'.format(cnt*4+2) , rotate)
    # cv2.imwrite(output_folder+'/{}.jpg'.format(cnt*4+3) , rotate_flip)
    cv2.imwrite(output_folder+'/{}.jpg'.format(cnt) , img)
    cnt += 1
