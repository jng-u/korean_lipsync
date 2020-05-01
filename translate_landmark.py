import os
import sys
import math
import cv2
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

def to_shape(file, dtype='int'):
    coords = np.zeros(shape=(20, 2), dtype=dtype)
    i=0
    while True:
        line = file.readline()
        if not line: break
        point = line.split()
        coords[i] = (int(point[0]), int(point[1]))
        i+=1

    return coords

def translate(start, end, duration):
    shape = np.zeros(shape=(20, 2), dtype='int')
    for iter in range(duration+1):
        landmark = np.zeros(shape=(256,256), dtype='uint8')
        for i in range(len(start)):
            shape[i] = (start[i][0] + int((end[i][0]-start[i][0])*iter/duration), start[i][1] + int((end[i][1]-start[i][1])*iter/duration))
        cv2.polylines(landmark, [shape[:12]], True, (255, 255, 255), 2)
        cv2.polylines(landmark, [shape[12:]], True, (255, 255, 255), 2)
        yield landmark

# flist = glob.glob(input_folder+'/**/*.txt', recursive=True)
# for file in flist:
    # print("reading file: %s" % file)
start = open('data/trans/land/landmark/34.jpg.txt', 'r')
end = open('data/trans/land/landmark/196.jpg.txt', 'r')

start = to_shape(start)
end = to_shape(end)

ls = translate(start, end, 10)

fig = plt.figure(figsize=(11, 2))
axs = fig.subplots(ncols=11)
for i, land in enumerate(ls):
    land = cv2.cvtColor(land, cv2.COLOR_GRAY2RGB)
    axs[i].imshow(land)
    axs[i].set_title(i)
    axs[i].axis('off')
fig.savefig('data/trans/test/000.png')
plt.show()
plt.close()

# for landmark in ls:
#     cv2.imshow('landmark', landmark)
#     while(True):
#         if(cv2.waitKey(10) != -1):
#             break


# wpath = output_folder+file[len(input_folder):]
# os.makedirs(os.path.dirname(wpath), exist_ok=True)

# cv2.imwrite(img_folder+'/{}'.format(os.path.basename(file)) , croped_img)
# cv2.imshow('source', source)
# cv2.imshow('target', target)
# cv2.imshow('landmark', landmark)
# while(True):
#     if(cv2.waitKey(10) != -1):
#         break