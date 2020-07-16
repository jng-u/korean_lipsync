import os
import sys
import cv2
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--A', dest='input_folder_A', type=str)
parser.add_argument('--B', dest='input_folder_B', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder_A = args.input_folder_A
input_folder_B = args.input_folder_B
output_folder = args.output_folder

flist_A = glob.glob(input_folder_A+'/**/*.*', recursive=True)
flist_A.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
flist_B = glob.glob(input_folder_B+'/**/*.*', recursive=True)
flist_B.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# tmp = cv2.imread(flist_A[0])
# out = cv2.VideoWriter(output_folder+'/output.avi', fourcc, 15.0, (tmp.shape[1], tmp.shape[0]))

for i, file in enumerate(flist_A):
    print("reading file: %s" % file)
    img = cv2.imread(file)
    img2 = cv2.imread(flist_B[i])

    h, w, c = img.shape

    ret = np.copy(img2)
    ret[:, 0:int(w/2)] = img[:, int(w/2):w]

    # ret = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ret = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # out.write(ret)
    cv2.imwrite(output_folder+'/66rr/{}'.format(os.path.basename(file)) , ret)
    # cv2.imshow('source', source)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', ret)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
# out.release()
cv2.destroyAllWindows()