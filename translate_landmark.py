import os
import sys
import math
import cv2
import numpy as np  
import glob
import argparse
import matplotlib.pyplot as plt

# landmark_path = ['../data/taylor/land/txt/0.txt', \
#             '../data/taylor/land/txt/0.txt', \
#             # '../data/taylor/land/txt/1292.txt', \
#             '../data/taylor/land/txt/1425.txt', \
#             '../data/taylor/land/txt/94.txt', \
#             # '../data/taylor/land/txt/196.txt', \
#             '../data/taylor/land/5.txt', \
#             # '../data/taylor/land/txt/1541.txt', \
#             '../data/taylor/land/txt/933.txt', \
#             '../data/taylor/land/txt/0.txt', \
#             # '../data/taylor/land/txt/924.txt']
#             '../data/taylor/land/8.txt']
landmark_path = ['../data/ljw/stdland/txt/1.txt', \
                '../data/ljw/stdland/txt/89.txt', \
                '../data/ljw/stdland/txt/3.txt', \
                '../data/ljw/stdland/txt/346.txt', \
                '../data/ljw/stdland/txt/929.txt', \
                '../data/ljw/stdland/txt/6.txt', \
                '../data/ljw/stdland/txt/1191.txt', \
                '../data/ljw/stdland/txt/8.txt']

parser = argparse.ArgumentParser()
parser.add_argument('--number', dest='num', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
parser.add_argument('--iter', dest='iteration', type=int, default=10)
args = parser.parse_args()

output_folder = args.output_folder
os.makedirs(output_folder+'/land', exist_ok = True)
os.makedirs(output_folder+'/txt', exist_ok = True)

def to_shape(file, dtype='int'):
    coords = np.zeros(shape=(69, 2), dtype=dtype)
    i=0
    while True:
        line = file.readline()
        if not line: break
        point = line.split()
        coords[i] = (int(point[0]), int(point[1]))
        i+=1

    return coords

def translate(start, end, duration):
    shapes = []
    landmarks = []
    for iter in range(duration+1):
        shape = np.zeros(shape=(69, 2), dtype='int')
        landmark = np.zeros(shape=(256,256), dtype='uint8')
        for i in range(len(start)):
            shape[i] = (start[i][0] + int(round((end[i][0]-start[i][0])*iter/duration)), start[i][1] + int(round((end[i][1]-start[i][1])*iter/duration)))
        cv2.polylines(landmark, [shape[49:61]], True, (255, 255, 255), 2)
        cv2.polylines(landmark, [shape[61:69]], True, (255, 255, 255), 2)
        landmarks.append(landmark)
        shapes.append(shape)
        # yield landmark, shape
    return landmarks, shapes

nums = args.num
iteration = args.iteration
cnt=0
for i in range(len(nums)-1):
    s = nums[i]
    e = nums[i+1]
    print('{} -> {}'.format(s, e))
    start = open(landmark_path[int(s)-1], 'r')
    end = open(landmark_path[int(e)-1], 'r')

    start = to_shape(start)
    end = to_shape(end)

    lands, shapes = translate(start, end, iteration)
    
    os.makedirs(output_folder+'/land/{}_{}'.format(s, e), exist_ok = True)
    os.makedirs(output_folder+'/txt/{}_{}'.format(s, e), exist_ok = True)

    # fig = plt.figure(figsize=(13, 2))
    # axs = fig.subplots(ncols=13)
    # if i==0:
    #     land = cv2.cvtColor(lands[0], cv2.COLOR_GRAY2RGB)
    #     cv2.imwrite(output_folder+'/land/{}.jpg'.format(cnt), land)
    #     f = open(output_folder+'/txt/{}.txt'.format(cnt), 'w')
    #     for shape in shapes[0]:
    #         data = '{} {}\n'.format(shape[0], shape[1])
    #         f.write(data)
    #     f.close()
    #     cnt+=1
    for j, land in enumerate(lands):
        # if j==0: continue
        land = cv2.cvtColor(land, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(output_folder+'/land/{}_{}/{}.jpg'.format(s, e, cnt), land)
        f = open(output_folder+'/txt/{}_{}/{}.txt'.format(s, e, cnt), 'w')
        for shape in shapes[j]:
            data = '{} {}\n'.format(shape[0], shape[1])
            f.write(data)
        f.close()
        cnt+=1
        # if j==len(lands)-1:
        #     cv2.imwrite(output_folder+'/land/{}.jpg'.format(cnt), land)
        #     f = open(output_folder+'/txt/{}.txt'.format(cnt), 'w')
        #     for shape in shapes[j]:
        #         data = '{} {}\n'.format(shape[0], shape[1])
        #         f.write(data)
        #     f.close()
        #     cnt+=1
            
        # axs[i].imshow(land)
        # axs[i].set_title(i)
        # axs[i].axis('off')
    # fig.savefig('data/trans/test/000.png')
    # plt.show()
    # plt.close()

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