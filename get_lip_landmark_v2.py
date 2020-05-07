import os
import sys
import math
import dlib
import cv2
import numpy as np
import glob
import argparse

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
landmark_folder = output_folder+'/landmark'
os.makedirs(landmark_folder, exist_ok=True)
img_folder = output_folder+'/img'
os.makedirs(img_folder, exist_ok=True)

def get_rect(shape):
    xs = shape[:, 0]
    ys = shape[:, 1]
    x = min(xs)
    y = min(ys)
    w = max(xs)-x
    h = max(ys)-y
    return (x,y,w,h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

file_list = glob.glob(input_folder+'/**/*.*', recursive=True)
for file in file_list:
    print("reading file: %s" % file)
    img = cv2.imread(file)

    ##### 전처리
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식 향상을 위해 Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive_img = clahe.apply(gray_img)

    ##### detect face and shape
    # detects whole face
    rects = detector(adaptive_img, 1)
    if len(rects) == 0:
        print("error finding face at file: %s" % file)
        continue
    
    # 얼굴이 하나임을 가정
    # rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    shape = predictor(image=adaptive_img,box=rects[0])
    shape = shape_to_np(shape)
    
    under_shape = shape[2:15]
    outer_mouth_shape = shape[48:60]
    inner_mouth_shape = shape[60:68]
    
    ##### make landmark image and training image
    (x, y, w, h) = get_rect(under_shape)
    x=x-10
    w=w+20
    h=h+20
    landmark_img = np.copy(img)
    # landmark_img[y:y+h, x:x+w] = np.zeros((h, w, 3))
    land_box = np.zeros((h, w, 3))
    under_shape = shape[0:17]
    for s in under_shape:
        s -= [x, y]
    for s in outer_mouth_shape:
        s -= [x, y]
    for s in inner_mouth_shape:
        s -= [x, y]
    cv2.polylines(land_box, [under_shape], False, (255, 255, 255), 1)
    cv2.polylines(land_box, [outer_mouth_shape], True, (255, 255, 255), 1)
    cv2.polylines(land_box, [inner_mouth_shape], True, (255, 255, 255), 1)
    landmark_img[y:y+h, x:x+w] = land_box

    # f = open(landmark_folder+'/{}.txt'.format(os.path.basename(file)), 'w')
    # for s in outer_mouth_shape:
    #     data = '{} {}\n'.format(s[0], s[1])
    #     f.write(data)
    # for s in inner_mouth_shape:
    #     data = '{} {}\n'.format(s[0], s[1])
    #     f.write(data)
    # f.close()

    cv2.imwrite(landmark_folder+'/{}'.format(os.path.basename(file)) , landmark_img)
    cv2.imwrite(img_folder+'/{}'.format(os.path.basename(file)) , img)
    # cv2.imshow('landmark', landmark_img)
    # cv2.imshow('masked', img)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break