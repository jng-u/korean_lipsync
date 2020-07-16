import os
import sys
import glob
import argparse
import math

import cv2
import dlib
from imutils import face_utils

import numpy as np
import matplotlib.pyplot as plt

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
landdata_folder = output_folder+'/txt'
os.makedirs(landdata_folder, exist_ok=True)

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

def get_rotate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan(dy/dx))

def get_rotate_point(pivot, rotate_matrix, point_list):
    rotated_point = np.zeros(point_list.shape, point_list.dtype)
    for i in range(len(rotated_point)):
        rotated_point[i] = pivot + np.dot(rotate_matrix, point_list[i]-pivot)
    return rotated_point

def middle_point(s, e):
    return (int(round((s[0]+e[0])/2)), int(round((s[1]+e[1])/2)))

def get_face(img):
    ##### 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식 향상을 위해 Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive = clahe.apply(gray)
    ratio = 1/5
    adaptive_resize = cv2.resize(adaptive, None, fx=ratio, fy=ratio)

    #### detect face and shape
    # detects whole face
    rects = detector(adaptive_resize, 1)
    if len(rects) == 0:
        print("error finding face at file: %s" % file)
        return []

    rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    rect = dlib.rectangle(x-10, y-10, x+w+10, y+h+10)   
    
    # 얼굴이 하나임을 가정
    shape = predictor(image=adaptive, box=rect)
    shape = shape_to_np(shape)
    return shape

file_list = glob.glob(input_folder+'/**/*.*', recursive=True)
file_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
for i, file in enumerate(file_list):
    print("{} / {}".format(i+1, len(file_list)))
    img = cv2.imread(file)

    shape = get_face(img)
    if len(shape)==0:
        continue

    ##### rotation 처리 
    # 광대를 수평에 맞추도록 이미지를 rotation
    # rotate_matrix = cv2.getRotationMatrix2D((tuple(shape[0])), get_rotate_angle(shape[48], shape[54]), 1)
    rotate_matrix = cv2.getRotationMatrix2D((tuple(shape[0])), get_rotate_angle(shape[2], shape[14]), 1)
    rotated_img = cv2.warpAffine(img, rotate_matrix, tuple(np.flip(img.shape[:2])))

    # 0~16 : 턱(2~14:광대 아래쪽) / 31~35 : 아래코 / 48~67 : mouth
    # 얼굴 하관 점들의 좌표

    # 회전된 shape에서 랜드마크 구하기
    rotated_shape = get_rotate_point(shape[0], rotate_matrix[:, :2], shape)    
    # under_shape = np.concatenate((rotated_shape[2:15], np.flip(rotated_shape[31:36], axis=0)), axis=0)
    under_shape = rotated_shape[2:15]
    # outer_mouth_shape = rotated_shape[48:60]
    # inner_mouth_shape = rotated_shape[60:68]
    # under_shape = np.concatenate((rotated_shape[2:15], [middle_point(rotated_shape[51], rotated_shape[33])][0:1]), axis=0)
    
    ##### make landmark image and training image
    (x, y, w, h) = get_rect(under_shape)
    for s in rotated_shape:
        s -= [x, y]
    # for s in inner_mouth_shape:
    #     s -= [x, y]
    # for s in under_shape:
    #     s -= [x, y]
    # landmark_img = np.zeros((h, w))
    # cv2.polylines(landmark_img, [under_shape], False, (255, 255, 255), 2)
    # cv2.polylines(landmark_img, [outer_mouth_shape], True, (255, 255, 255), 1)
    # cv2.polylines(landmark_img, [inner_mouth_shape], True, (255, 255, 255), 1)

    # masking underface
    croped_img = rotated_img[y:y+h, x:x+w]

    ratio = [256/w, 256/h]
    rotated_shape[:,0] = np.dot(rotated_shape[:,0], 256/w)
    rotated_shape[:,1] = np.dot(rotated_shape[:,1], 256/h)
    # inner_mouth_shape[:,0] = np.dot(inner_mouth_shape[:,0], 256/w)
    # inner_mouth_shape[:,1] = np.dot(inner_mouth_shape[:,1], 256/h)
    # under_shape[:,0] = np.dot(under_shape[:,0], 256/w)
    # under_shape[:,1] = np.dot(under_shape[:,1], 256/h)
    outer_mouth_shape = rotated_shape[48:60]
    inner_mouth_shape = rotated_shape[60:68]
    # under_shape = np.concatenate((rotated_shape[2:15], [middle_point(rotated_shape[51], rotated_shape[33])][0:1]), axis=0)

    landmark_img = np.zeros(shape=(256, 256))
    cv2.polylines(landmark_img, [outer_mouth_shape], True, (255, 255, 255), 2)
    cv2.polylines(landmark_img, [inner_mouth_shape], True, (255, 255, 255), 2)
    croped_img = cv2.resize(croped_img, None, fx=ratio[0], fy=ratio[1])

    f = open(landdata_folder+'/{}.txt'.format(os.path.basename(file)[0:len(os.path.basename(file))-4]), 'w')
    data = '{} {}\n'.format(w, h)
    f.write(data)
    for s in rotated_shape:
        data = '{} {}\n'.format(s[0], s[1])
        f.write(data)
    # for s in outer_mouth_shape:
    #     data = '{} {}\n'.format(s[0], s[1])
    #     f.write(data)
    # for s in inner_mouth_shape:
    #     data = '{} {}\n'.format(s[0], s[1])
    #     f.write(data)
    f.close()

    cv2.imwrite(landmark_folder+'/{}'.format(os.path.basename(file)) , landmark_img)
    cv2.imwrite(img_folder+'/{}'.format(os.path.basename(file)) , croped_img)
    # cv2.imshow('landmark', landmark_img)
    # cv2.imshow('masked', croped_img)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break