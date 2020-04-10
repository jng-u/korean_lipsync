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
os.makedirs(output_folder, exist_ok=True)
landmark_folder = output_folder+'/landmark'
os.makedirs(landmark_folder, exist_ok=True)
crop_folder = output_folder+'/crop'
os.makedirs(crop_folder, exist_ok=True)

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

img = None
mouth = None
ratio = 1.0/2.0

file_list = glob.glob(input_folder+'/*.*', recursive=True)
for file in file_list:
    print("reading file: %s" % file)
    img = cv2.imread(file)

    ##### 전처리
    resized_img = cv2.resize(img, None, fx=ratio, fy=ratio)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

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

    ##### rotation 처리 
    # 광대를 수평에 맞추도록 이미지를 rotation
    rotate_matrix = cv2.getRotationMatrix2D((tuple(shape[0])), get_rotate_angle(shape[2], shape[14]), 1)
    rotated_img = cv2.warpAffine(resized_img, rotate_matrix, tuple(np.flip(resized_img.shape[:2])))

    # 0~16 : 턱(2~14:광대 아래쪽) / 31~35 : 아래코 / 48~67 : mouth
    # 얼굴 하관 점들의 좌표
    # under_shape = np.concatenate((shape[2:15], np.flip(shape[31:36], axis=0)), axis=0)
    # mouth_shape = shape[48:68]
    # jaw = shape[0:17]
    # left_eyebrow = shape[22:27]
    # right_eyebrow = shape[17:22]
    # nose_bridge = shape[27:31]
    # lower_nose = shape[30:35]
    # left_eye = shape[42:48]
    # right_eye = shape[36:42]
    # outer_lip = shape[48:60]
    # inner_lip = shape[60:68]
    # jaw = shape[0:17]
    # left_eyebrow = shape[22:27]
    # right_eyebrow = shape[17:22]
    # nose_bridge = shape[27:31]
    # lower_nose = shape[30:35]
    # left_eye = shape[42:48]
    # right_eye = shape[36:42]
    # outer_lip = shape[48:60]
    # inner_lip = shape[60:68]

    # cv2.polylines(resized_img, [jaw], False, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [left_eyebrow], False, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [right_eyebrow], False, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [nose_bridge], False, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [lower_nose], True, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [left_eye], True, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [right_eye], True, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [outer_lip], True, (0, 0, 0), 1)
    # cv2.polylines(resized_img, [inner_lip], True, (0, 0, 0), 1)

    # 회전된 shape에서 랜드마크 구하기
    rotated_shape = get_rotate_point(shape[0], rotate_matrix[:, :2], shape)    
    # under_shape = np.concatenate((rotated_shape[2:15], np.flip(rotated_shape[31:36], axis=0)), axis=0)
    under_shape = rotated_shape[2:15]
    outer_mouth_shape = rotated_shape[48:60]
    inner_mouth_shape = rotated_shape[60:68]
    
    ##### make landmark image and training image
    (x, y, w, h) = get_rect(under_shape)
    landmark_img = np.zeros((h, w))
    for s in under_shape:
        s -= [x, y]
    for s in outer_mouth_shape:
        s -= [x, y]
    for s in inner_mouth_shape:
        s -= [x, y]
    cv2.polylines(landmark_img, [under_shape], False, (255, 255, 255), 2)
    cv2.polylines(landmark_img, [outer_mouth_shape], True, (255, 255, 255), 2)
    cv2.polylines(landmark_img, [inner_mouth_shape], True, (255, 255, 255), 2)

    # masking underface
    croped_img = rotated_img[y:y+h, x:x+w]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [under_shape], 255)
    masked_img = cv2.bitwise_and(croped_img, croped_img, mask=mask)

    cv2.imwrite(landmark_folder+'/{}'.format(os.path.basename(file)) , landmark_img)
    cv2.imwrite(crop_folder+'/{}'.format(os.path.basename(file)) , masked_img)
    # cv2.imshow('landmark', landmark_img)
    # cv2.imshow('masked', masked_img)
    # cv2.waitKey(10000)
