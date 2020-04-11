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

def masked_histogram_equalization(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cnt=0
    elems = []
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if mask[y, x] != 0:
                elems.insert(cnt, gray[y, x])
                cnt += 1
    hist, bins = np.histogram(elems, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[gray]

file_list = glob.glob(input_folder+'**/*.*', recursive=True)
for file in file_list:
    print("reading file: %s" % file)
    img = cv2.imread(file)

    ##### 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식 향상을 위해 Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive = clahe.apply(gray)

    ##### detect face and shape
    # detects whole face
    rects = detector(adaptive, 1)
    if len(rects) == 0:
        print("error finding face at file: %s" % file)
        continue
    
    # 얼굴이 하나임을 가정
    # rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    shape = predictor(image=adaptive, box=rects[0])
    shape = shape_to_np(shape)

    # 0~16 : 턱(2~14:광대 아래쪽) / 31~35 : 아래코 / 48~67 : mouth
    # 얼굴 하관 점들의 좌표
    under_shape = shape[2:15]
    outer_lip = shape[48:60]
    inner_lip = shape[60:68]
    
    ##### make landmark image and training image
    landmark = np.zeros(img.shape[:2])
    cv2.polylines(landmark, [under_shape], False, (255, 255, 255), 2)
    cv2.polylines(landmark, [outer_lip], True, (255, 255, 255), 2)
    cv2.polylines(landmark, [inner_lip], True, (255, 255, 255), 2)

    cv2.imwrite(output_folder+'/{}'.format(os.path.basename(file)) , landmark)
