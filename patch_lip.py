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
parser.add_argument('--source', dest='source_folder', type=str)
parser.add_argument('--target', dest='target_folder', type=str)
parser.add_argument('--output', dest='output_folder', type=str)
args = parser.parse_args()

source_folder = args.source_folder
target_folder = args.target_folder
output_folder = args.output_folder

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

def get_face(img):
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
        return None
    
    # 얼굴이 하나임을 가정
    # rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    shape = predictor(image=adaptive,box=rects[0])
    shape = shape_to_np(shape)
    return shape

def get_rotate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan(dy/dx))

def get_rotate_point(pivot, rotate_matrix, point_list):
    rotated_point = np.zeros(point_list.shape, point_list.dtype)
    for i in range(len(rotated_point)):
        rotated_point[i] = pivot + np.dot(rotate_matrix, point_list[i]-pivot)
    return rotated_point

def patch_lip(source, target, target_shape):    
    shape = get_face(source)
    if shape is None:
        return None

    # rescale 타겟이미지의 크기를 소스의 얼굴 하관부분에 맞춘다.
    p1 = shape[2]
    p2 = shape[14]
    p3 = shape[8]
    vw = [a-b for a,b in zip(p2, p1)]
    vh = [a-b for a,b in zip(p3, p1)]
    vwp = [vw[1], -vw[0]]
    w = math.sqrt(vw[0]**2+vw[1]**2)
    h = abs(np.dot(vh, vwp)/math.sqrt(vwp[0]**2+vwp[1]**2))
    target = cv2.resize(target, None, fx=w/256, fy=h/256)
    target_shape[:, 0] = target_shape[:, 0]*w/256
    target_shape[:, 1] = target_shape[:, 1]*h/256

    # 타겟이미지를 source이미지의 크기만큼 span한다
    under_shape = shape[2:15]
    outer_mouth_shape = shape[48:60]

    (x, y, w, h) = get_rect(outer_mouth_shape)
    span_target = np.zeros(shape=source.shape, dtype='uint8')
    cp = (int((shape[57][0]+shape[51][0])/2), int((shape[57][1]+shape[51][1])/2))
    ctp = (int((target_shape[3][0]+target_shape[9][0])/2), int((target_shape[3][1]+target_shape[9][1])/2))
    span_target[cp[1]-ctp[1]:cp[1]-ctp[1]+target.shape[0], cp[0]-ctp[0]:cp[0]-ctp[0]+target.shape[1]] = target
    
    # 타겟이미지를 회전하여 source이미지와 각도를 맞춘다.
    angle = get_rotate_angle(target_shape[0], target_shape[6]) - get_rotate_angle(shape[48], shape[54])
    rotate_matrix = cv2.getRotationMatrix2D((tuple(shape[51])), angle, 1)
    span_target = cv2.warpAffine(span_target, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    
    # 소스이미지에서 합성하고싶은 만큼의 크기를 정한다.
    padding = 20
    mask = np.zeros(shape=source.shape[:2], dtype='uint8')
    mask[y-int(padding/2):y+h+int(padding/2), x-int(padding/2):x+w+int(padding/2)] = np.ones(shape=(h+int(padding), w+int(padding)))
    span_target = cv2.bitwise_and(span_target, span_target, mask=mask)

    # BGR채널을 BGRA로 바꾼다.
    copy = np.copy(source)
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2BGRA)
    span_target = cv2.cvtColor(span_target, cv2.COLOR_BGR2BGRA)

    #합성한 이미지의 outer lip의 영역을 구한다.
    mask = 1-mask
    tmp = cv2.bitwise_and(copy, copy, mask=mask)
    tmp = cv2.add(span_target, tmp)
    tshape = get_face(tmp)
    if tshape is None:
        return None
    tx, ty, tw, th = get_rect(tshape[48:60])
    
    # alpha 를 가장자리쪽으로 가면 크게하여 합성을 자연스럽게 한다.
    for i in range(y-int(padding/2), y+h+int(padding/2)):
        for j in range(x-int(padding/2), x+w+int(padding/2)):
                ca = 1
                ta = 1
                if i < ty:
                    l = ty - (y-int(padding/2))
                    d = ty - i
                    if j < tx:
                        if tx - j > d:
                            l = tx - (x-int(padding/2))
                            d = tx - j
                    elif tx+tw <= j:
                        if j - (tx+tw) + 1 > d:
                            l = x+w+int(padding/2) - (tx+tw)
                            d = j - (tx+tw) + 1
                    ca = d/l
                    ta = (l-d)/l
                    copy[i,j] = copy[i, j]*ca
                    span_target[i,j] = span_target[i,j]*ta
                    # copy[i,j] += span_target[i, j]
                elif ty+th <= i:
                    l = y+h+int(padding/2) - (ty+th)
                    d = i - (ty+th) + 1
                    if j < tx:
                        if tx - j > d:
                            l = tx - (x-int(padding/2))
                            d = tx - j
                    elif tx+tw <= j:
                        if j - (tx+tw) + 1 > d:
                            l = x+w+int(padding/2) - (tx+tw)
                            d = j - (tx+tw) + 1
                    ca = d/l
                    ta = (l-d)/l
                    copy[i,j] = copy[i, j]*ca
                    span_target[i,j] = span_target[i,j]*ta
                    # copy[i,j] += span_target[i, j]
                elif j < tx:
                    l = tx - (x-int(padding/2))
                    d = tx - j
                    ca = d/l
                    ta = (l-d)/l
                    copy[i,j] = copy[i, j]*ca
                    span_target[i,j] = span_target[i,j]*ta
                    # copy[i,j] += span_target[i, j]
                elif tx+tw <= j:
                    l = x+w+int(padding/2) - (tx+tw)
                    d = j - (tx+tw) + 1
                    ca = d/l
                    ta = (l-d)/l
                    copy[i,j] = copy[i, j]*ca
                    span_target[i,j] = span_target[i,j]*ta
                    # copy[i,j] += span_target[i, j]
                else :
                    copy[i,j] = 0
    copy = cv2.add(span_target, copy)
    return copy

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_folder+'/test.avi', fourcc, 10.0, (1280, 720))

source_list = glob.glob(source_folder+'/**/*.*', recursive=True)
source_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
target_list = glob.glob(target_folder+'/**/*.*', recursive=True)
target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
for i, file in enumerate(source_list):
    print("reading file: %s" % file)
    source = cv2.imread(file)
    target = cv2.imread(target_list[i])
    target_shape = np.zeros((20, 2), dtype='int')
    f = open(os.path.dirname(target_list[i])+'/../landmark/{}.txt'.format(os.path.basename(target_list[i])[:len(os.path.basename(target_list[i]))-4]), 'r')
    idx=0
    while True:
        line = f.readline()
        if not line: break
        p = line.split(' ')
        target_shape[idx] = (p[0], p[1])
        idx+=1
    f.close()
    
    dst = patch_lip(source, target, target_shape)
    # wpath = output_folder+file[len(input_folder):]
    # os.makedirs(os.path.dirname(wpath), exist_ok=True)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    out.write(dst)
    cv2.imwrite(output_folder+'/img/{}'.format(os.path.basename(file)) , dst)
    # cv2.imshow('source', source)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', dst)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
out.release()
cv2.destroyAllWindows()