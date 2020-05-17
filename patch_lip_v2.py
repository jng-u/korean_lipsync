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
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder+'/img', exist_ok=True)

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

def middle_point(s, e):
    return (int(round((s[0]+e[0])/2)), int(round((s[1]+e[1])/2)))

def get_points_inline(points):
    ret = []
    for i, p in enumerate(points):
        if i == len(points)-1: 
            ret.append((p[0], p[1]))
            break
        pn = (points[i+1][0], points[i+1][1])
        v = (pn[0]-p[0], pn[1]-p[1])
        v = (v[0]/math.sqrt(v[0]**2+v[1]**2), v[1]/math.sqrt(v[0]**2+v[1]**2))
        x=(p[0], p[1])
        while True:
            # print((round(x[0]), round(x[1])))
            # print(points[i+1])
            if (round(x[0]), round(x[1]))==pn: break
            ret.append((int(round(x[0])), int(round(x[1]))))
            x = (x[0]+v[0], x[1]+v[1])
    return ret

def patch_lip(source, target, target_shape):    
    shape = get_face(source)
    if shape is None:
        return None

    # rescale 타겟이미지의 크기를 소스의 얼굴 하관부분에 맞춘다.
    p1 = shape[2]
    p2 = shape[14]
    # p3 = shape[8]
    vw = [a-b for a,b in zip(p2, p1)]
    # vh = [a-b for a,b in zip(p3, p1)]
    # vwp = [vw[1], -vw[0]]
    # vh = np.dot(vh, vwp)
    w = math.sqrt(vw[0]**2+vw[1]**2)
    scale = w/target.shape[1]
    target = cv2.resize(target, None, fx=scale, fy=scale)

    tshape = np.zeros(target_shape.shape)
    tshape[:,:] = target_shape[:,:]*scale

    # 타겟이미지를 source이미지의 크기만큼 span한다
    under_shape = shape[1:16]
    outer_mouth_shape = shape[48:60]

    (x, y, w, h) = get_rect(outer_mouth_shape)
    span_target = np.zeros(shape=source.shape, dtype='uint8')
    span_target[p1[1]:p1[1]+target.shape[0], p1[0]:p1[0]+target.shape[1]] = target
    # for s in target_shape:
    #     s += [p1[0], p1[1]]
    for s in tshape:
        s += [p1[0], p1[1]]
    
    # 타겟이미지를 회전하여 source이미지와 각도를 맞춘다.
    angle = -get_rotate_angle(p1, p2)
    rotate_matrix = cv2.getRotationMatrix2D((tuple(p1)), angle, 1)
    span_target = cv2.warpAffine(span_target, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    # taget_shape
    tshape = get_rotate_point(p1, rotate_matrix[:, :2], tshape)
    for i, s in enumerate(tshape):
        target_shape[i] = s
    mask = np.zeros(shape=span_target.shape[:2], dtype='uint8')
    mask = cv2.fillPoly(mask, [target_shape], (255, 255, 255))
    span_target = cv2.bitwise_and(span_target, span_target, mask=mask)

    copy = np.copy(source)
    under_shape = np.concatenate((shape[2:15], np.flip(target_shape[0:len(target_shape)-1], axis=0)), axis=0)
    # under_shape = shape[2:15]
    copy = cv2.fillPoly(copy, [under_shape], (0, 0, 0))
    # copy = cv2.cvtColor(copy, cv2.COLOR_BGR2BGRA)
    # span_target = cv2.cvtColor(span_target, cv2.COLOR_BGR2BGRA)


    # fill Gap
    # points = get_points_inline(shape[2:15])
    # for p in points:
    #     x=p[0]
    #     y=p[1]
    #     while all(k==0 for k in copy[y][x]):
    #         y+=1
    #     while True:
    #         if any(k!=0 for k in copy[y-1][x]):
    #             break
    #         copy[y][x] = copy[y+1][x]
    #         y-=1

    # (x, y, w, h) = get_rect(under_shape)
    
    # BGR채널을 BGRA로 바꾼다.
    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            # if y<=i and i<y+60 and all(x!=0 for x in span_target[i][j]):
            #     l = 60
            #     d = i-y+1
            #     ca = d/l
            #     ta = (l-d)/l
            #     copy[i][j]=copy[i][j]*ta
            #     span_target[i][j]=span_target[i][j]*ca
            if all(x!=0 for x in span_target[i][j]):
                copy[i][j] = 0
    copy = cv2.add(span_target, copy)
    return copy

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# original = cv2.VideoWriter(output_folder+'/original.avi', fourcc, 10.0, (1280, 720))
# out = cv2.VideoWriter(output_folder+'/output.avi', fourcc, 10.0, (1280, 720))
out = cv2.VideoWriter(output_folder+'/output.avi', fourcc, 10.0, (1280, 360))

source_list = glob.glob(source_folder+'/**/*.*', recursive=True)
source_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
# target_list = glob.glob(target_folder+'/**/*.*', recursive=True)
# target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
for i, file in enumerate(source_list):
    print("reading file: %s" % file)
    source = cv2.imread(file)
    # target = cv2.imread(target_list[i])
    target = cv2.imread('../data/ann2land/img/424.jpg')
    under_shape = np.zeros((14, 2), dtype='int')

    f = open('../data/ann2land/txt/424.txt')
    line = f.readline()
    p = line.split(' ')
    w = int(p[0])
    h = int(p[1])
    idx=0
    while True:
        line = f.readline()
        if not line: break
        p = line.split(' ')
        under_shape[idx] = (p[0], p[1])
        idx+=1
    f.close()

    target = cv2.resize(target, None, fx=w/256, fy=h/256)
    under_shape[:,0] = np.dot(under_shape[:,0], w/256)
    under_shape[:,1] = np.dot(under_shape[:,1], h/256)
    
    # shape = get_face(target)

    # p1 = shape[2]
    # p2 = shape[14]
    # angle = get_rotate_angle(p1, p2)
    # rotate_matrix = cv2.getRotationMatrix2D((tuple(p1)), angle, 1)
    # target = cv2.warpAffine(target, rotate_matrix, tuple(np.flip(target.shape[:2])))
    # rotated_shape = get_rotate_point(p1, rotate_matrix[:, :2], shape)
    # (x, y, w, h) = get_rect(rotated_shape[2:15])
    # mask = np.zeros(shape=target.shape[:2], dtype='uint8')
    # under_shape = np.concatenate((rotated_shape[2:15], [middle_point(rotated_shape[51], rotated_shape[33])][0:1]), axis=0)
    # for s in under_shape:
    #     s -= [x, y]
    # mask = cv2.fillPoly(mask, [under_shape], (255, 255, 255))
    # target = cv2.bitwise_and(target, target, mask=mask)
    # target = target[y:y+h, x:x+w]

    dst = patch_lip(source, target, under_shape)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    source = cv2.resize(source, None, fx=0.5, fy=0.5)
    dst = cv2.resize(dst, None, fx=0.5, fy=0.5)

    ret = np.concatenate((source, dst), axis=1)

    # original.write(source)
    # out.write(dst)
    out.write(ret)
    # cv2.imwrite(output_folder+'/img/{}'.format(os.path.basename(file)) , dst)
    # cv2.imshow('source', ret)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', dst)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
# original.release()
out.release()
cv2.destroyAllWindows()