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

def patch_lip(source, shape, target, target_shape):    
    # shape = get_face(source)
    # if shape is None:
    #     return None

    shape = np.array(shape, dtype='int')

    # print(shape)

    # rescale 타겟이미지의 크기를 소스의 얼굴 하관부분에 맞춘다.
    # p1 = shape[2]
    # p2 = shape[14]
    # p3 = shape[8]
    # vw = [a-b for a,b in zip(p2, p1)]
    # vh = [a-b for a,b in zip(p3, p1)]
    # vwp = [vw[1], -vw[0]]
    # w = math.sqrt(vw[0]**2+vw[1]**2)
    # h = abs(np.dot(vh, vwp)/math.sqrt(vwp[0]**2+vwp[1]**2))
    # target = cv2.resize(target, None, fx=w/256, fy=h/256)
    # target_shape[:, 0] = target_shape[:, 0]*w/256
    # target_shape[:, 1] = target_shape[:, 1]*h/256

    # 타겟이미지를 source이미지의 크기만큼 span한다
    under_shape = shape[2:15]
    outer_mouth_shape = shape[48:60]

    (x, y, w, h) = get_rect(outer_mouth_shape)
    span_target = np.zeros(shape=source.shape, dtype='uint8')
    # cp = (int((shape[57][0]+shape[51][0])/2), int((shape[57][1]+shape[51][1])/2))
    # ctp = (int((target_shape[3][0]+target_shape[9][0])/2), int((target_shape[3][1]+target_shape[9][1])/2))
    cp = middle_point(shape[57], shape[51])
    ctp = middle_point(target_shape[51], target_shape[57])
    span_target[cp[1]-ctp[1]:cp[1]-ctp[1]+target.shape[0], cp[0]-ctp[0]:cp[0]-ctp[0]+target.shape[1]] = target
    
    # 타겟이미지를 회전하여 source이미지와 각도를 맞춘다.
    # rp1 = (int((shape[49][0]+shape[59][0])/2), int((shape[49][1]+shape[59][1])/2))
    # rp2 = (int((shape[53][0]+shape[55][0])/2), int((shape[53][1]+shape[55][1])/2))
    rp1 = middle_point(shape[49], shape[59])
    rp2 = middle_point(shape[53], shape[55])
    rp1 = middle_point(rp1, shape[48])
    rp2 = middle_point(rp2, shape[54])
    # angle = get_rotate_angle(target_shape[0], target_shape[6]) - get_rotate_angle(shape[48], shape[54])
    angle = get_rotate_angle(target_shape[48], target_shape[54]) - get_rotate_angle(rp1, rp2)
    rotate_matrix = cv2.getRotationMatrix2D((tuple(shape[51])), angle, 1)
    span_target = cv2.warpAffine(span_target, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    
    # 소스이미지에서 합성하고싶은 만큼의 크기를 정한다.
    padding = 30
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
out = cv2.VideoWriter(output_folder+'/output.avi', fourcc, 10.0, (1280, 360))

source_list = glob.glob(source_folder+'/**/*.*', recursive=True)
source_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
target_list = glob.glob(target_folder+'/**/*.*', recursive=True)
target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))

shapes = []
for i, file in enumerate(source_list):
    print("reading file: %s" % file)
    source = cv2.imread(file)
    shapes.append(get_face(source))

shape2 = []
shape2.append(shapes[0])
for i in range(1, len(shapes)-1):
    # shapes[i] = (shapes[i-1]+shapes[i+1])/2
    w = (shapes[i-1]+shapes[i+1])/2
    shape2.append(shapes[i]*0.5 + w*0.5)
shape2.append(shapes[len(shapes)-1])

print("start making video")    
for i, file in enumerate(source_list):
    print("{} / {}".format(i+1, len(source_list)))
    source = cv2.imread(file)
    # target = cv2.imread(target_list[i])
    target = cv2.imread('../data/ann2land/img/424.jpg')
    target_shape = np.zeros((69, 2), dtype='int')
    
    f = open('../data/ann2land/txt/424.txt')
    # f = open(os.path.dirname(target_list[i])+'/../txt/'+os.path.basename(target_list[i])[:len(os.path.basename(target_list[i]))-4]+'.txt')
    line = f.readline()
    p = line.split(' ')
    w = int(p[0])
    h = int(p[1])
    idx=0
    while True:
        line = f.readline()
        if not line: break
        p = line.split(' ')
        target_shape[idx] = (p[0], p[1])
        idx+=1
    f.close()

    target = cv2.resize(target, None, fx=w/256, fy=h/256)
    target_shape[:,0] = np.dot(target_shape[:,0], w/256)
    target_shape[:,1] = np.dot(target_shape[:,1], h/256)

    dst = patch_lip(source, shape2[i], target, target_shape)
    # dst = patch_lip(source, shapes[i], target, target_shape)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    source = cv2.resize(source, None, fx=0.5, fy=0.5)
    dst = cv2.resize(dst, None, fx=0.5, fy=0.5)

    ret = np.concatenate((source, dst), axis=1)

    out.write(ret)
    cv2.imwrite(output_folder+'/img/{}'.format(os.path.basename(file)) , dst)
    # cv2.imshow('source', ret)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', dst)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
out.release()
cv2.destroyAllWindows()