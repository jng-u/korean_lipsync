import os
import sys
import glob
import argparse
import math

import cv2
import dlib
from imutils import face_utils

import numpy as np

detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")

def get_rect(shape):
    xs = shape[:, 0]
    ys = shape[:, 1]
    x = min(xs)
    y = min(ys)
    w = max(xs)-x
    h = max(ys)-y
    return (x,y,w,h)

def get_face(img):
    ##### 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식 향상을 위해 Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive = clahe.apply(gray)
    ratio = 1/6
    adaptive_resize = cv2.resize(adaptive, None, fx=ratio, fy=ratio)

    #### detect face and shape
    # detects whole face
    rects = detector(adaptive_resize, 1)
    if len(rects) == 0:
        print('error finding face')
        return None

    rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    rect = dlib.rectangle(x-10, y-10, x+w+10, y+h+10)   
    
    # 얼굴이 하나임을 가정
    shape = predictor(image=adaptive, box=rect)
    shape = face_utils.shape_to_np(shape)
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

def linear_reg(x, y):
    n = len(x)
    xmean = sum(x)/n
    ymean = sum(y)/n
    a1 = (n*sum([a*b for a,b in zip(x, y)]) - sum(x)*sum(y)) / (n*sum([a**2 for a in x]) - sum(x)**2)
    a0 = ymean - a1*xmean
    return a1, a0

def curve_fitting(n, data, max_i):
    w0 = 2 * math.pi / n 
    x = range(0, n)
    
    A = np.zeros(max_i)
    B = np.zeros(max_i)

    A0 = sum(data)/n
    for i in range(0, max_i):
        A[i] = sum([d*math.cos((i+1)*w0*t) for d, t in zip(data, x)])*2/n
    for i in range(0, max_i):
        B[i] = sum([d*math.sin((i+1)*w0*t) for d, t in zip(data, x)])*2/n

    ret = np.ones(n)
    ret = np.dot(A0, ret)
    for i in range(0, max_i):
        ret = [r + A[i]*math.cos((i+1)*w0*t) + B[i]*math.sin((i+1)*w0*t) for r, t in zip(ret, x)]
    return ret

def patch_lip(source, shape, target, target_shape, width_fitting):    
    shape = np.array(shape, dtype='int')

    tp1 = target_shape[36:42].mean(axis=0).astype('int')
    tp2 = target_shape[42:48].mean(axis=0).astype('int')
    tvw = [a-b for a,b in zip(tp2, tp1)]
    tw = math.sqrt(tvw[0]**2+tvw[1]**2)
    scale = width_fitting/tw
    target = cv2.resize(target, None, fx=scale, fy=scale)
    target_shape[:,:] = target_shape[:,:]*scale

    span_target = np.zeros(shape=source.shape, dtype='uint8')
    ctp = middle_point(target_shape[51], target_shape[57])
    l = math.sqrt(sum([x**2 for x in ctp-target_shape[33]]))
    v = shape[4] - middle_point(middle_point(shape[0], shape[1]), middle_point(shape[2], shape[3]))
    th = math.atan(v[1]/v[0])
    v = v / math.sqrt(sum([x**2 for x in v]))
    v = (int(round(l*math.cos(th)*v[0])), abs(int(round(l*math.sin(th)*v[1]))))
    cp = shape[4] + v
    span_target[cp[1]-ctp[1]:cp[1]-ctp[1]+target.shape[0], cp[0]-ctp[0]:cp[0]-ctp[0]+target.shape[1]] = target
    for s in target_shape:
        s += [cp[0]-ctp[0], cp[1]-ctp[1]]
    
    mask = np.zeros(shape=span_target.shape[:2], dtype='float')
    (tx, ty, tw, th) = get_rect(target_shape[2:15])
    (mx, my, mw, mh) = get_rect(target_shape[48:60])
    padding = 15
    (x, y, w, h) = (mx-padding, my-padding, mw+2*padding, mh+2*padding)
    if tx>x: x=tx
    if ty>y: y=ty
    if x+w>x+tw: w=tw
    if y+h>y+th: h=th
    for i in range(y, y+h):
        for j in range(x, x+w):
            d=1
            l=0
            if i<=my and j<=mx:
                # d=1
                if mx-j < my-i:
                    d = math.sqrt((j-mx)**2 + (my-y)**2)
                else:
                    d = math.sqrt((x-mx)**2 + (my-i)**2)
                # d = math.sqrt((x-mx)**2 + (y-my)**2)
                l = math.sqrt((j-mx)**2 + (i-my)**2)
            elif i<=my and j>=mx+mw:
                # d=1
                if j-(mx+mw) < my-i:
                    d = math.sqrt((j-mx-mw)**2 + (my-y)**2)
                else:
                    d = math.sqrt((x+w-mx-mw)**2 + (my-i)**2)
                # d = math.sqrt((x+w-mx-mw)**2 + (y-my)**2)
                l = math.sqrt((j-mx-mw)**2 + (i-my)**2)
            elif i>=my+mh and j<=mx:
                # d=1
                # if mx-j < i-(my+mh):
                #     d = math.sqrt((j-mx)**2 + (y+h-my-mh)**2)
                # else:
                #     d = math.sqrt((x-mx)**2 + (i-my-mh)**2)
                # d = math.sqrt((x-mx)**2 + (y+h-my-mh)**2)
                d = math.sqrt(((x-mx)/2)**2 + ((y+h-my-mh)/2)**2)
                l = math.sqrt((j-mx)**2 + (i-my-mh)**2)
            elif i>=my+mh and j>=mx+mw:
                # d=1
                # if j-(mx+mw) < i-(my+mh):
                #     d = math.sqrt((j-mx-mw)**2 + (y+h-my-mh)**2)
                # else:
                #     d = math.sqrt((x+w-mx-mw)**2 + (i-my-mh)**2)
                # d = math.sqrt((x+w-mx-mw)**2 + (y+h-my-mh)**2)
                d = math.sqrt(((x+w-mx-mw)/2)**2 + ((y+h-my-mh)/2)**2)
                l = math.sqrt((j-mx-mw)**2 + (i-my-mh)**2)
            elif i < my:
                d = my - y
                l = my - i
            elif i > my+mh:
                # d=1
                d = y+h-my-mh
                l = i-my-mh
            elif j < mx:
                d = mx - x
                l = mx - j
            elif j > mx+mw:
                d = x+w-mx-mw
                l = j-mx-mw
            ca = l/d
            ta = (d-l)/d
            if ta<0: ta=0
            mask[i, j] = ta
            

    copy = np.copy(source)    


    # 타겟이미지를 회전하여 source이미지와 각도를 맞춘다.
    ta1, ta0 = linear_reg(target_shape[36:48][:, 0], target_shape[36:48][:, 1])
    trp1 = (0, ta0)
    trp2 = (1, ta1+ta0)
    a1, a0 = linear_reg(shape[0:4][:, 0], shape[0:4][:, 1])
    rp1 = (0, a0)
    rp2 = (1, a1+a0)
    angle = get_rotate_angle(trp1, trp2) - get_rotate_angle(rp1, rp2)

    rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[4]), angle, 1)
    span_target = cv2.warpAffine(span_target, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    mask = cv2.warpAffine(mask, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    target_shape = get_rotate_point(tuple(shape[4]), rotate_matrix[:, :2], target_shape)
    
    mm = np.zeros(shape=span_target.shape[:2], dtype='uint8')
    mm = cv2.fillPoly(mm, [target_shape[2:15]], 255)
    mask = cv2.bitwise_and(mask, mask, mask=mm)

    span_target[:,:,0] = span_target[:,:,0]*mask
    span_target[:,:,1] = span_target[:,:,1]*mask
    span_target[:,:,2] = span_target[:,:,2]*mask

    mask = 1-mask
    copy[:,:,0] = copy[:,:,0]*mask
    copy[:,:,1] = copy[:,:,1]*mask
    copy[:,:,2] = copy[:,:,2]*mask

    copy = cv2.add(copy, span_target)


    return copy