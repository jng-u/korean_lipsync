import os
import sys
import glob
import argparse
import math
import time

import tkinter as tk
import tkinter.font

import cv2
import dlib
from imutils import face_utils
from imutils import video
from PIL import Image
from PIL import ImageTk

import numpy as np
import matplotlib.pyplot as plt

import hangul

FPS = 15

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
# predictor = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")

P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62

TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)

def get_rect(shape):
    xs = shape[:, 0]
    ys = shape[:, 1]
    x = min(xs)
    y = min(ys)
    w = max(xs)-x
    h = max(ys)-y
    return (x,y,w,h)

def get_face(img, rect):
    (x, y, w, h) = (rect[0], rect[1], rect[2], rect[3])
    img = img[y:y+h, x:x+h]
    ##### 전처리
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식 향상을 위해 Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive = clahe.apply(gray)
    ratio = 1/3
    adaptive_resize = cv2.resize(adaptive, None, fx=ratio, fy=ratio)

    #### detect face and shape
    # detects whole face
    rects = detector(adaptive_resize, 1)
    if len(rects) == 0:
        # print('error finding face')
        return (), []

    rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    (rx, ry, rw, rh) = face_utils.rect_to_bb(rects[0])
    rect = dlib.rectangle(rx-10, ry-10, rx+rw+10, ry+rh+10)   
    
    # 얼굴이 하나임을 가정
    shape = predictor(image=adaptive, box=rect)
    shape = face_utils.shape_to_np(shape)
    for s in shape:
        s += (x, y)
    return (rx+x, ry+y, rw, rh), shape

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

def linear_reg(x, y):
    n = len(x)
    xmean = sum(x)/n
    ymean = sum(y)/n
    a1 = (n*sum([a*b for a,b in zip(x, y)]) - sum(x)*sum(y)) / (n*sum([a**2 for a in x]) - sum(x)**2)
    a0 = ymean - a1*xmean
    return a1, a0

def rotateImage(input, alpha, beta, gamma, dx, dy, dz):
    alpha = (alpha)*math.pi/180
    beta = (beta)*math.pi/180
    gamma = (gamma)*math.pi/180

    # get width and height for ease of use in matrices
    w = input.shape[1]
    h = input.shape[0]

    d = np.sqrt(h**2 + w**2)
    f = d / (2 * math.sin(gamma) if math.sin(gamma) != 0 else 1)
    dz = f

    # Projection 2D -> 3D matrix
    A1 = np.array([
                    [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0,    0],
                    [0, 0,    1]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([
                    [1,               0,                0, 0],
                    [0, math.cos(alpha), -math.sin(alpha), 0],
                    [0, math.sin(alpha),  math.cos(alpha), 0],
                    [0,               0,                0, 1]])
    RY = np.array([
                    [math.cos(beta), 0, -math.sin(beta), 0],
                    [             0, 1,               0, 0],
                    [math.sin(beta), 0,  math.cos(beta), 0],
                    [             0, 0,               0, 1]])
    RZ = np.array([
                    [math.cos(gamma), -math.sin(gamma), 0, 0],
                    [math.sin(gamma),  math.cos(gamma), 0, 0],
                    [              0,                0, 1, 0],
                    [              0,                0, 0, 1]])
    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX,RY), RZ)
    # Translation matrix
    T = np.array([
                    [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0,  1]])
    # 3D -> 2D matrix
    A2 = np.array([
                    [f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0,   1, 0]])
    # Final transformation matrix
    trans = np.dot(A2, np.dot(T, np.dot(R, A1)))
    # Apply matrix transformation
    return cv2.warpPerspective(input, trans, (w, h))



def patch_lip(source, shape, target, target_shape):    
    shape = np.array(shape, dtype='int')

    # p1 = middle_point(shape[0], shape[1])
    # p2 = middle_point(shape[2], shape[3])
    p1 = middle_point(shape[36], shape[39])
    p2 = middle_point(shape[42], shape[45])
    vw = [a-b for a,b in zip(p2, p1)]
    w = math.sqrt(vw[0]**2+vw[1]**2)
    tp1 = middle_point(target_shape[36], target_shape[39])
    tp2 = middle_point(target_shape[42], target_shape[45])
    # tp1 = target_shape[36:42].mean(axis=0).astype('int')
    # tp2 = target_shape[42:48].mean(axis=0).astype('int')
    tvw = [a-b for a,b in zip(tp2, tp1)]
    tw = math.sqrt(tvw[0]**2+tvw[1]**2)
    scale = w/tw
    target = cv2.resize(target, None, fx=scale, fy=scale)
    target_shape[:,:] = target_shape[:,:]*scale

    target_r = rotateImage(target, 0, 45, 0, 0, 0, 0)
    cv2.imshow('assd', target_r)
    while(True):
        if(cv2.waitKey(10) != -1):
            break

    span_target = np.zeros(shape=source.shape, dtype='uint8')
    ctp = middle_point(target_shape[51], target_shape[57])
    l = math.sqrt(sum([x**2 for x in ctp-target_shape[33]]))
    # v = shape[4] - middle_point(middle_point(shape[0], shape[1]), middle_point(shape[2], shape[3]))
    v = shape[33] - middle_point(middle_point(shape[36], shape[39]), middle_point(shape[42], shape[45]))
    th = math.atan(v[1]/v[0])
    v = v / math.sqrt(sum([x**2 for x in v]))
    v = (int(round(l*math.cos(th)*v[0])), abs(int(round(l*math.sin(th)*v[1]))))
    # cp = shape[4] + v
    cp = shape[33] + v
    hs=cp[1]-ctp[1] if cp[1]-ctp[1]>=0 else 0
    he=cp[1]-ctp[1]+target.shape[0] if cp[1]-ctp[1]+target.shape[0]<=span_target.shape[0] else span_target.shape[0]
    ws=cp[0]-ctp[0] if cp[0]-ctp[0]>=0 else 0
    we=cp[0]-ctp[0]+target.shape[1] if cp[0]-ctp[0]+target.shape[1]<=span_target.shape[1] else span_target.shape[1]
    span_target[hs:he, ws:we] = target[0:he-hs,0:we-ws]
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
    # a1, a0 = linear_reg(shape[0:4][:, 0], shape[0:4][:, 1])
    a1, a0 = linear_reg(shape[36:48][:, 0], shape[36:48][:, 1])
    rp1 = (0, a0)
    rp2 = (1, a1+a0)
    angle = get_rotate_angle(trp1, trp2) - get_rotate_angle(rp1, rp2)

    # rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[4]), angle, 1)
    rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[33]), angle, 1)
    span_target = cv2.warpAffine(span_target, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    mask = cv2.warpAffine(mask, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    # target_shape = get_rotate_point(tuple(shape[4]), rotate_matrix[:, :2], target_shape)
    target_shape = get_rotate_point(tuple(shape[33]), rotate_matrix[:, :2], target_shape)
    
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

# OpenCV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)

flag = False
target_idx = 0
# target_list = glob.glob('../data/ljw/test/images/'+'/**/*.*', recursive=True)
target_list = glob.glob('../data/ljw/test/0612land/img'+'/**/*.*', recursive=True)
target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))

window = tk.Tk()
window.bind('<Escape>', lambda e: window.quit())
window.title("camera practice")

tk_imlabel = tk.Label(window)
tk_imlabel.pack()

string = tk.StringVar()
textbox = tk.Entry(window, font=tk.font.Font(family='mincho', size=20), width=50, textvariable=str)
textbox.pack(side=tk.LEFT)

def click():
    global flag, target_idx
    flag = True
    target_idx = 0
    tk_imlabel.after_cancel(after_id)
    show_frame()

btn = tk.Button(window, text="OK", width=15, command=click)
btn.pack(side=tk.LEFT)

pre_face = None
after_id = None
rotation_vector = np.array([[0.01891013], [0.08560084], [-3.14392813]])
translation_vector = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
# while True:
def show_frame():
    global pre_face, flag, target_idx, target_list, after_id, rotation_vector, translation_vector
    start_time = time.time()
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, flipCode=1)
    h,w,c = frame.shape

    if pre_face is None:
        pre_face = (0, 0, w, h)
    rect, shape = get_face(frame, pre_face)
    # print(frame.shape)
    # if len(shape)==0: continue
    ret = None
    if len(shape)!=0:
        pre_face = (rect[0]-int(rect[2]/4) if rect[0]-int(rect[2]/4)>=0 else 0, \
                    rect[1]-int(rect[3]/4) if rect[1]-int(rect[3]/4)>=0 else 0, \
                    int(3/2*rect[2]) if int(3/2*rect[2])<=w else w, \
                    int(3/2*rect[3]) if int(3/2*rect[3])<=h else h)

        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 1)
        
        size = frame.shape

        # image_points = np.array([
                                #     shape[33],      # Nose tip
                                #     shape[8],       # Chin
                                #     shape[36],      # Left eye left corner
                                #     shape[45],      # Right eye right corne
                                #     shape[48],      # Left Mouth corner
                                #     shape[54]       # Right mouth corner
                                # ], dtype="double")
        image_points = np.array([shape[x] for x in TRACKED_POINTS], dtype="double")

        # 3D model points.
        # model_points = np.array([
        #                             (0.0, 0.0, 0.0),             # Nose tip
        #                             (0.0, -330.0, -65.0),        # Chin
        #                             (-225.0, 170.0, -135.0),     # Left eye left corner
        #                             (225.0, 170.0, -135.0),      # Right eye right corne
        #                             (-150.0, -150.0, -125.0),    # Left Mouth corner
        #                             (150.0, -150.0, -125.0)      # Right mouth corner
                                
        #                         ])
        model_points = np.array([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION], dtype='float32') 


        # Camera internals

        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], 
                                dtype = "double")

        # print("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        # (_, rotation_vector, translation_vector) = cv2.solvePnP(
        #                                                         model_points,
        #                                                         image_points,
        #                                                         camera_matrix,
        #                                                         dist_coeffs,
        #                                                         rvec=rotation_vector,
        #                                                         tvec=translation_vector,
        #                                                         useExtrinsicGuess=True)
        # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_oeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        retval, rotation_vector, translation_vector = cv2.solvePnP( model_points, image_points, camera_matrix, dist_coeffs)

        # rotation_vector[0] =-1
        # rotation_vector[1] =1
        # rotation_vector[2] =1

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))


        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose


        axis = np.float32([[100,0,0], 
                            [0,100,0], 
                            [0,0,100]])
        (imgpts, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # for p in image_points:
        #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


        # p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        # p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # dst = np.copy(frame)

        sellion_xy = (int(image_points[7][0]), int(image_points[7][1]))
        # sellion_xy = (int(image_points[0][0]), int(image_points[0][1]))
        # cv2.line(dst, p1, p2, (255,0,0), 2)

        target = cv2.imread('./data/jland/img/3.jpg')
        # target = cv2.imread(target_list[target_idx])
        target_shape = np.zeros((69, 2), dtype='int')

        f = open('./data/jland/txt/3.txt')
        # f = open(os.path.dirname(target_list[target_idx])+'/../txt/'+os.path.basename(target_list[target_idx])[:len(os.path.basename(target_list[target_idx]))-4]+'.txt')
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

        dst = patch_lip(frame, shape, target, target_shape)

        cv2.line(dst, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
        cv2.line(dst, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
        cv2.line(dst, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

        ret = np.concatenate((frame, dst), axis=1)

        target_idx+=1
        if target_idx==len(target_list):
            flag=False
    else:
        ret = np.concatenate((frame, frame), axis=1)

    duration = time.time()-start_time

    text = '%.2f FPS' % (1/duration)
    cv2.putText(ret, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    img = cv2.cvtColor(ret, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    tk_imlabel.img = img
    tk_imlabel.configure(image=img)
    after_id = tk_imlabel.after(1, show_frame)
    # cv2.imwrite('./3.jpg', frame)
    # cv2.imshow('aa', ret)
    # cv2.waitKey(1)

show_frame()
window.mainloop()