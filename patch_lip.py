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

import tools.hangul

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
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62
#P3D_LIP_RIGHT = np.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = np.float32([-20.0, 65.5,-5.0]) #54

TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)

# cascPath = "/home/jngu/anaconda3/envs/py3/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
# predictor = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")

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
    ##### �쟾泥섎━
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # �뼹援� �씤�떇 �뼢�긽�쓣 �쐞�빐 Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive = clahe.apply(gray)
    ratio = 1/7
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
    
    # �뼹援댁씠 �븯�굹�엫�쓣 媛��젙
    shape = predictor(image=adaptive, box=rect)
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

def patch_lip(source, shape, target, target_shape):    
    shape = np.array(shape, dtype='int')

    p1 = shape[2]
    p2 = shape[14]
    vw = [a-b for a,b in zip(p2, p1)]
    w = math.sqrt(vw[0]**2+vw[1]**2)
    # tp1 = target_shape[48]
    # tp2 = target_shape[54]
    # tvw = [a-b for a,b in zip(tp2, tp1)]
    # tw = math.sqrt(tvw[0]**2+tvw[1]**2)
    # scale = w/tw
    p3 = shape[8]
    vh = [a-b for a,b in zip(p3, p1)]
    vwp = [vw[1], -vw[0]]
    scale = w/target.shape[1]
    target = cv2.resize(target, None, fx=scale, fy=scale)
    target_shape[:,:] = target_shape[:,:]*scale
    source_height = abs(np.dot(vh, vwp)/math.sqrt(vwp[0]**2+vwp[1]**2))
    target_height = target.shape[0]

    # mm = np.zeros(shape=target.shape[:2], dtype='uint8')
    # mm = cv2.fillPoly(mm, [target_shape[2:15]], 255)
    # mm = cv2.erode(mm, np.ones((5, 1)), iterations=5)

    bg_shape = np.copy(shape)
    background = np.copy(source)
        
    angle = get_rotate_angle(shape[2], shape[14])
    rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[2]), angle, 1)
    background = cv2.warpAffine(background, rotate_matrix, tuple(np.flip(background.shape[:2])))
    bg_shape = get_rotate_point(tuple(shape[2]), rotate_matrix[:, :2], bg_shape)

    (x, y, w, h) = get_rect(bg_shape[2:15])
    background_crop = np.zeros((w, h))
    background_crop = background[y:y+h, x:x+w]
    for s in bg_shape:
        s -= [x, y]
    if target_height > h:
        scale = target_height/h
    else:
        scale = 1
    background_crop = cv2.resize(background_crop, None, fx=1, fy=scale)
    bg_shape[:,1] = bg_shape[:,1]*scale

    span_background = np.zeros(shape=source.shape, dtype='uint8')
    span_background[y:y+background_crop.shape[0], x:x+background_crop.shape[1]] = background_crop
    for s in bg_shape:
        s += [x, y]
        
    angle = - get_rotate_angle(shape[2], shape[14])
    rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[2]), angle, 1)
    span_background = cv2.warpAffine(span_background, rotate_matrix, tuple(np.flip(background.shape[:2])))
    bg_shape = get_rotate_point(tuple(shape[2]), rotate_matrix[:, :2], bg_shape)



    span_target = np.zeros(shape=source.shape, dtype='uint8')
    v = target_shape[51] - target_shape[33]
    # v = [middle_point(target_shape[48], target_shape[54])[0], target_shape[51][1] - target_shape[33][1]]
    # v = middle_point(target_shape[48], target_shape[54]) - target_shape[33]
    
    ctp = target_shape[51]
    # ctp = [middle_point(target_shape[48], target_shape[54])[0], target_shape[51][1]]
    # ctp = middle_point(target_shape[48], target_shape[54])
    cp = shape[33] + v 
    wd = cp[0]-ctp[0]
    hd = cp[1]-ctp[1]
    span_target[hd:hd+target.shape[0], wd:wd+target.shape[1]] = target
    for s in target_shape:
        s += [wd, hd]

    mask = np.zeros(shape=span_target.shape[:2], dtype='float')
    (tx, ty, tw, th) = get_rect(target_shape[2:15])
    (mx, my, mw, mh) = get_rect(target_shape[48:60])
    padding = int(mh/2)
    (x, y, w, h) = get_rect(target_shape[2:15])
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

    # ���寃잛씠誘몄��瑜� �쉶�쟾�븯�뿬 source�씠誘몄����� 媛곷룄瑜� 留욎텣�떎.
    ta1, ta0 = linear_reg(target_shape[48:55][:, 0], target_shape[48:55][:, 1])
    trp1 = (0, ta0)
    trp2 = (1, ta1+ta0)
    a1, a0 = linear_reg(shape[48:55][:, 0], shape[48:55][:, 1])
    rp1 = (0, a0)
    rp2 = (1, a1+a0)
    angle = get_rotate_angle(trp1, trp2) - get_rotate_angle(rp1, rp2)
    # angle = get_rotate_angle(target_shape[48], target_shape[54]) - get_rotate_angle(shape[48], shape[54])
    # angle = - get_rotate_angle(shape[2], shape[14])
    
    rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[33]), angle, 1)
    # rotate_matrix = cv2.getRotationMatrix2D(tuple(shape[2]), angle, 1)
    span_target = cv2.warpAffine(span_target, rotate_matrix, tuple(np.flip(span_target.shape[:2])))
    target_shape = get_rotate_point(tuple(shape[33]), rotate_matrix[:, :2], target_shape)
    # target_shape = get_rotate_point(tuple(shape[2]), rotate_matrix[:, :2], target_shape)

    copy = np.copy(source)   

    # removed_undershape = np.concatenate((shape[2:15], np.flip(bg_shape[2:15], axis=0)), axis=0)     
    # copy = cv2.fillPoly(copy, [removed_undershape], (0, 0, 0))

    # if target_height > source_height-source_height-10:        
    if True:        
        bg_mask = np.zeros(shape=span_background.shape[:2], dtype='float')
        (x, y, w, h) = get_rect(bg_shape[2:15])
        m = int(h/2)
        for i in range(y, y+h):
            if i>y+m:
                bg_mask[i, :] = (i-(y+m))/m
            else:    
                # bg_mask[i, :] = 1-(y+m-i)/m
                bg_mask[i, :] = 0

        # print(bg_mask)

        # copycopy = np.copy(copy)
        # copycopy = cv2.fillPoly(copycopy, [shape[2:15]], (0, 0, 0))
        
        # bg_copy = np.copy(span_background)
        # mm = np.zeros(shape=span_background.shape[:2], dtype='uint8')
        # mm = cv2.fillPoly(mm, [bg_shape[2:15]], 255)
        # bg_copy = cv2.bitwise_and(bg_copy, bg_copy, mask=mm)

        # copycopy = cv2.add(copycopy, bg_copy)
        # sh = get_face(copycopy)
        
        # jaw = sh[0:17]
        # cv2.polylines(copy, [jaw], False, (0, 0, 0), 2)
        # cv2.imshow('sasas', copycopy)
        # while True:
        #     if cv2.waitKey(1) != -1:
        #         break

        mm = np.zeros(shape=span_background.shape[:2], dtype='uint8')
        # mm = cv2.fillPoly(mm, [sh[2:15]], 255)
        mm = cv2.fillPoly(mm, [bg_shape[2:15]], 255)
        # mm = cv2.erode(mm, np.ones((5, 1)), iterations=5)
        bg_mask = cv2.bitwise_and(bg_mask, bg_mask, mask=mm)

        # bg_mask = np.zeros(shape=span_background.shape[:2], dtype='uint8')
        # bg_mask = cv2.fillPoly(bg_mask, [bg_shape[2:15]], 255)
        # # (x, y, w, h) = get_rect(bg_shape[48:60])
        # # x-=10
        # # y-=10
        # # w+=10
        # # h+=10
        # # mouth = np.array(([x, y], [x+w, y], [x+w, y+h], [x, y+h]))
        # bg_mask = cv2.fillPoly(bg_mask, [bg_shape[48:60]], 0)
        # # bg_mask = cv2.erode(bg_mask, np.ones((5, 5)), iterations=5)
        # span_background = cv2.bitwise_and(span_background, span_background, mask=bg_mask)
        # bg_mask = 255-bg_mask
        # copy = cv2.bitwise_and(copy, copy, mask=bg_mask)
        
        span_background[:,:,0] = span_background[:,:,0]*bg_mask
        span_background[:,:,1] = span_background[:,:,1]*bg_mask
        span_background[:,:,2] = span_background[:,:,2]*bg_mask

        bg_mask = 1-bg_mask
        copy[:,:,0] = copy[:,:,0]*bg_mask
        copy[:,:,1] = copy[:,:,1]*bg_mask
        copy[:,:,2] = copy[:,:,2]*bg_mask

        copy = cv2.add(copy, span_background)

    mm = np.zeros(shape=span_target.shape[:2], dtype='uint8')
    mm = cv2.fillPoly(mm, [shape[2:15]], 255)
    mask = cv2.bitwise_and(mask, mask, mask=mm)

    mm = np.zeros(shape=span_target.shape[:2], dtype='uint8')
    mm = cv2.fillPoly(mm, [target_shape[2:15]], 255)
    mm = cv2.erode(mm, np.ones((3, 3)), iterations=2)
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

source_list = glob.glob(source_folder+'/**/*.*', recursive=True)
source_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
# target_list = glob.glob(target_folder+'/**/*.*', recursive=True)
# target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
txt_list = glob.glob(target_folder+'/../txt/**/*.*', recursive=True)
txt_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
tmp = cv2.imread(source_list[0])
out = cv2.VideoWriter(output_folder+'/output.avi', fourcc, 30.0, (tmp.shape[1]*2, tmp.shape[0]))
out_convert = cv2.VideoWriter(output_folder+'/output_convert.avi', fourcc, 30.0, (tmp.shape[1], tmp.shape[0]))
out_original = cv2.VideoWriter(output_folder+'/output_original.avi', fourcc, 30.0, (tmp.shape[1], tmp.shape[0]))

# 448855444233666
sequence = tools.hangul.sound_to_img('안녕하세요')
# sequence = [4, 4, 8, 8, 5, 5, 4, 4, 4, 2, 3, 3, 6, 6, 6]
# sequence = tools.hangul.sound_to_img('밥은먹고다니냐')
# sequence = [1, 4, 1, 2, 2, 2, 1, 5, 5, 6, 6, 6, 8, 4, 4, 8, 2, 2, 8, 4, 4]
# sequence = tools.hangul.sound_to_img('컴퓨터비전')
# sequence = tools.hangul.sound_to_img('이정우입니다')
# sequence = [2, 2, 2, 2, 5, 5, 7, 7, 7, 2, 2, 1, 1, 2, 2, 8, 4, 4]
# sequence = [2, 2, 2, 7, 7, 7, 6, 6, 6, 2, 2, 1, 1, 2, 2, 8, 4, 4]
# sequence = tools.hangul.sound_to_img('안녕하세요')
# sequence = [4, 4, 8, 8, 5, 5, 4, 4, 4, 2, 3, 3, 6, 6, 6, 1, 1, 1, 2, 2, 2, 2, 5, 5, 7, 7, 7, 2, 2, 1, 1, 2, 2, 8, 4, 4]
# sequence = [4, 4, 8, 8, 5, 5, 4, 4, 4, 2, 3, 3, 6, 6, 6, 1, 1, 1, 2, 2, 2, 7, 7, 7, 6, 6, 6, 2, 2, 1, 1, 2, 2, 8, 4, 4]
sequence = [1, 2]
# print(sequence)
# target_list = glob.glob(target_folder+'/**/*.*', recursive=True)
# target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
# shapes = []
# ws = []
# for i in range(len(target_list)):
#     print("reading file: %s" % source_list[i])
#     source = cv2.imread(source_list[i])
#     shape = get_face(source)
#     shapes.append(shape)

print("start making video")    
# for i, file in enumerate(target_list):
cnt=0
# for i, file in enumerate(target_list):
for i in range(len(sequence)-1):

    # s = sequence[i]
    # e = sequence[i+1]
    # if s == e:
    #     if s<8:
    #         e = s+1
    #         target_list = glob.glob(target_folder+'/{}{}/*.*'.format(s, e), recursive=True)
    #         txt_list = glob.glob(target_folder+'/../txt/{}{}/*.*'.format(s, e), recursive=True)
    #         target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #         txt_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #     else:
    #         s = e-1
    #         target_list = glob.glob(target_folder+'/{}{}/*.*'.format(s, e), recursive=True)
    #         txt_list = glob.glob(target_folder+'/../txt/{}{}/*.*'.format(s, e), recursive=True)
    #         target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #         txt_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #         target_list.reverse()
    #         txt_list.reverse()
    #     target_list = [target_list[0] for i in range(11)]
    #     txt_list = [txt_list[0] for i in range(11)]
    # elif s < e:
    #     target_list = glob.glob(target_folder+'/{}{}/*.*'.format(s, e), recursive=True)
    #     txt_list = glob.glob(target_folder+'/../txt/{}{}/*.*'.format(s, e), recursive=True)
    #     target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #     txt_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    # elif e < s:
    #     target_list = glob.glob(target_folder+'/{}{}/*.*'.format(e, s), recursive=True)
    #     txt_list = glob.glob(target_folder+'/../txt/{}{}/*.*'.format(e, s), recursive=True)
    #     target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #     txt_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    #     target_list.reverse()
    #     txt_list.reverse()
    
    # dnum = int(len(target_list)/2)
    # if i+1==len(sequence)-1:
    #     # tmp = target_list[len(target_list)-1]
    #     target_list = [target_list[i*2] if i<dnum else target_list[len(target_list)-1] for i in range(dnum+20)]
    #     # tmp = txt_list[len(txt_list)-1]
    #     txt_list = [txt_list[i*2] if i<dnum else txt_list[len(txt_list)-1] for i in range(dnum+20)]
    # else:
    #     # tmp = target_list[len(target_list)-1]
    #     target_list = [target_list[i*2] if i<dnum else target_list[len(target_list)-1] for i in range(dnum+1)]
    #     # tmp = txt_list[len(txt_list)-1]
    #     txt_list = [txt_list[i*2] if i<dnum else txt_list[len(txt_list)-1] for i in range(dnum+1)]


    target_list = glob.glob(target_folder+'/*.*', recursive=True)
    txt_list = glob.glob(target_folder+'/../txt/*.*', recursive=True)
    target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    txt_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
    for i, file in enumerate(target_list):        
        print("{}".format(cnt+1))
        source = cv2.imread(source_list[cnt])
        shape = get_face(source)

        target = cv2.imread(file)
        # target = cv2.imread('../data/taylor/land/img/127.jpg')
        target_shape = np.zeros((69, 2), dtype='int')
        
        # f = open('../data/taylor/land/txt/127.txt')
        # f = open(os.path.dirname(file)+'/../txt/'+os.path.basename(file)[:len(os.path.basename(file))-4]+'.txt')
        f = open(txt_list[i])
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

        # jaw = shapes[i][0:17]
        # left_eyebrow = shapes[i][22:27]
        # right_eyebrow = shapes[i][17:22]
        # nose_bridge = shapes[i][27:31]
        # lower_nose = shapes[i][30:35]
        # left_eye = shapes[i][42:48]
        # right_eye = shapes[i][36:42]
        # outer_lip = shapes[i][48:60]
        # inner_lip = shapes[i][60:68]
        # jaw = shapes[i][0:17]
        # left_eyebrow = shapes[i][22:27]
        # right_eyebrow = shapes[i][17:22]
        # nose_bridge = shapes[i][27:31]
        # lower_nose = shapes[i][30:35]
        # left_eye = shapes[i][42:48]
        # right_eye = shapes[i][36:42]
        # outer_lip = shapes[i][48:60]
        # inner_lip = shapes[i][60:68]

        # cv2.polylines(source, [jaw], False, (0, 0, 0), 1)
        # cv2.polylines(source, [left_eyebrow], False, (0, 0, 0), 1)
        # cv2.polylines(source, [right_eyebrow], False, (0, 0, 0), 1)
        # cv2.polylines(source, [nose_bridge], False, (0, 0, 0), 1)
        # cv2.polylines(source, [lower_nose], False, (0, 0, 0), 1)
        # cv2.polylines(source, [left_eye], True, (0, 0, 0), 1)
        # cv2.polylines(source, [right_eye], True, (0, 0, 0), 1)
        # cv2.polylines(source, [outer_lip], True, (0, 0, 0), 1)
        # cv2.polylines(source, [inner_lip], True, (0, 0, 0), 1)

        target = cv2.resize(target, None, fx=w/256, fy=h/256)
        target_shape[:,0] = np.dot(target_shape[:,0], w/256)
        target_shape[:,1] = np.dot(target_shape[:,1], h/256)

        # dst = patch_lip(source, shape2[i], target, target_shape)
        dst = patch_lip(source, shape, target, target_shape)

        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

        ret = np.concatenate((source, dst), axis=1)

        out.write(ret)
        out_convert.write(dst)
        out_original.write(source)
        cv2.imwrite(output_folder+'/img/{}.png'.format(cnt) , ret)
        cnt+=1
    # cv2.imshow('source', ret)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', dst)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
out.release()
out_convert.release()
out_original.release()
cv2.destroyAllWindows()