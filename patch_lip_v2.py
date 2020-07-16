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
        print('error finding face')
        return None

    rects[0] = dlib.scale_rect(rects[0], 1/ratio)
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    rect = dlib.rectangle(x-10, y-10, x+w+10, y+h+10)   
    
    # 얼굴이 하나임을 가정
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

def rotateImage(input, alpha=0, beta=0, gamma=0, dx=0, dy=0, dz=0, rotation_matrix=None):
    # alpha = (alpha)*math.pi/180
    # beta = (beta)*math.pi/180
    # gamma = (gamma)*math.pi/180

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
    if rotation_matrix is not None:
        # Composed rotation matrix with (RX, RY, RZ)
        R = np.zeros((4, 4))
        R[:3, :3] = rotation_matrix
        R[3, 3] = 1
        # R = rotation_matrix
    else :
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
    output = cv2.warpPerspective(input, trans, (w, h))
    return output

def patch_lip(source, shape, target, target_shape, trr):    
    shape = np.array(shape, dtype='int')

    # theta = 1.5*(trr-2.13)
    mat = np.zeros((3, 3))
    # trr[0] = 0.0
    # trr[1] = -trr[1]
    # trr[2] -= 0.785
    # trr[2] = -trr[2]
    # trr[0] += 1.57
    # trr[1] -= 0.785
    # trr[2] -= 1.57        
    cv2.Rodrigues(trr, mat)

    sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.math.atan2(mat[2, 1], mat[2, 2])
        y = np.math.atan2(-mat[2, 0], sy)
        z = np.math.atan2(mat[1, 0], mat[0, 0])

    else:
        x = np.math.atan2(-mat[1, 2], mat[1, 1])
        y = np.math.atan2(-mat[2, 0], sy)
        z = 0

    # print(trr)
    # print([x, y, z])
    # print(rm)
    p1 = shape[48]
    p2 = shape[54]
    vw = [a-b for a,b in zip(p2, p1)]
    w = math.sqrt(vw[0]**2+vw[1]**2)
    # w = w/math.cos(theta)
    tp1 = target_shape[48]
    tp2 = target_shape[54]
    tvw = [a-b for a,b in zip(tp2, tp1)]
    tw = math.sqrt(tvw[0]**2+tvw[1]**2)
    scale = w/tw
    target = cv2.resize(target, None, fx=scale, fy=scale)
    target_shape[:,:] = target_shape[:,:]*scale

    # target = rotateImage(target, rotation_matrix=rm)
    # target = rotateImage(target, alpha = 0, beta = y-0.785, gamma = 0)
    # cv2.imshow('ssasds', source)
    # cv2.imshow('asds', target)
    # while(True):
    #     if cv2.waitKey(1)!=-1:
    #         break

    span_target = np.zeros(shape=source.shape, dtype='uint8')
    ctp = middle_point(target_shape[48], target_shape[54])
    l = math.sqrt(sum([x**2 for x in ctp-target_shape[33]]))
    # v = shape[4] - middle_point(middle_point(shape[0], shape[1]), middle_point(shape[2], shape[3]))
    v = middle_point(shape[48], shape[54]) - shape[33]
    th = math.atan(v[1]/v[0])
    v = v / math.sqrt(sum([x**2 for x in v]))
    v = (int(round(l*math.cos(th)*v[0])), abs(int(round(l*math.sin(th)*v[1]))))
    # cp = shape[4] + v
    cp = shape[33] + v
    # cp = middle_point(shape[48], shape[54])
    hs=cp[1]-ctp[1] if cp[1]-ctp[1]>=0 else 0
    he=cp[1]-ctp[1]+target.shape[0] if cp[1]-ctp[1]+target.shape[0]<=span_target.shape[0] else span_target.shape[0]
    ws=cp[0]-ctp[0] if cp[0]-ctp[0]>=0 else 0
    we=cp[0]-ctp[0]+target.shape[1] if cp[0]-ctp[0]+target.shape[1]<=span_target.shape[1] else span_target.shape[1]
    span_target[hs:he, ws:we] = target[0:he-hs,0:we-ws]
    for s in target_shape:
        s += [cp[0]-ctp[0], cp[1]-ctp[1]]
    # cv2.imshow('asd', span_target)
    # while(True):
    #     if cv2.waitKey(1)!=-1:
    #         break
    
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
    mm = cv2.fillPoly(mm, [shape[2:15]], 255)
    mask = cv2.bitwise_and(mask, mask, mask=mm)

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

source_list = glob.glob(source_folder+'/**/*.*', recursive=True)
source_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))
target_list = glob.glob(target_folder+'/**/*.*', recursive=True)
target_list.sort(key=lambda file : int(os.path.basename(file)[:len(os.path.basename(file))-4]))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
tmp = cv2.imread(source_list[0])
out = cv2.VideoWriter(output_folder+'/output.avi', fourcc, 24.0, (tmp.shape[1]*2, tmp.shape[0]))

shapes = []
ws = []
for i in range(len(target_list)):
    print("reading file: %s" % source_list[i])
    source = cv2.imread(source_list[i])
    shape = get_face(source)
    shapes.append(shape)
    # p1 = middle_point(shape[0], shape[1])
    # p2 = middle_point(shape[2], shape[3])
    # vw = [a-b for a,b in zip(p2, p1)]
    # w = math.sqrt(vw[0]**2+vw[1]**2)
    # ws.append(w)

# shape2 = []
# shape2.append(shapes[0])
# for i in range(1, len(shapes)-1):
#     w = (shapes[i-1]+shapes[i+1])/2
#     shape2.append(shapes[i]*0.8 + w*0.2)
# shape2.append(shapes[len(shapes)-1])

# shapes = np.array(shapes)
# iter = int(round(len(target_list))/2)
# for i in range(0, len(shapes[0])):
#     shapes[:, i, 0] = curve_fitting(len(shapes), shapes[:, i, 0], iter)
#     shapes[:, i, 1] = curve_fitting(len(shapes), shapes[:, i, 1], iter)

# ws = curve_fitting(len(ws), ws, iter)
    
print("start making video")    
for i, file in enumerate(target_list):
    print("{} / {}".format(i+1, len(target_list)))
    source = cv2.imread(source_list[i])
    target = cv2.imread(file)
    # target = cv2.imread('../data/taylor/land/img/127.jpg')
    target_shape = np.zeros((69, 2), dtype='int')
    
    # f = open('../data/taylor/land/txt/127.txt')
    f = open(os.path.dirname(file)+'/../txt/'+os.path.basename(file)[:len(os.path.basename(file))-4]+'.txt')
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

    size = source.shape

    image_points = np.array([shapes[i][x] for x in TRACKED_POINTS], dtype="double")

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



    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], 
                            dtype = "double")

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    retval, rotation_vector, translation_vector = cv2.solvePnP( model_points, image_points, camera_matrix, dist_coeffs)

    axis = np.float32([[100,0,0], 
                        [0,100,0], 
                        [0,0,100]])
    
    # cv2.Rodrigues(rotation_vector)
    (imgpts, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    sellion_xy = (int(image_points[7][0]), int(image_points[7][1]))


    # dst = patch_lip(source, shape2[i], target, target_shape)
    dst = patch_lip(source, shapes[i], target, target_shape, rotation_vector)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)


    cv2.line(source, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(source, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
    cv2.line(source, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED


    ret = np.concatenate((source, dst), axis=1)

    out.write(ret)
    cv2.imwrite(output_folder+'/img/{}.png'.format(os.path.basename(file)[:len(os.path.basename(file))-4]) , ret)
    # cv2.imshow('source', ret)
    # cv2.imshow('target', target)
    # cv2.imshow('copy', dst)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break
out.release()
cv2.destroyAllWindows()