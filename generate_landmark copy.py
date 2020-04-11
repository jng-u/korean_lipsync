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

    # cv2.polylines(img, [jaw], False, (0, 0, 0), 1)
    # cv2.polylines(img, [left_eyebrow], False, (0, 0, 0), 1)
    # cv2.polylines(img, [right_eyebrow], False, (0, 0, 0), 1)
    # cv2.polylines(img, [nose_bridge], False, (0, 0, 0), 1)
    # cv2.polylines(img, [lower_nose], True, (0, 0, 0), 1)
    # cv2.polylines(img, [left_eye], True, (0, 0, 0), 1)
    # cv2.polylines(img, [right_eye], True, (0, 0, 0), 1)
    # cv2.polylines(img, [outer_lip], True, (0, 0, 0), 1)
    # cv2.polylines(img, [inner_lip], True, (0, 0, 0), 1)

    under_shape = shape[2:15]
    outer_lip = shape[48:60]
    inner_lip = shape[60:68]
    
    ##### make landmark image and training image
    landmark = np.zeros(img.shape[:2])
    cv2.polylines(landmark, [under_shape], False, (255, 255, 255), 1)
    cv2.polylines(landmark, [outer_lip], True, (255, 255, 255), 1)
    cv2.polylines(landmark, [inner_lip], True, (255, 255, 255), 1)

    # find canny
    # (mx, my, mw, mh) = np.dot(3, get_rect(inner_lip))
    # inner_lip_crop = img[my:my+mh, mx:mx+mw]
    # mask = np.zeros((mh, mw), dtype=np.uint8)
    # inner_lip_org = np.dot(3, inner_lip)
    # for s in inner_lip_org:
    #     s -= [mx, my]
    # cv2.fillPoly(mask, [inner_lip_org], 255)
    # # mask_erode = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    # sharpening = np.array([[-1, -1, -1, -1, -1],
    #                         [-1, 2, 2, 2, -1],
    #                         [-1, 2, 9, 2, -1],
    #                         [-1, 2, 2, 2, -1],
    #                         [-1, -1, -1, -1, -1]]) / 9.0
    # inner_lip_crop = cv2.filter2D(inner_lip_crop, -1, sharpening)
    # inner_lip_masked = cv2.bitwise_and(inner_lip_crop, inner_lip_crop, mask=mask)

    # # gray_imm = masked_histogram_equalization(inner_lip_masked, mask)

    # # gray_imm = cv2.medianBlur(gray_imm, 3)

    # # ret, gray_imm = cv2.threshold(gray_imm, 159, 255, cv2.THRESH_BINARY)
    # # hori = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1] , [1, 1, 1, 1, 1]])
    # # gray_imm = cv2.erode(gray_imm, np.ones((5, 5), np.uint8), iterations=1)
    # # gray_imm = cv2.dilate(gray_imm, np.ones((5, 5), np.uint8), iterations=1)
    # # gray_imm = cv2.erode(gray_imm, np.ones((5, 5), np.uint8), iterations=1)
    # # gray_imm = cv2.erode(gray_imm, hori, iterations=1)
    # # gray_imm = cv2.dilate(gray_imm, hori, iterations=1)
    # # gray_imm = cv2.erode(gray_imm, hori, iterations=2)
    
    # tooth = cv2.Canny(inner_lip_masked, 100, 150)

    # lines = cv2.HoughLines(tooth, 1, np.pi/2, 20)

    # (x1, y1, x2, y2) = (0, 0, 0, 0)
    # tooth_landmark = np.zeros(img.shape[:2], dtype='uint8')
    # if lines is not None:
    #     for i in range(len(lines)):
    #         for rho, theta in lines[i]:
    #             if theta > 1:
    #                 a = np.cos(theta)
    #                 b = np.sin(theta)
    #                 x0 = a*rho
    #                 y0 = b*rho
    #                 x1 += int(x0 + 1000*(-b))
    #                 x2 += int(x0 - 1000*(-b))
    #                 y1 += int(y0+1000*(a))
    #                 y2 += int(y0 -1000*(a))
    #     x1 /=len(lines)
    #     x2 /=len(lines)
    #     y1 /=len(lines)
    #     y2 /=len(lines)
    #     cv2.line(tooth_landmark, (mx+int(x1), my+int(y1)),(mx+int(x2), my+int(y2)), 255, 1)

    # tooth_landmark = cv2.bitwise_and(tooth_landmark, tooth_landmark, mask=mask)
    # tooth_landmark = cv2.resize(tooth_landmark, None, fx=ratio, fy=ratio)
    # landmark += tooth_landmark

    cv2.imwrite(output_folder+'/{}'.format(os.path.basename(file)) , landmark)
    # cv2.imshow('source', img)
    # cv2.imshow('landmark', landmark)
    # cv2.imshow('inner', inner_lip_masked)
    # cv2.imshow('tooth', tooth)
    # cv2.waitKey(10000)
