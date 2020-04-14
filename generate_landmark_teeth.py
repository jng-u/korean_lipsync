import os
import glob
import dlib
import cv2
import numpy as np
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

err_list = []

file_list = glob.glob(input_folder+'/**/*.*', recursive=True)
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
        err_list.insert(len(err_list), file)
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
    cv2.polylines(landmark, [under_shape], False, (255, 255, 255), 1)
    cv2.polylines(landmark, [outer_lip], True, (255, 255, 255), 1)
    cv2.polylines(landmark, [inner_lip], True, (255, 255, 255), 1)
    
    # find canny
    (mx, my, mw, mh) = get_rect(inner_lip)
    inner_lip_crop = img[my:my+mh, mx:mx+mw]
    mask = np.zeros((mh, mw), dtype=np.uint8)
    inner_lip_s = np.array(inner_lip)
    for s in inner_lip_s:
        s -= [mx, my]
    cv2.fillPoly(mask, [inner_lip_s], 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    sharpening = np.array([[-1, -1, -1, -1, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, 2, 9, 2, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, -1, -1, -1, -1]]) / 9.0
    inner_lip_crop = cv2.filter2D(inner_lip_crop, -1, sharpening)
    inner_lip_masked = cv2.bitwise_and(inner_lip_crop, inner_lip_crop, mask=mask)
    
    tooth = cv2.Canny(inner_lip_masked, 100, 120)
    tooth = cv2.bitwise_and(tooth, tooth, mask=mask)

    lines = cv2.HoughLines(tooth, 1, np.pi/180, 10)
    (x1, y1, x2, y2) = (0, 0, 0, 0)
    tooth_landmark = np.zeros(img.shape[:2], dtype='uint8')
    if lines is not None:
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                if theta > 1:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 += int(x0 + 1000*(-b))
                    x2 += int(x0 - 1000*(-b))
                    y1 += int(y0+1000*(a))
                    y2 += int(y0 -1000*(a))
        x1 /=len(lines)
        x2 /=len(lines)
        y1 /=len(lines)
        y2 /=len(lines)
        cv2.line(tooth_landmark, (mx+int(x1), my+int(y1)),(mx+int(x2), my+int(y2)), 255, 1)
        # cv2.line(img, (mx+int(x1), my+int(y1)),(mx+int(x2), my+int(y2)), 255, 1)

    mask = np.zeros(landmark.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [inner_lip], 255)
    tooth_landmark = cv2.bitwise_and(tooth_landmark, tooth_landmark, mask=mask)
    landmark += tooth_landmark

    cv2.imwrite(output_folder+'/{}'.format(os.path.basename(file)) , landmark)
    # cv2.imshow('source', img)
    # cv2.imshow('landmark', landmark)
    # while(True):
    #     if(cv2.waitKey(10) != -1):
    #         break

print('<<error file list>>')
print(err_list)