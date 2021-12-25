# -*- coding:utf-8 -*-
# @Date :2021/12/24 11:05
# @Author:KittyLess
# @name: dlib库人脸关键点检测

# 导入工具包
from collections import OrderedDict
import numpy as np
import argparse
import dlib
import cv2

# 第一步：参数设置
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', default='shape_predictor_68_face_landmarks.dat',
                help='path to facial landmark predictor')
ap.add_argument('-i', '--image', default='./person/meixi.jpg',
                help='path to input image')
args = vars(ap.parse_args())

# 第二步：使用OrderedDict构造脸部循环字典时是有序的
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
 ("right_eye", (2, 3)),
 ("left_eye", (0, 1)),
 ("nose", (4)),
])

def shape_to_np(shape, dtype='int'):
    # 创建68*2用于存放坐标
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # 创建两个copy
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    # 设置一些颜色区域
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        # 得到每一个点的坐标
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        if name == 'jaw':
            # 用线条连起来
            for l in range(1, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                # 对位置进行连线
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        else:
            # 使用cv2.convexHull获得位置的凸包位置
            hull = cv2.convexHull(pts)
            # 使用cv2.drawContours画出轮廓图
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    return output
# 第三步：加载人脸检测与关键点定位
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

# 读取输入数据，预处理，进行图像的维度重构和灰度化
image = cv2.imread(args['image'])
(h, w) = image.shape[:2]
width = 500
r = width / float(w)
dim = (width, int(r*h))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 第四步：进行人脸检测，获得人脸框的位置信息
rects = detector(gray, 1)
print(rects)
# 遍历检测到的框
clone = image.copy()
for (i, rect) in enumerate(rects):
    # 将检测到的框画出来
    y1 = rect.top() if rect.top() > 0 else 0
    y2 = rect.bottom() if rect.bottom() > 0 else 0
    x1 = rect.left() if rect.left() > 0 else 0
    x2 = rect.right() if rect.right() > 0 else 0
    #第五步： 对人脸框进行关键点定位
    shape = predictor(gray, rect)
    #第六步：将检测到的关键点转换为ndarray格式
    shape = shape_to_np(shape)
    # 第七步：对字典进行循环
    for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
        drawRect = cv2.rectangle(clone,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2)
        # 根据位置画点
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

        #第八步： 使用cv2.boundingRect获得脸部轮廓位置信息
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        # 第九步：根据位置提取脸部的图片
        roi = image[y:y+h, x:x+w]
        (h, w) = roi.shape[:2]
        width = 250
        r = width / float(w)
        dim = (width, int(r*h))
        roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
# 第十步：进行画图操作显示每个部分
        cv2.imshow('Image', clone)
        cv2.waitKey(0)
    # 第十一步: 进行脸部位置的画图，如果是脸部：进行连线操作，如果是其他位置，使用cv2.convexHull()获得凸包的位置信息，进行drawcontour进行画图
    #output = visualize_facial_landmarks(image, shape)
    #cv2.imshow('Image', output)
    #cv2.waitKey(0)