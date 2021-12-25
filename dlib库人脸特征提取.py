# -*- coding:utf-8 -*-
# @Date :2021/12/24 10:28
# @Author:KittyLess
# @name: dlib库人脸特征提取

import cv2
import dlib
import os
import sys
import random

# 存储位置
output_dir = 'e:/PycharmProj/DlibStudy/person/person1'
size = 128

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 改变图片的亮度与对比度
def relight(img,light=1,bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0,w):
            for j in range(0,h):
                for c in range(3):
                    tmp = int(img[j,i,c]*light + bias)
                    if tmp > 255:
                        tmp = 255
                    elif tmp < 0:
                        tmp = 0
                    img[j,i,c] = tmp
    return img

# 特征提取器
detector = dlib.get_frontal_face_detector()
# 打开摄像头
camera = cv2.VideoCapture(0)

index = 1
while True:
    if (index <= 15):
        print('Being processed picture %s' % index)
        # 从摄像头读取照片
       # success, img = camera.read()
        img = cv2.imread('./person/wulei.jpg')
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector 进行人脸检测
        dets = detector(gray_img,1)

        for i,d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            face = cv2.resize(face, (size, size))

            cv2.imshow('image', face)

            cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)

            index += 1

        key = cv2.waitKey(1000) & 0xff
        if key == 27:
            break
    else:
        print('Finished!!')
        # 释放摄像头
        camera.release()
        # 删除建立的窗口
        cv2.destroyAllWindows()
        break
