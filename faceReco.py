import dlib
import cv2
import  numpy as np
import pandas as pd
import random
import os
from skimage import io
import matplotlib.pyplot as plt

import util

pic_test_origin_path='pic_test_origin'
pic_test_path='pic_test'
size=64
#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
# Dlib 人脸预测器
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Dlib 人脸识别模型
# Face recognition model, the object maps human faces into 128D vectors
face_rec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")



def getFaceRegion(imgName):
    fileName=os.path.join(pic_test_origin_path,imgName)
    img=cv2.imread(fileName)
    # 转为灰度图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用detector进行人脸检测
    dets = detector(gray_img,0)
    outputPath=None
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0

        face = img[x1:y1,x2:y2]
        # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
        # face = util.relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

        face = cv2.resize(face, (size,size))
        outputPath=os.path.join(pic_test_path,imgName)
        cv2.imwrite(outputPath, face)
    return outputPath

def showLandmarks(imgName):
    fileName=os.path.join(pic_test_origin_path,imgName)
    img=cv2.imread(fileName)
    # 转为灰度图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用detector进行人脸检测
    faces = detector(gray_img,0)
    if(len(faces)<=0):
        return
    
    for i in range(len(faces)):
        shapes=predictor(img,faces[i])
        landmarks = np.matrix([[p.x, p.y] for p in shapes.parts()])
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        # print(idx,pos)
        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 2, color=(0, 255, 0))
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.5, (0, 0, 255), 1,cv2.LINE_AA)

    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    

# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    # print("%-40s %-20s" % ("检测到人脸的图像 / image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor

def getDistances(feature):
    distances=[]
    persons=[]
    df=pd.read_csv('feature/features2_all.csv')
    for personName in df.columns:
        baseFeat=df[personName]
        baseFeat=np.array(baseFeat)
        distance=np.sqrt(np.sum(np.power(feature-baseFeat,2)))
        distances.append(distance)
        persons.append(personName)
    return distances,persons

def recognize(distances,persons):
    return persons[np.argmin(distances)]


if __name__=='__main__':
    # showLandmarks('dyx4.jpg')
    img=getFaceRegion('dbc5.jpg')
    feat=return_128d_features(img)
    if(feat!=0):
        feat=np.array(feat)
        distances,persons=getDistances(feat)
        print(list(zip(persons,distances)))
        proposalPerson=recognize(distances,persons)
        print(proposalPerson)




