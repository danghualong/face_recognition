#pip install opencv-python
import cv2
import dlib
import os
import sys
import random

import util

# 爬取的图片文件路径
input_dir='pic'
# 存储位置
output_dir = 'pic_data'
size = 64
 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

 
#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

def captureImg():
    # 打开摄像头 参数为输入流，可以为摄像头或视频文件
    camera = cv2.VideoCapture(0)
    #camera = cv2.VideoCapture('C:/Users/CUNGU/Videos/Captures/wang.mp4')


    index = 1
    while True:
        if (index <= 15):#存储15张人脸特征图像
            print('Being processed picture %s' % index)
            # 从摄像头读取照片
            success, img = camera.read()
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测
            dets = detector(gray_img, 1)
    
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
    
                face = img[x1:y1,x2:y2]
                # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
    
                face = cv2.resize(face, (size,size))
    
                cv2.imshow('image', face)
    
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
    
                index += 1
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            print('Finished!')
            # 释放摄像头 release camera
            camera.release()
            # 删除建立的窗口 delete all the windows
            cv2.destroyAllWindows()
            break

def getFiles(curDir):
    files={}
    dirs=os.listdir(curDir)
    for subdir in dirs:
        tmpdir=os.path.join(curDir,subdir)
        fileNames= os.listdir(tmpdir)
        fileList=[]
        for fileName in fileNames:
            fileList.append(os.path.join(tmpdir,fileName))
        files[subdir]=fileList
    return files


def handleImg(fileName,output_dir):
    img=cv2.imread(fileName)
    print(fileName)
    # 转为灰度图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用detector进行人脸检测
    dets = detector(gray_img,1)
    
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0

        face = img[x1:y1,x2:y2]
        # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
        face = util.relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

        face = cv2.resize(face, (size,size))

        cv2.imshow('image', face)

        index=len(os.listdir(output_dir))
        index+=1
        cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)


def handleImgs(input_dir,output_dir):
    files=getFiles(input_dir)
    for key in files:
        sub_output_dir=os.path.join(output_dir,key)
        if(not os.path.exists(sub_output_dir)):
            os.makedirs(sub_output_dir)
        fileNames=files[key]
        for fileName in fileNames:
            handleImg(fileName,sub_output_dir)


def handleOnePersonImgs(input_dir,personName,output_dir):
    sub_output_dir=os.path.join(output_dir,personName)
    if(not os.path.exists(sub_output_dir)):
        os.makedirs(sub_output_dir)
    sub_input_dir=os.path.join(input_dir,personName)
    files=os.listdir(sub_input_dir)
    for file in files:
        filePath=os.path.join(sub_input_dir,file)
        handleImg(filePath,sub_output_dir)


# handleImgs(input_dir,output_dir)
# handleOnePersonImgs(input_dir,'dangyuxuan',output_dir)
handleOnePersonImgs(input_dir,'dangbingchen',output_dir)

# import matplotlib.pyplot as plt
# img=plt.imread('pic/dangyuxuan/IMG_20180430_101724.jpg')
# plt.imshow(img)
# plt.show()



    