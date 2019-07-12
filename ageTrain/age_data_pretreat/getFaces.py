import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
CASE_PATH = "xml/haarcascade_frontalface_default.xml"
CASE_PATH = "xml/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)

def getFace(imgs,labels,flag=0):
    information = []
    for i in range(len(imgs)):
        img = imgs[i]
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,
                                                  scaleFactor=1.2,
                                                  minNeighbors=2,
                                                  minSize=(
                                                  2, 2), )  # 根据检测到的坐标及尺寸裁剪、无形变resize、并送入模型运算，得到结果后在人脸上打上矩形框并在矩形框上方写上识别结果：
            for (x, y, width, height) in faces:
                # cv2.imshow('img', image[y:y + height, x:x + width])
                img = img[y:y + height, x:x + width]
                img=cv2.resize(img,(128,128))
                filename = 'D:/agePicture/'+str(flag)+'/'+'picture'+str(i)+'.jpg'
                information.append([filename,labels[i]])

                cv2.imwrite(filename,img)
        except:
            continue

            #xxx = cv2.imread('sss.jpg')
            # cv2.imwrite('11111.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return information

# def shengCWJ(flag=0,batch=10,data_set=None):
#     age_images = np.zeros([batch, 227, 227, 3])
#     labels =[]
#     begin = flag*batch
#     for i in range(batch):
#         '''首先获取图片的路径，然后读取图片进行放缩存入[1,width,heigth,3]的数组之中'''
#         path=data_set[begin+i].split(' ')[0]
#         image=exchange(path)
#         age_images[i,:,:,:]=cv2.resize(image,(227,227))
#         age = int(data_set[begin+i].split(' ')[2])
#         age_targets[i,age]=1.0
#     age_images=np.array(age_images,dtype='float32')/255.0
#     age_targets=np.array(age_targets,dtype='float32')
#     return age_images,age_targets


f1= open('trainImagesAndLabels.txt','r')
data_set = f1.read().splitlines()
f1.close()


#len(data_set)
for x in range(len(data_set)//1000):
    imgs = []
    labels = []
    flag = x
    for i in range(x*1000,(x+1)*1000):
        first = data_set[i].split(' ')[0]
        img = cv2.imread(first)
        imgs.append(img)
        label = data_set[i].split(' ')[2]
        labels.append(label)

    x = getFace(imgs,labels,flag)

    with open('ageFaces.txt','a+') as f2:
        for i in range(len(x)):
            f2.write(x[i][0])
            f2.write('\t')
            f2.write(x[i][1])
            f2.write('\n')
    f2.close()

#
# r=Image.fromarray(xx[:,:,0]).convert('L')
# g=Image.fromarray(xx[:,:,1]).convert('L')
# b=Image.fromarray(xx[:,:,2]).convert('L')
#
# #b g r
# T2=Image.merge('RGB',(b,g,r))
# #print(type(T2))
#
# plt.imshow(T2)
# # plt.imshow(b)
# # plt.imshow(T2)
# plt.show()
