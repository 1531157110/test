import numpy as np
from utils import load_data
from keras.utils import np_utils
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
CASE_PATH = "xml/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)
def getPicLab():
    input_path = "outputdata.mat"
    output_path = Path(__file__).resolve().parent.joinpath("a")
    output_path.mkdir(parents=True, exist_ok=True)
    image, gender, age, _, image_size, _ = load_data(input_path)
    #X_data中存储了像素值 32*32*3
    #y_data_g中存储了性别 长度为2
    #y_data_a中存储了年龄 长度为101
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    return X_data,y_data_g

def resize_without_deformation(image, size = (32, 32)):
    height, width,_ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left
    image_with_border = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])
    resized_image = cv2.resize(image_with_border, size)
    return resized_image

def getFilterPicLab():
    a, b = getPicLab()
    count = 0
    IMAGE_SIZE = 32
    filtera, filterb = [], []
    for i in range(35000):
        gray = cv2.cvtColor(a[i], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.2,
                                              minNeighbors=1,
                                              minSize=(
                                                  2, 2), )
        if np.shape(faces) != (0,):
            for (x, y, width, height) in faces:

                img = gray[y:y + height, x :x + width]
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img=np.reshape(img,(32,32,1))
            filtera.append(img)
            filterb.append(b[count])

        count += 1
    # print(np.shape(filtera))
    return filtera,filterb
if __name__ == '__main__':
    filtera,filterb=getFilterPicLab()
    print(np.shape(filtera))
    print(np.shape(filterb))
    for i in range(10):
        if filterb[i][1] == 1:
            cv2.imshow("",filtera[i])
            # cv2.imshow("aa", a[i])
            cv2.waitKey(0)
            #
            # plt.imshow(filtera[i])
            # print(i)
            # # plt.imshow(a[1])
            # plt.show()
