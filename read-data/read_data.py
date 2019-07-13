import numpy as np
from utils import load_data
from keras.utils import np_utils
from pathlib import Path
from matplotlib import pyplot as plt

input_path = "outputdata128.mat"
output_path = Path(__file__).resolve().parent.joinpath("a")
output_path.mkdir(parents=True, exist_ok=True)
image, gender, age, _, image_size, _ = load_data(input_path)
X_data = image
y_data_g = np_utils.to_categorical(gender, 2)	#one-hot
y_data_a = np_utils.to_categorical(age, 101)	#one-hot

#测试
print("单张图片的像素值和通道数目{}".format(X_data[1].shape))
#plt.imshow(X_data[2])
#plt.show()
print("图片总数以及性别分类个数{}".format(y_data_g.shape))
print("图片总数以及年龄分类个数{}".format(y_data_a.shape))
