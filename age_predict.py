import numpy as np
import tensorflow as tf
import age_model
import cv2


class PredictAge():
    def __init__(self):
        self.age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        self.sess = tf.Session()
        with tf.compat.v1.variable_scope('train_age') as scope:
            self.xs = tf.compat.v1.placeholder(tf.float32, [None, 227, 227, 3], name='xInput')
            self.keep_drop = tf.compat.v1.placeholder(tf.float32)
            self.y, self.variables = age_model.convolutional(self.xs, self.keep_drop)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver(self.variables)
        self.saver.restore(self.sess, "savers/train_age29.ckpt")
        print('模型恢复成功')

    def predict_age(self, path=None, image=None):
        print(path)
        print(image)
        if path is not None:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        age_images = cv2.resize(image, (227, 227))
        xxi = np.array(age_images, dtype='float32')/255.0

        xxi = xxi.reshape([1, 227, 227, 3])
        ap = self.sess.run(self.y, feed_dict={self.xs: xxi, self.keep_drop: 1.0}).flatten().tolist()
        ap1 = np.array(ap)
        label = np.argmax(ap1)
        result = self.age_classes[label]
        return result

# #
# f1= open('trainImagesAndLabels.txt','r')
# data_set = f1.read().splitlines()
# f1.close()
# for i in range(10):
#     aaa=data_set[i].split(' ')[0]
#
# # #
#     rs = Msf(path=aaa)
#
#
#     print(rs)

# aaa1=data_set[100].split(' ')[0]
# print(aaa1)
# #
# rs1 = Msf(path=aaa1)
# print(rs1)
#
