import os
import age_model
import getData as gd
import get_age_set as gt
import tensorflow as tf
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
age_classes =['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_classes = ['f','m']

def exchange(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
def get_one_batch(flag=0,batch=10,data_set=None):
    age_images = np.zeros([batch, 227, 227, 3])
    age_targets = np.zeros([batch, 8])
    begin = flag*batch
    for i in range(batch):
        '''首先获取图片的路径，然后读取图片进行放缩存入[1,width,heigth,3]的数组之中'''
        path=data_set[begin+i].split(' ')[0]
        image=exchange(path)
        age_images[i,:,:,:]=cv2.resize(image,(227,227))
        age = int(data_set[begin+i].split(' ')[2])
        age_targets[i,age]=1.0
    age_images=np.array(age_images,dtype='float32')/255.0
    age_targets=np.array(age_targets,dtype='float32')
    return age_images,age_targets


def shuffxy(image,label):
    ssss = []
    shuffx=[]
    shuffy=[]
    for i in image(len(image)):
        ssss.append(i)
    ssss=random.shuffle(ssss)
    for x in ssss:
        shuffx.append(image[x])
        shuffy.append(label[x])
    return shuffx,shuffy

f1= open('trainImagesAndLabels.txt','r')
data_set = f1.read().splitlines()
f1.close()

# filenames = []
# labels = []
# for i in range(len(data_set)):
#     filenames.append(data_set[i].split(' ')[0])
#     #data_set[i].split(' ')[2]是空格
#     labels.append(data_set[i].split(' ')[2])



testfilenames = []
testlabels = []
# for i in range(len(data_set2)):
#     testfilenames.append(data_set2[i].split(' ')[0])
#     #data_set[i].split(' ')[2]是空格
#     testlabels.append(data_set2[i].split(' ')[2])

batchsize = 32
print(len(data_set)//batchsize)
with tf.compat.v1.variable_scope('train_age') as scope:
    xs = tf.compat.v1.placeholder(tf.float32,[None,227,227,3],name='xInput')
    tf.compat.v1.summary.histogram('train_age/weight', xs)
    keep_drop = tf.compat.v1.placeholder(tf.float32)
    tf.compat.v1.summary.histogram('train_age/Dropout', keep_drop)
    y,variables= age_model.convolutional(xs,keep_drop)

ys = tf.compat.v1.placeholder(tf.float32,[None,8],name='yOutput')
tf.compat.v1.summary.histogram('yOutput', ys)
cross_entropy = -tf.reduce_sum(ys * tf.log(y))
# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=ys)

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y))
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(ys,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#用于存储模型

step = []
loss =[]
accuraterate = []
# loss1=0
saver = tf.train.Saver(variables)

t = 0
with tf.compat.v1.Session() as sess:
    merged_summary_op = tf.compat.v1.summary.merge_all()
    summay_writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)
    summay_writer.add_graph(sess.graph)

    init =tf.global_variables_initializer()
    sess.run(init)

    for x in range(50):
        for i in range(len(data_set)//batchsize):
            #获取32个数据
            age_images,target =get_one_batch(flag=i,batch=batchsize,data_set=data_set)
            if i %50==0:
                t+=1
               # train_accuracy = accuracy.eval(feed_dict={xs: age_images[0:1,:,:,:], ys: target[1], keep_drop: 1.0})
                lossx=sess.run(cross_entropy,feed_dict={xs:age_images,ys:target,keep_drop:0.5})
                loss.append(lossx)
                print('训练误差：', lossx)
                step.append(t)
                acc= sess.run(accuracy,feed_dict={xs:age_images,ys:target,keep_drop:1.0})
                accuraterate.append(acc)
                print('训练集准确率：',acc)
            sess.run(train_step, feed_dict={xs: age_images, ys: target, keep_drop: 0.5})
        print('Epoch:', x)

        # pt = 'train_age'+str(x)+'.ckpt'
        # path = saver.save( sess, os.path.join(os.path.dirname(__file__), 'saver', pt),write_meta_graph=False, write_state=False)
        # print('Saved：',path)
        # print('Epoch:', x)
        # print('\n')

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'savers', 'train_age_lsss.ckpt'),
                      write_meta_graph=False, write_state=False)

    print('Saved：', path)
    print('Epoch:', 30, t)
with open('ageAccuracy.txt','a+') as f2:
    for i in range(len(step)):
        f2.write(str(step[i]))
        f2.write('\t')
        f2.write(str(accuraterate[i]))
        f2.write('\n')

fig = plt.figure()
plt.plot(step, loss)
plt.plot(step, accuraterate)
plt.legend(['loss','accuracy'])
plt.xlabel('step')
plt.ylabel('loss/accuracy')
plt.title('loss and accuracy lines')
plt.show()