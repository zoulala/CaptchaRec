'''
abst: 4字符验证码图片、处理为单字符图片集，CNN训练
time: 2019-03-07
author: zoulingwei

result: acc:95% （4000字符样本集，CNN能够识别字符旋转特性）
'''
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from libs.noise_prc import NoiseDel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from libs.cut_prc import cut_box, seg_img

nd = NoiseDel()

class Config():
    file_name = 'char_03'  # 存放模型文件夹

    w_alpha = 0.01  #cnn权重系数
    b_alpha = 0.1  #cnn偏执系数
    image_h = 18
    image_w = 16


    cnn_f = [16,32,32,512]  # [cov1输出特征维度，cov2输出特征维度，cov3输出特征维度，全连接层输出维度]

    max_captcha = 1  #验证码最大长度
    char_set_len = 50  #字符集长度

    lr = 0.001
    batch_size = 32  # 每批训练大小
    max_steps = 200000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model():

    def __init__(self, config):
        self.config = config

        self.input()
        self.cnn()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def input(self):
        self.X = tf.placeholder(tf.float32, [None, self.config.image_h * self.config.image_w])
        self.Y = tf.placeholder(tf.float32, [None, self.config.max_captcha * self.config.char_set_len])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(0, dtype=tf.float32, trainable=False, name="global_loss")

    def cnn(self):
        x = tf.reshape(self.X, shape=[-1, self.config.image_h , self.config.image_w, 1])

        # 3 conv layer
        w_c1 = tf.Variable(self.config.w_alpha * tf.random_normal([3, 3, 1, self.config.cnn_f[0]]))
        b_c1 = tf.Variable(self.config.b_alpha * tf.random_normal([self.config.cnn_f[0]]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        w_c2 = tf.Variable(self.config.w_alpha * tf.random_normal([3, 3, self.config.cnn_f[0], self.config.cnn_f[1]]))
        b_c2 = tf.Variable(self.config.b_alpha * tf.random_normal([self.config.cnn_f[1]]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        w_c3 = tf.Variable(self.config.w_alpha * tf.random_normal([3, 3, self.config.cnn_f[1], self.config.cnn_f[2]]))
        b_c3 = tf.Variable(self.config.b_alpha * tf.random_normal([self.config.cnn_f[2]]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        # Fully connected layer
        h =tf.cast( conv3.shape[1],tf.int32)
        w = tf.cast( conv3.shape[2],tf.int32)
        f = tf.cast( conv3.shape[3],tf.int32)

        w_d = tf.Variable(self.config.w_alpha * tf.random_normal([h* w * f, self.config.cnn_f[3]]))
        b_d = tf.Variable(self.config.b_alpha * tf.random_normal([self.config.cnn_f[3]]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)

        w_out = tf.Variable(self.config.w_alpha * tf.random_normal([self.config.cnn_f[3], self.config.max_captcha * self.config.char_set_len]))
        b_out = tf.Variable(self.config.b_alpha * tf.random_normal([self.config.max_captcha * self.config.char_set_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        # out = tf.nn.softmax(out)

        # loss
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.Y))

        # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss,global_step=self.global_step)

        predict = tf.reshape(out, [-1, self.config.max_captcha, self.config.char_set_len])
        self.max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.config.max_captcha, self.config.char_set_len]), 2)
        correct_pred = tf.equal(self.max_idx_p, max_idx_l)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))


    def train(self, get_next_batch, model_path, x_train, y_train, x_test, y_test):
        with self.session as sess:
            while True:
                batch_x, batch_y = get_next_batch(x_train, y_train,self.config.batch_size)
                _, loss_ = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})

                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {:.4f}... '.format(loss_))

                # 每100 step计算一次准确率
                if self.global_step.eval() % self.config.save_every_n == 0:
                    # batch_x_test, batch_y_test = get_next_batch(100)
                    acc = sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: 1.})
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          '--------- acc: {:.4f} '.format(acc),
                          '   best: {:.4f} '.format(self.global_loss.eval()))

                    if acc > self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, acc)  # 更新最优值
                        sess.run(update)
                        self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)
                    if self.global_step.eval() >= self.config.max_steps:
                        #self.saver.save(sess, os.path.join(model_path, 'model_last'), global_step=self.global_step)
                        break

    def test(self, batch_x_test):
        sess = self.session
        max_idx_p = sess.run(self.max_idx_p, feed_dict={self.X: batch_x_test, self.keep_prob: 1.})
        return max_idx_p


def get_next_batch( train_x, train_y, batch_size=32):
    n = train_x.shape[0]
    chi_list = np.random.choice(n, batch_size)
    return train_x[chi_list],train_y[chi_list]


def img_cut_to_arry(img):
    imgs = []
    image = nd.noise_del(img)
    image = cut_box(image)
    image_list = seg_img(image)
    for im in image_list:
        im = nd.convert2gray(im)
        im = im.reshape(-1)
        imgs.append(im)
    return imgs


if __name__=="__main__":
    # nd = NoiseDel()
    # 获取训练数据
    path = '../data/png_cut'
    classes = os.listdir(path)
    x = []
    y = []
    for c in classes:
        c_f = os.path.join(path, c)
        if os.path.isdir(c_f):
            files = os.listdir(c_f)
            for file in files:
                img = Image.open(os.path.join(c_f, file))
                img = np.array(img)
                img = nd.convert2gray(img)
                img = img.reshape(-1)
                x.append(img)
                y.append(c.replace('_',''))

    lb = LabelBinarizer()
    ly = lb.fit_transform(y)  # one-hot
    x = np.array(x)
    y = np.array(ly)

    # 拆分训练数据与测试数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02)


    # 创建模型目录
    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    # train and val
    print('start to training...')
    model.train(get_next_batch, model_path, x_train,y_train, x_test, y_test)


    # test

    # 预测新样本集
    newpath = '../data/png_new'
    newfiles = os.listdir(newpath)
    for file in newfiles:
        pre_text=''
        image = Image.open(os.path.join(newpath,file))
        image = np.array(image)
        imgs= img_cut_to_arry(image)
        for img in imgs:
            p = model.test([img,])
            p_arr = np.zeros([1,50])
            p_arr[0,p] =1
            c = lb.inverse_transform(p_arr)
            pre_text += c[0]


        print(pre_text)


