import numpy as np
import base64
import cv2
import socket
import os
from io import BytesIO
from threading import Thread

import tensorflow as tf
from PIL import Image
from libs.noise_prc import NoiseDel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from method_CNN_char.model_cnn_train_test import Config,Model,img_cut_to_arry


nd = NoiseDel()


def base64_gif_to_arr(gif_str):
    gif_bytes = base64.b64decode(gif_str)
    bytesIO = BytesIO(gif_bytes)
    gif = Image.open(bytesIO)
    gif.seek(0) # 参数是选择第几张图
    png_im = Image.new("RGB", gif.size)
    png_im.paste(gif)
    np_img = np.array(png_im)
    # gif = cv2.VideoCapture(bytesIO)  # 打开视频文件
    # ret, frame = gif.read()
    return np_img

def base64_png_to_arr(png_str):
    png_bytes = base64.b64decode(png_str)
    png_img = Image.open(BytesIO(png_bytes))
    # png_img.show()
    np_img = np.array(png_img)
    return np_img

def communicate(sk, conn, addr,lb,model):
    print('新建连接：', addr)
    conn.sendall(bytes("欢迎光临厉害的验证码识别器", encoding="utf-8"))
    while True:
        try:
            size = 5000
            data = conn.recv(size)
        except:
            break
        image = base64_gif_to_arr(data)

        # 预测新样本集
        pre_text=''
        imgs = img_cut_to_arry(image)
        for img in imgs:
            p = model.test([img, ])
            p_arr = np.zeros([1, 50])
            p_arr[0, p] = 1
            c = lb.inverse_transform(p_arr)
            pre_text += c[0]
        print(pre_text)
        conn.sendall(pre_text.encode(encoding='utf-8'))
    conn.close()
    print('关闭连接：', addr)





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


    sk = socket.socket()
    sk.bind(("10.12.0.54",1515))
    sk.listen(5)
    print('开始服务...')
    while True:
        c, addr = sk.accept()
        Thread(target=communicate, args=(sk, c, addr,lb,model)).start()
