import numpy as np
import base64
import os
import cv2
from io import BytesIO

from flask import Flask, request, jsonify


import tensorflow as tf
from PIL import Image
from noise_prc import NoiseDel
from sklearn.preprocessing import LabelBinarizer
from model_cnn_train_test import Config,Model,img_cut_to_arry

app = Flask(__name__)
nd = NoiseDel()







def bi_img(img, throd1=135, throd2=156):
    n,m = img.shape
    for i in range(n):
        for j in range(m):
            if img[i][j] == 255:
                img[i][j] = 0
            else:
                if img[i][j]< throd1 or img[i][j]>throd2:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
    return img

def reverse_img(img):
    n,m = img.shape
    for i in range(n):
        for j in range(m):
            if img[i][j]==255:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

def base64_to_image(png_str):
    png_bytes = base64.b64decode(png_str)
    png_img = Image.open(BytesIO(png_bytes))
    return png_img

def gen_big_img(image):
    img = image.convert('L')
    nimg = np.array(img)
    bimg = bi_img(nimg)

    big_img = Image.fromarray(bimg, 'L')
    # big_img.show()
    # mathe = np.array(big_img)
    return bimg

def gen_lit_img(image):
    n,m = image.size
    # img.show()
    l_img = image.convert('L')
    nimg = np.array(l_img)
    bimg = bi_img(nimg)
    img_2 = Image.fromarray(bimg, 'L')
    # img_2.show()
    # 分离通道
    r, g, b, a = image.split()
    aa = np.array(a)
    aa = reverse_img(aa)
    va = Image.fromarray(aa, 'L')
    # 粘贴
    new_img = Image.new("L", (n, m), (255,))  # RGBA的真色模式
    new_img.paste(img_2, (0, 0), mask=va)
    # new_img.show()
    temp = np.array(new_img)
    return temp


@app.route('/v2/cyou/captcha/toutiao', methods=['POST'])
def slip_captcha():
    try:
        data = request.get_json()

        big_data = data.get('jpeg')
        lit_data = data.get('png')

        big_data = big_data.split(';base64,')[-1]
        lit_data = lit_data.split(';base64,')[-1]

        big_image = base64_to_image(big_data)
        lit_image = base64_to_image(lit_data)

        big = gen_big_img(big_image)
        temp = gen_lit_img(lit_image)

        methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
        result = cv2.matchTemplate(big,temp,methods[1])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(max_loc, max_val)

        return jsonify({'code': 1, 'message': 'ok', 'data': max_loc[0]})
    except:
        return jsonify({'code': 0, 'message': 'error','data':None})





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
            y.append(c.replace('_', ''))

lb = LabelBinarizer()
ly = lb.fit_transform(y)  # one-hot
x = np.array(x)
y = np.array(ly)

# 创建模型目录
model_path = os.path.join('models', Config.file_name)
if os.path.exists(model_path) is False:
    os.makedirs(model_path)

# 加载上一次保存的模型
model = Model(Config)
checkpoint_path = tf.train.latest_checkpoint(model_path)
if checkpoint_path:
    model.load(checkpoint_path)



@app.route('/v1/cyou/captcha/toutiao', methods=['POST'])
def hello_world():
    try:
        data = request.get_json()
        image_data = data.get('captcha')
        image = base64_gif_to_arr(image_data)
        # 预测新样本集
        pre_text = ''
        imgs = img_cut_to_arry(image)
        for img in imgs:
            p = model.test([img, ])
            p_arr = np.zeros([1, 50])
            p_arr[0, p] = 1
            c = lb.inverse_transform(p_arr)
            pre_text += c[0]
        return jsonify({'code': 1, 'message': pre_text})
    except:
        return jsonify({'code': 0, 'message': 'error'})

if __name__ == '__main__':
    app.run(host='10.12.28.27', port=1313)



