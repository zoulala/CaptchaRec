import numpy as np
import base64
import os
import tensorflow as tf
from io import BytesIO
from PIL import Image
from sklearn.externals import joblib
from method_CNN_char.model_cnn_train_test import Config,Model,img_cut_to_arry
from flask import Flask, request, jsonify


app = Flask(__name__)

# 加载标签映射模型
lb = joblib.load('models/label.pkl')
# 加载保存的模型
model = Model(Config)
checkpoint_path = tf.train.latest_checkpoint(os.path.join('models', Config.file_name))
if checkpoint_path:
    model.load(checkpoint_path)


def base64_gif_to_arr(gif_str):
    gif_bytes = base64.b64decode(gif_str)
    bytesIO = BytesIO(gif_bytes)
    gif = Image.open(bytesIO)
    gif.seek(0) # 参数是选择第几张图
    png_im = Image.new("RGB", gif.size)
    png_im.paste(gif)
    png_im.show()
    np_img = np.array(png_im)
    return np_img

def base64_png_to_arr(png_str):
    png_bytes = base64.b64decode(png_str)
    png_img = Image.open(BytesIO(png_bytes))
    # png_img.show()
    np_img = np.array(png_img)
    return np_img



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
    app.run(host='10.12.0.54', port=1313)



