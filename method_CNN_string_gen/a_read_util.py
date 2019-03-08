import os

import numpy as np
from PIL import Image
from method_CNN_string_gen.a_model import Config
from libs.noise_prc import NoiseDel

from libs.cut_prc import cut_box
from method_CNN_string_gen.a_gen_captcha import gen_captcha_text_and_image, random_captcha_text

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

# 文本转向量
char_set = number + ALPHABET + alphabet + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = Config.char_set_len  #len(char_set)

nd = NoiseDel()

def text2vec(text):
    text_len = len(text)
    if text_len > Config.max_captcha:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(Config.max_captcha * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, Config.image_h * Config.image_w])
    batch_y = np.zeros([batch_size, Config.max_captcha * Config.char_set_len])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text = random_captcha_text(captcha_size=4)
            image = gen_captcha_text_and_image(text)
            if image.shape == (30, 106, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()

        if batch_size == 1:
            print('test text:', text)
        # image1 = Image.fromarray(image)
        # image1.show()

        # 去噪、灰度、二值化处理
        image = nd.noise_del(image)
        image = cut_box(image, resize=(Config.image_w, Config.image_h))  # 剪切最小框图
        image = nd.convert2gray(image)  # 三维变2维
        # image = nd.binarizing(image,threshold=200, cov=False)

        # image1 = Image.fromarray(image)
        # image1.show()

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def get_text_xy(text):
    batch_x = np.zeros([1, Config.image_h * Config.image_w])
    batch_y = np.zeros([1, Config.max_captcha * Config.char_set_len])

    image = gen_captcha_text_and_image(text)

    # 去噪、灰度、二值化处理
    image = nd.noise_del(image)
    image = cut_box(image, resize=(Config.image_w, Config.image_h))  # 剪切
    image = nd.convert2gray(image)  # 三维变2维
    # image = nd.binarizing(image, threshold=200, cov=False)

    # image1 = Image.fromarray(image)
    # image1.show()

    batch_x[0, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
    batch_y[0, :] = text2vec(text)
    return batch_x,batch_y

# 读取第三方测试数据
def read_test_xy(filenames=None):
    n = len(filenames)
    batch_size = n
    batch_x = np.zeros([batch_size, Config.image_h * Config.image_w])
    batch_y = np.zeros([batch_size, Config.max_captcha * Config.char_set_len])

    def get_image_sample(file):
        # image = cv2.imread(file)
        image = Image.open(file)
        # image.show()
        image = np.array(image)
        filename = os.path.basename(file)
        text = filename.replace('.png', '')
        return text, image

    for i in range(batch_size):
        text, image = get_image_sample(filenames[i])
        # 去噪、灰度、二值化处理
        image = nd.noise_del(image)
        image = cut_box(image, resize=(Config.image_w, Config.image_h))  # 剪切
        image = nd.convert2gray(image)  # 三维变2维
        # image = nd.binarizing(image, threshold=200, cov=False)

        # image1 = Image.fromarray(image)
        # image1.show()

        # batch_x[i, :] = (image.flatten() - 128) / 128  # image.flatten() / 255  #  mean为0
        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


if __name__=="__main__":
    get_next_batch(6)
    # get_text_xy('hMk4')
    import pickle
    with open('train_test_set.pkl','rb') as f:
        train_set,test_set = pickle.load(f)

    _ ,_ = read_test_xy(test_set)

