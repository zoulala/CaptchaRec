import os
import pickle
import random

import cv2
import numpy as np

from method_CNN_string.model import Config

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

# 文本转向量, 当前样本只有小写字符
char_set = number + alphabet + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = Config.char_set_len  #len(char_set)


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

        if k>9:
            k = ord(c)-87

        # if k > 9:
        #     k = ord(c) - 55
        #     if k > 35:
        #         k = ord(c) - 61
        #         if k > 61:
        #             raise ValueError('No Map')
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

def split_train_test(filepath, proportion=0.2):
    filenames = os.listdir(filepath)
    filenames = [os.path.join(filepath,file) for file in filenames]
    n = len(filenames)
    k = int(n*(1-proportion))
    random.shuffle(filenames)
    train_set = filenames[:k]
    test_set = filenames[k:]
    return train_set, test_set

def get_image_sample(file):
    image = cv2.imread(file)
    filename = os.path.basename(file)
    text = filename.replace('.png','')
    return text,image

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 生成一个训练batch
def get_next_batch(batch_size=32, filenames=None, test=False):
    n = len(filenames)
    if test:
        batch_size = n
    batch_x = np.zeros([batch_size, Config.image_h * Config.image_w])
    batch_y = np.zeros([batch_size, Config.max_captcha * Config.char_set_len])

    # 有时生成图像大小不是(60, 160, 3)
    for i in range(batch_size):
        if test:k = i
        else:k = random.randint(0,n-1)
        text, image = get_image_sample(filenames[k])
        if batch_size == 1:
            print('test text:', text)

            # f = plt.figure()
            # ax = f.add_subplot(111)
            # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
            # plt.imshow(image)
            # plt.show()
        image = convert2gray(image)

        batch_x[i, :] = (image.flatten()-128)/128 #image.flatten() / 255  #  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def get_next_batch_one(batch_size=32, filenames=None, test=False):
    n = len(filenames)
    if test:
        batch_size = n
    batch_x = np.zeros([batch_size, Config.image_h * Config.image_w])
    batch_y = np.zeros([batch_size, Config.max_captcha * Config.char_set_len])

    # 有时生成图像大小不是(60, 160, 3)
    for i in range(batch_size):
        if test:k = i
        else:k = random.randint(0,n-1)
        text, image = get_image_sample(filenames[k])
        if batch_size == 1:
            print('test text:', text)

            # f = plt.figure()
            # ax = f.add_subplot(111)
            # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
            # plt.imshow(image)
            # plt.show()
        image = convert2gray(image)

        batch_x[i, :] = (image.flatten()-128)/128 #image.flatten() / 255  #  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y,text


if __name__=="__main__":
    train_set, test_set = split_train_test('../data/png')
    with open('../data/train_test_set.pkl', 'wb') as f:
        pickle.dump((train_set, test_set), f)



