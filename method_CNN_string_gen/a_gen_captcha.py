import random

# from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image

from method_CNN_string_gen.a_images import ImageCaptcha

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set=number + alphabet + ALPHABET
fonts = ["../font/TruenoBdOlIt.otf", "../font/AsimovOu.otf", "../font/STCAIYUN.TTF"]
Captcha = ImageCaptcha(width=106, height=30, fonts=[fonts[1]], font_sizes=(16, 16, 17))

# 验证码一般都无视大小写；验证码长度默认4个字符
def random_captcha_text(char_set=char_set, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image(text):
    text = ''.join(text)
    captcha = Captcha.generate(text)
    #image.write(captcha, 'data/png_t/'+captcha_text + '.png')  # 写到文件

    image = Image.open(captcha)
    image = np.array(image)
    return image


if __name__ == '__main__':
    # 测试
    # text, image = gen_captcha_text_and_image()
    for i in range(100):
        text = random_captcha_text(char_set,4)
        image = gen_captcha_text_and_image(text)

