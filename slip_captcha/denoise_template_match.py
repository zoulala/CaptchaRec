
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask,request,jsonify

app = Flask(__name__)

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

def gen_big_img(filename):
    img = Image.open(filename)
    img.show()
    img = img.convert('L')
    nimg = np.array(img)
    bimg = bi_img(nimg)

    big_img = Image.fromarray(bimg, 'L')
    big_img.show()
    # mathe = np.array(big_img)
    return bimg

def gen_lit_img(filename):
    img = Image.open(filename)
    n,m = img.size
    # img.show()
    l_img = img.convert('L')
    nimg = np.array(l_img)
    bimg = bi_img(nimg)
    img_2 = Image.fromarray(bimg, 'L')
    # img_2.show()
    # 分离通道
    r, g, b, a = img.split()
    aa = np.array(a)
    aa = reverse_img(aa)
    va = Image.fromarray(aa, 'L')
    # 粘贴
    new_img = Image.new("L", (n, m), (255,))  # RGBA的真色模式
    new_img.paste(img_2, (0, 0), mask=va)
    new_img.show()
    temp = np.array(new_img)
    return temp


@app.route('/v2/cyou/captcha/toutiao', methods=['POST'])
def slip_captcha():
    data = request.get_json()

    big_img = data['big_img']
    lit_img = data['lit_img']

    big = gen_big_img(filename0)
    temp = gen_lit_img(filename1)

    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    result = cv2.matchTemplate(big,temp,methods[1])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(max_loc, max_val)

    return jsonify({'code': code, 'message': message})


if __name__=="__main__":
    path = r"C:\Users\zoulingwei\Desktop\captchaRec\RR"

    file_list = os.listdir(path)

    filename0 = os.path.join(path,file_list[2])
    filename1 = os.path.join(path,file_list[3])

    big = gen_big_img(filename0)
    temp = gen_lit_img(filename1)


    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    result = cv2.matchTemplate(big,temp,methods[1])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(max_loc, max_val)