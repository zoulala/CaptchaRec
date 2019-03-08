
import os

import numpy as np
from PIL import Image

from libs.noise_prc import NoiseDel

nd = NoiseDel()


# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

def cut_box(img,resize=(64,18)):

    # 灰度，二值化
    image = nd.convert2gray(img)
    image = nd.binarizing(image,190, True)

    image = Image.fromarray(image)
    img0 = Image.fromarray(img)
    box = image.getbbox()
    box1 = (box[0]-2,box[1]-2,box[2]+2,box[3]+2)
    image = img0.crop(box1)
    image = image.resize(resize)

    return np.array(image)


def seg_img(img):
    h,w,c = img.shape
    d = int(w/4)
    img_list = []
    for i in range(4):
        img_list.append(img[:,i*d:(i+1)*d])
    return img_list



if __name__ =="__main__":
    filepath = 'data/png'
    filenames = os.listdir(filepath)

    savepath = 'data/png_cut'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    k = 0
    for file in filenames:

        text = file.replace('.png','')

        openname = os.path.join(filepath, file)
        img = Image.open(openname)
        img = np.array(img)

        # 去噪，灰度，二值化
        img = nd.noise_del(img)
        # img = nd.convert2gray(img)
        # img = nd.binarizing(img,190, False)
        img = cut_box(img)

        # img1 = Image.fromarray(img)
        # img1.show()

        img_list = seg_img(img)
        for i in range(4):
            k +=1
            img1 = Image.fromarray(img_list[i])
            img1.show()
            c = text[i]
            if 64<ord(c)<97:
                c = c+'_'
            savepath1 = os.path.join(savepath, '{}/'.format(c))
            if not os.path.exists(savepath1):
                os.mkdir(savepath1)
            savename = os.path.join(savepath1,str(k)+'.png')
            img1.save(savename)
            if k%100==0:
                print(k)
        # print('a')