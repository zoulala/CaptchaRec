import os
import numpy as np
from PIL import Image

class NoiseDel():
    #去除干扰噪声
    def noise_del(self,img):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        # 清除四周噪点
        for row in [0,height-1]:
            for column in range(0, width):
                if img[row, column, 0] == 0 and img[row, column, 1] == 0:
                    img[row, column, 0] = 255
                    img[row, column, 1] = 255
        for column in [0,width-1]:
            for row in range(0, height):
                if img[row, column, 0] == 0 and img[row, column, 1] == 0:
                    img[row, column, 0] = 255
                    img[row, column, 1] = 255

        # 清除中间区域噪点
        for row in range(1,height-1):
            for column in range(1,width-1):
                if img[row, column, 0] == 0 and img[row, column, 1] == 0:
                    a = img[row - 1, column]  # 上
                    b = img[row + 1, column]  # 下
                    c = img[row, column - 1]  # 左
                    d = img[row, column + 1]  # 右
                    ps = [p for p in [a, b, c, d] if 1 < p[1] < 255]
                    # 如果上下or左右为白色,设置白色
                    if  (a[1]== 255 and b[1]== 255) or (c[1]== 255 and d[1]== 255):
                        img[row, column, 0] = 255
                        img[row, column, 1] = 255
                    # 设置灰色

                    elif len(ps)>1:
                        kk = np.array(ps).mean(axis=0)
                        img[row, column, 0] = kk[0]
                        img[row, column, 1] = kk[1]
                        img[row, column, 2] = kk[2]
                    else:
                        img[row, column, 0] = 255
                        img[row, column, 1] = 255
        return img

    # 灰度化
    def convert2gray(self,img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            # 上面的转法较快，正规转法如下
            # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    # 二值化
    def binarizing(self,img,threshold, cov=False):
        w, h = img.shape
        if cov:
            for y in range(h):
                for x in range(w):
                    if img[x, y] > threshold:
                        img[x, y] = 0
                    else:
                        img[x, y] = 255
        else:
            for y in range(h):
                for x in range(w):
                    if img[x, y] < threshold:
                        img[x, y] = 0
                    else:
                        img[x, y] = 255
        return img


if __name__=="__main__":

    filepath = 'data/png'
    savepath = 'data/png_a2'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    filenames = os.listdir(filepath)
    nd = NoiseDel()
    for file in filenames:
        openname = os.path.join(filepath,file)
        image = Image.open(openname)
        image = np.array(image)

        # 去噪、灰度、二值化处理
        image = nd.noise_del(image)
        image = nd.convert2gray(image)
        image = nd.binarizing(image,threshold=190, cov=True)

        #
        image = Image.fromarray(image).convert('L')  # Image.fromarray(image)默认转换到‘F’模式，即浮点型，‘L’是整形

        savename = os.path.join(savepath,file)
        image.save(savename)



