import os
import cv2
import numpy as np
import xlrd
from PIL import Image



class ReadFiles():
    '''三种方法，读取多级文件夹下的文件'''

    def recurrence(self, path,file_list):
        '''递归方法'''
        for file in os.listdir(path):
            fs = os.path.join(path, file)
            if os.path.isfile(fs):
                file_list.append(fs)
            elif os.path.isdir(fs):
                self.recurrence(fs, file_list)

    def walk(self, path, file_list):
        '''walk方法'''
        ff = os.walk(path)
        for root, dirs, files in ff:
            for file in files:
                file_list.append(os.path.join(root, file))

    def scandir(self, path,file_list):
        '''scandir方法+递归'''
        for item in os.scandir(path):
            if item.is_dir():
                self.scandir(item.path,file_list)
            elif item.is_file():
                file_list.append(item.path)


class Gif2Png():

    def Image2arr(self,file, k=0):
        im = Image.open(file)
        im.seek(k)  # 参数是选择第几张图
        new_im = Image.new("RGB", im.size)
        new_im.paste(im)
        image = np.array(new_im)
        return image

    def cv2arr(self, file):
        gif = cv2.VideoCapture(file) #打开视频文件
        ret, frame = gif.read()  # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
        return frame


def read_exl_keys(exl_file):
    '''获取验证码图片对应的字符'''
    text_dict = {}
    bk = xlrd.open_workbook(exl_file)
    ex = bk.sheet_by_index(0)
    n = ex.nrows
    for i in range(n):
        row = ex.row_values(i)
        if isinstance(row[1],float):
            key = str(int(row[1]))
            if key not in text_dict:
                text_dict[key] = row[2]
            else:
                raise('数据重复')
    return text_dict


if __name__=="__main__":

    # 获取所有图片文件
    path = 'C:\\Users\\zoulingwei\\Desktop\\captchaRec\\验证码图片1020'
    filenames = []
    rf = ReadFiles()
    rf.recurrence(path, filenames)
    print(len(filenames))


    # 获取验证码图片对应的字符
    text_dict = read_exl_keys('data/验证码整理.xlsx')

    # gif 转为png 并保存
    savepath = os.path.join(os.getcwd(),'data/png')
    gf = Gif2Png()
    for file in filenames:
        image = gf.cv2arr(file)

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        number = os.path.basename(file).replace('.gif','')
        text = text_dict[number]#.lower()
        if len(text) <4:
            print('error')
        filename = text+'.png'
        savename = os.path.join(savepath,filename)
        cv2.imwrite(savename,image)


