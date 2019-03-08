
'''
abst: 4字符验证码图片、处理为单字符图片集，KNN、SVM,DT,LR训练
time: 2019-03-07
author: zoulingwei

result: acc:80%（字符有旋转特性，KNN特征单一，无法识别旋转性）
'''
import os
import numpy as np
from PIL import Image
from libs.noise_prc import NoiseDel
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from libs.cut_prc import cut_box, seg_img

nd = NoiseDel()
def predict_img(img, clf):
    text = ''
    image = nd.noise_del(img)
    image = cut_box(image)
    image_list = seg_img(image)

    for im in image_list:
        im = nd.convert2gray(im)
        im = im.reshape(-1)
        c = clf.predict([im,])[0]
        text += c
    return text

if __name__=="__main__":

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

    x = np.array(x)
    y = np.array(y)

    # 拆分训练数据与测试数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02)

    # 训练KNN分类器
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train)

    # 预测测试集
    test_pre = clf.predict(x_test)


    print("KNN accuracy score:", (test_pre == y_test).mean())

    # 预测新样本集
    newpath = '../data/png_new'
    newfiles = os.listdir(newpath)
    for file in newfiles:
        image = Image.open(os.path.join(newpath,file))
        image = np.array(image)
        text = predict_img(image, clf)
        print(text)

    # # 训练svm分类器
    # clf = svm.LinearSVC()  ###
    # clf.fit(x_train, y_train)
    #
    # test_pre = clf.predict(x_test)
    # print("SVM accuracy score:", (test_pre == y_test).mean())
    #
    # # 训练dt分类器
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(x_train, y_train)
    #
    # test_pre = clf.predict(x_test)
    # print("DT accuracy score:", (test_pre == y_test).mean())
    #
    # # 训练lr分类器
    # clf = linear_model.LogisticRegression()  ### t
    # clf.fit(x_train, y_train)
    #
    # test_pre = clf.predict(x_test)
    # print("LR accuracy score:", (test_pre == y_test).mean())









