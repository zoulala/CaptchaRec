'''
abst: 自动生成4字符验证码图片（尽量和目标图相似）、端到端CNN训练
time: 2019-03-07
author: zoulingwei

result: acc:40%（自动生成图片质量有待提高）
'''
import os
import pickle

import tensorflow as tf
from method_CNN_string_gen.a_model import Model, Config
from method_CNN_string_gen.a_read_util import get_next_batch, read_test_xy


def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    with open('../data/train_test_set.pkl','rb') as f:
        train_set,test_set = pickle.load(f)
    x_test, y_test = read_test_xy(test_set)  # 测试第三方数据
    model.train(get_next_batch, model_path, x_test, y_test)

if __name__ == '__main__':
    tf.app.run()
