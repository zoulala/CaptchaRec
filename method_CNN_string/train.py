'''
abst: 4字符验证码图片，端到端CNN训练
time: 2019-03-07
author: zoulingwei

result: acc:50% (样本集1000，太少)
'''
import os
import pickle

import tensorflow as tf
from method_CNN_string.read_util import get_next_batch

from method_CNN_string.model import Model, Config


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
    model.train(get_next_batch, model_path, train_set,test_set)


if __name__ == '__main__':
    tf.app.run()
