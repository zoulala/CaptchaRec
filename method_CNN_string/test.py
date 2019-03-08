import os
import pickle

import tensorflow as tf
from method_CNN_string.read_util import char_set, get_next_batch_one

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

    print('start to test...')
    with open('train_test_set.pkl','rb') as f:
        train_set,test_set = pickle.load(f)
    k = 0
    for ts in test_set:
        batch_x_test, batch_y_test, text = get_next_batch_one(1,[ts])
        max_idx_p = model.test(batch_x_test)
        chars = [char_set[idx] for idx in max_idx_p[0]]
        pre_text = "".join(chars)
        print('predict text:',pre_text)
        if text == pre_text:
            k+=1
    print(k)


if __name__ == '__main__':

    tf.app.run()
