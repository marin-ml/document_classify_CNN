
"""
    This is code for training CNN for Image Classification
    Required: tensorflow 0.10.0rc0
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import sys
import logging
from classify import FuncMl


def load_training_data(path):
    """
        Load training data from path
    """
    cat_folder, cat_display, cat_id = func_class.load_standard_base_documents_json()

    train_data_x = []
    train_data_y = []
    cat_count = 0

    for cat_index in range(cat_folder.__len__()):
        sub_folder = cat_folder[cat_index]
        path_in = path + '/' + sub_folder
        if os.path.exists(path_in):
            cat_count += 1
            for f_name in os.listdir(path_in):
                img_name = path_in + '/' + f_name
                image_data = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
                training_data = image_data.reshape(224, 224, 1)
                train_data_x.append(training_data)
                train_data_y.append(func_class.expand(cat_id[cat_index], 1000))

    log_print("  Total count of Categories    : " + str(cat_count))
    log_print("  Total count of training data : " + str(len(train_data_x)))
    return train_data_x, train_data_y


def log_print(info_str):
    print info_str
    logging.info(info_str)


""" ------------------ Input argument process --------------------- """
in_arg = ['training',               # training data folder
          'models/classify_CNN',    # model name
          '1000',                   # training step
          '0.001',                  # learning rate
          '0.8',                    # convolution parameter for CNN
          '0.5']                    # hidden parameter for CNN

for arg_ind in range(len(sys.argv) - 1):
    in_arg[arg_ind] = sys.argv[arg_ind + 1]

path_training = in_arg[0]
model_name = in_arg[1]
step = int(in_arg[2])
para_rate = float(in_arg[3])
p_conv = float(in_arg[4])
p_hidden = float(in_arg[5])

""" --------------------- set the log information --------------------------- """
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d/%b/%Y %H:%M:%S',
                    filename='log/training.log')

""" ------------ Loading the training data from selected path --------------- """
log_print("Loading training data ...")
func_class = FuncMl()
[train_x, train_y] = load_training_data(path_training)

""" --------------------- Configuration of CNN model ------------------------ """
print ("Configuration of CNN model ...")
py_x, p_keep_hidden, p_keep_con, X, Y = func_class.model_config()
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.AdamOptimizer(para_rate).minimize(cost_op)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.initialize_all_variables())

""" ------------Loading model weight from previous training result----------- """
print ("Loading model weight from previous training result ...")
try:
    saver.restore(sess, model_name)
except:
    pass

""" ---------------------------- Training data ------------------------------ """
log_print("Training data ...")
data_len = train_y.__len__()
batch_size = 500

for step_i in range(step + 1):
    pred_y = []
    cost = 0
    for sub_step in range(0, data_len, batch_size):
        tr_x = train_x[sub_step:sub_step + batch_size]
        tr_y = train_y[sub_step:sub_step + batch_size]
        ret_cost = sess.run([train_op, cost_op], feed_dict={X: tr_x, Y: tr_y,
                                                            p_keep_con: p_conv, p_keep_hidden: p_hidden})
        cost += ret_cost[1]
        if step_i % 10 == 0:
            ret_y = sess.run(predict_op, feed_dict={X: tr_x, Y: tr_y, p_keep_con: 1, p_keep_hidden: 1})
            pred_y = np.append(pred_y, ret_y)

    if step_i % 10 == 0:
        acc = np.mean(np.argmax(train_y, axis=1) == pred_y)
        log_print('  %s: %d, %s: %f, %s: %.2f' % ("step", step_i, "cost", cost, "accuracy", acc * 100))
        saver.save(sess, model_name)
    else:
        print "  step:", step_i

print ("Optimization Finished!")
