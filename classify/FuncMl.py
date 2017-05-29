# updated in 4/20/2017

import numpy as np
import cv2
from wand.image import Image
import json
import os
try:
    import tensorflow as tf
except ImportError:
    pass


class FuncMl:

    def __init__(self):
        # Constructor for the class
        self.my_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.categories = 20

    def load_standard_base_documents_json(self):
        """
            load the standard base documents json file and return the folder and display name of categories.
        """
        standard_file = open(os.path.join(self.my_dir, "config/categories.json"))
        standard_data = json.load(standard_file)
        cat_display = []
        cat_folder = []
        cat_id = []
        for standard_list in standard_data:
            cat_folder.append(standard_list)
            cat_display.append(standard_data[standard_list]['display_name'])
            cat_id.append(standard_data[standard_list]['id'])

        return cat_folder, cat_display, cat_id

    def get_file_list(self, root_dir):
        """
            get all files in root_dir directory
        """
        path_list = []
        file_list = []
        join_list = []
        for path, _, files in os.walk(root_dir):
            for name in files:
                path_list.append(path)
                file_list.append(name)
                join_list.append(os.path.join(path, name))

        return path_list, file_list, join_list

    def pdf2jpg(self, in_name, out_name, only_first_page):
        """
            convert the pdf file to jpg file
                :param in_name:             pdf file name
                :param out_name:            jpg file name
                :param only_first_page:     convert only first page or whole page
        """

        if only_first_page:
            img = Image(filename=in_name+'[0]')       # convert the first page of pdf to jpg
        else:
            img = Image(filename=in_name)             # convert the whole page of pdf to jpg
        converted = img.convert('jpg')
        converted.save(filename=out_name)

    def expand(self, number, width):
        s = np.zeros(width)
        s[number - 1] = 1
        return s

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))
    
    def model_CNN(self, X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
        """
            Image classification CNN model
        """
        l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 2, 2, 1], padding='SAME'))          # l1a shape=(?,224, 224, 32)
        l1 = tf.nn.max_pool(l1a, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')  # l1 shape=(?, 38, 38, 32)
        l1 = tf.nn.dropout(l1, p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))        # l2a shape=(?, 38, 38, 64)
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # l2 shape=(?, 19, 19, 64)
        l2 = tf.nn.dropout(l2, p_keep_conv)

        l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))        # l3a shape=(?, 19, 19, 128)
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # l3 shape=(?, 10, 10, 128)

        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])                              # reshape to (?, 12800)
        l3 = tf.nn.dropout(l3, p_keep_conv)

        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        pyx = tf.matmul(l4, w_o)
        return pyx

    def model_config(self):
        """ --------------------- Configuration of CNN model ------------------------ """
        tf.reset_default_graph()

        X = tf.placeholder("float", [None, 224, 224, 1])
        Y = tf.placeholder("float", [None, self.categories])

        w1 = self.init_weights([3, 3, 1, 16])               # 3x3x1 conv, 16 outputs
        w2 = self.init_weights([3, 3, 16, 32])              # 3x3x16 conv, 32 outputs
        w3 = self.init_weights([3, 3, 32, 64])              # 3x3x32 conv, 64 outputs
        w4 = self.init_weights([6400, 1024])                # 64*10*10 input, 1024 outputs
        w5 = self.init_weights([1024, self.categories])     # 1024 inputs, 20 outputs (labels)

        p_keep_con = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")
        py_x = self.model_CNN(X, w1, w2, w3, w4, w5, p_keep_con, p_keep_hidden)

        return py_x, p_keep_hidden, p_keep_con, X, Y

    def standardization(self, data):
        ret = []
        list_sum = 0

        for i in data:
            list_sum += i
    
        for i in data:
            ret.append(float(i)/list_sum*100)
    
        return ret
    
    def load_convert_image(self, img_file, img_size=224):
        """
            load the image file as gray and resize it.
        """
        ori_img = cv2.imread(img_file, 0)
        if ori_img is not None:
            new_img = cv2.resize(ori_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        else:
            new_img = None

        return new_img

    def get_classify_result(self, acc_list, cat_list, cat_id, divider):
        """
            display the accuracy data as high level ascend
        """
        comb_acc = []
        for j in range(cat_list.__len__()):
            comb_acc.append([acc_list[cat_id[j] - 1] / divider, cat_list[j], cat_id[j]])
    
        comb_acc.sort(reverse=True)

        return comb_acc

    def get_match_files(self, p_cat_folder, p_list_file, p_list_join, path_in, path_out):
        """
            get matching file lists
        """
        match_path_train = []
        match_file = []
        match_join = []
        for i in range(len(p_list_file)):
            for cat_list in p_cat_folder:
                s1 = cat_list
                s2 = p_list_join[i][len(path_in) + 1:len(path_in) + len(s1) + 1]
                f_name = p_list_file[i].upper()
                if s1.upper() == s2.upper():
                    if f_name[-4:] == '.BMP' or f_name[-4:] == '.JPG' or f_name[-4:] == '.PNG' or \
                                    f_name[-4:] == '.PDF' or f_name[-4:] == 'JPEG':
                        match_path_train.append(os.path.join(path_out, cat_list))
                        match_file.append(p_list_file[i])
                        match_join.append(p_list_join[i])

        return match_path_train, match_file, match_join

    def display_test_result(self, result):
        """
            Calculate the accuracy and display the result data
        """
        print("")
        print("  Category               Right    Wrong  NoDetect   Accuracy")
        for key in result.keys():
            if len(key) < 20:
                display1 = '{:<20}'.format(key)
            else:
                display1 = '{:<20}'.format(key[:17] + '...')
            display2 = '{:>8}'.format(result[key][0])
            display3 = '{:>7}'.format(result[key][1])
            display4 = '{:>8}'.format(result[key][2])
            acc_str = '{:.2%}'.format(float(result[key][0])/(result[key][0]+result[key][1]+result[key][2]))
            # acc_str = '{:.2%}'.format(float(result[key][0])/(result[key][0]+result[key][1]))
            display5 = '{:>12}'.format(acc_str)
            print(display1, display2, display3, display4, display5)
