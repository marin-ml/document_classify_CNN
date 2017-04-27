
import numpy as np
from FuncMl import FuncMl
import os
from datetime import datetime
try:
    import tensorflow as tf
except ImportError:
    pass


class ImageClassify:

    def __init__(self, model_name):
        """
            Variable Initialization and CNN model Configuration
        """
        self.cat_cnt = 1000
        self.image_size = 224
        self.func_ml = FuncMl()
        self.cat_folder, self.cat_data, self.cat_id = self.func_ml.load_standard_base_documents_json()

        py_x, self.p_keep_hidden, self.p_keep_con, self.X, _ = self.func_ml.model_config()
        self.soft_op = tf.nn.softmax(py_x)
        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, model_name)

    def __pre_image(self, image_name, img_size):
        """
            Load the image data and classify using machine learning model
        """
        if image_name[-4:].upper() == ".PDF":  # in case of pdf files
            temp_image = "temp_pdf" + str(datetime.now().microsecond) + ".bmp"
            self.func_ml.pdf2jpg(image_name, temp_image, True)
            blank_img = self.func_ml.load_convert_image(temp_image)
            os.remove(temp_image)
        else:
            blank_img = self.func_ml.load_convert_image(image_name)

        if blank_img is not None:
            """ ------- Convert image to standard style and size for model input ----------- """
            data_pre = blank_img.reshape([1, img_size, img_size, 1])

            """ ---------- Get the accuracy data from input image data to model ------------ """
            soft_data_pre = self.sess.run(self.soft_op,
                                          feed_dict={self.X: data_pre, self.p_keep_con: 1, self.p_keep_hidden: 1})
            acc_data_pre = self.func_ml.standardization(soft_data_pre[0])
            return acc_data_pre
        else:
            return None

    def __classify(self, input_name):
        """
            Classify the input file or folder
        """
        if os.path.isfile(input_name):      # ------------ Case of file ------------------
            acc_data = self.__pre_image(input_name, self.image_size)
            if acc_data is None:
                return None
            else:
                comb_acc = self.func_ml.get_classify_result(acc_data, self.cat_folder, self.cat_id, 1)
                return comb_acc

        elif os.path.exists(input_name):    # ------ Case of input is directory -----------
            acc = np.zeros(self.cat_cnt + 1)

            for f_name in os.listdir(input_name):
                acc_data = self.__pre_image(input_name + '/' + f_name, self.image_size)
                if acc_data:
                    for i in range(len(acc_data)):
                        acc[i] += acc_data[i]
                    acc[self.cat_cnt] += 1

            comb_acc = self.func_ml.get_classify_result(acc, self.cat_folder, self.cat_id, acc[self.cat_cnt])

            return comb_acc

        else:
            return None
        
    def classify_print(self, input_name, min_cnt=None):
        comb_acc = self.__classify(input_name)
        if comb_acc:
            n = comb_acc.__len__()
            if min_cnt is not None:
                n = min(min_cnt, comb_acc.__len__())
                
            for j in range(n):
                if comb_acc[j][0] > 1:
                    print "   ", '{:20}'.format(comb_acc[j][1]), ':', "%2.6f" % comb_acc[j][0]
        else:
            print "Incorrect file name or image format!"

    def classify(self, input_name, min_cnt=None):
        comb_acc = self.__classify(input_name)
        classify_ret = []
        if comb_acc:
            n = comb_acc.__len__()
            if min_cnt is not None:
                n = min(min_cnt, comb_acc.__len__())
            
            for j in range(n):
                if comb_acc[j][0] > 1:
                    dic_cat = {"id": comb_acc[j][2], "name": str(comb_acc[j][1]), "score": "%2.6f" % comb_acc[j][0]}
                    classify_ret.append(dic_cat)

            return classify_ret

        else:
            return None
