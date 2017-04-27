"""OCR in Python using the Tesseract engine from Google
http://code.google.com/p/pytesser/
by Michael J.T. O'Kelly
V 0.0.1, 3/10/07"""

from PIL import Image
import cv2
import subprocess
from OCRUtil import OCRUtil
from OCRErrors import OCRErrors
from OCRErrors import Tesser_General_Exception
import os
import json
import numpy as np
from datetime import datetime
from FuncMl import FuncMl


class OCRClassify:
    def __init__(self, temp_dir):
        """
            class initial function
        """
        # Name of executable to be called at command line
        self.tesseract_exe_name = 'tesseract'

        # Name of scratch image and text file
        self.scratch_image_name = "temp" + str(datetime.now().microsecond) + ".bmp"
        self.scratch_text_name_root = "temp" + str(datetime.now().microsecond)

        # define class and variables
        self.util = OCRUtil()
        self.errors = OCRErrors()
        self.func_ml = FuncMl()
        self.my_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.scratch_dir = temp_dir

        # load the keyword json file
        data_file = open(os.path.join(self.my_dir, "config/keywords.json"))
        self.key_list = json.load(data_file)

    def call_tesseract(self, input_filename, output_filename):
        """
            Calls external tesseract.exe on input file (restrictions on types), outputting output_filename+'txt'
        """
        args = [self.tesseract_exe_name, input_filename, output_filename]
        process_call = subprocess.Popen(args)
        ret_code = process_call.wait()
        if ret_code != 0:
            self.errors.check_for_errors()

    def image_file_to_string(self, filename):
        """
            Applies tesseract to filename; or, if image is incompatible and graceful_errors=True,
            converts to compatible format and then applies tesseract.  Fetches resulting text.
            If cleanup=True, delete scratch files after operation.
        """
        try:
            self.call_tesseract(filename, self.scratch_text_name_root)
            text_ret = self.util.retrieve_text(self.scratch_text_name_root)
        except Tesser_General_Exception:
            text_ret = ''
            pass

        self.util.perform_cleanup(self.scratch_image_name, self.scratch_text_name_root)

        return text_ret

    def image_process(self, in_name, out_name):
        """
            Convert and pre-process the image for OCR. first convert the image files to grayscale image and save it.
        """
        ori_img = cv2.imread(in_name)
        if ori_img is not None:
            blank_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(out_name, blank_img)
            return True
        else:
            return False

    def count_key(self, ori_text, keyword):
        """
            calculate the count of keyword in ori_text .
            This information is used in analyse of OCR result and classify the documents.
        """
        key_upper = keyword.upper()
        text_upper = ori_text.upper().decode("utf-8")
        return text_upper.count(key_upper)

    def get_text(self, image):
        temp_file = os.path.join(self.scratch_dir, self.scratch_image_name)

        if self.image_process(image, temp_file):
            ocr_text = self.image_file_to_string(temp_file)
            os.remove(temp_file)
            return ocr_text
        else:
            return ''
        
    def __classify(self, input_name):
        """
            classify the file using OCR and return accuracy data and state information.
        """
        ret_ocr = False
        """ -------------- get the text using OCR -------------------- """
        text = ''
        if os.path.isfile(input_name):  # ------------ Case of file ------------------
            text = self.get_text(input_name)

        elif os.path.exists(input_name):  # ------ Case of input is directory -----------
            for f_name in os.listdir(input_name):
                text += self.get_text(input_name + '/' + f_name)

        """ -------------- classify the categories ------------------- """
        if text == '':
            info_data = "Failed to recognize this image!"
        else:
            key_categories = self.key_list.keys()                                # extract the key categories
            key_point_list = []
            all_zero = 1

            for key_category in key_categories:                             # extract the key category from categories
                value = 0
                for key in self.key_list[key_category]['keywords']:              # extract the key from key category
                    ret = self.count_key(text, key)                         # extract the contained count in text
                    weight_key = self.key_list[key_category]['keywords'][key]    # extract the weight of key
                    value += ret * weight_key                               # calculate the key point

                if value > 0:
                    all_zero = 0

                key_point_list.append(value)

            if all_zero == 0:
                acc_list = self.func_ml.standardization(key_point_list)     # standardization of key point
                cats, _, cat_ids = self.func_ml.load_standard_base_documents_json()
                ocr_cat_ids = []

                for j in xrange(key_categories.__len__()):
                    for i in xrange(cat_ids.__len__()):
                        if cats[i] == key_categories[j]:
                            ocr_cat_ids.append(cat_ids[i])
                            break

                if ocr_cat_ids.__len__() != key_categories.__len__():
                    info_data = "Categories in keywords.json do not match categories from standard_base_documents.json"
                else:
                    acc_list_all = np.zeros(1000)
                    for i in range(len(acc_list)):
                        acc_list_all[ocr_cat_ids[i] - 1] = acc_list[i]

                    info_data = self.func_ml.get_classify_result(acc_list_all, key_categories, ocr_cat_ids, 1)
                    ret_ocr = True

            else:
                info_data = "No key detected."
        return ret_ocr, info_data, text

    def classify_print(self, input_filename, min_cnt=None):
        """
            classify the file using OCR and display the result.
        """
        ret_ocr, comb_acc, _ = self.__classify(input_filename)
        if ret_ocr:
            n = comb_acc.__len__()
            if min_cnt is not None:
                n = min(min_cnt, comb_acc.__len__())
            for j in range(n):
                if comb_acc[j][0] > 1:
                    print "   ", '{:20}'.format(comb_acc[j][1]), ':', "%2.6f" % comb_acc[j][0]
        else:
            print comb_acc

    def classify(self, input_filename, min_cnt=None):
        """
            classify the file using OCR and return the result.
        """
        ret_ocr, comb_acc, _ = self.__classify(input_filename)
        classify_ret = []
        if ret_ocr:
            n = comb_acc.__len__()
            if min_cnt is not None:
                n = min(min_cnt, comb_acc.__len())
            for j in range(n):
                if comb_acc[j][0] > 1:
                    dic_cat = {"id": comb_acc[j][2], "name": str(comb_acc[j][1]), "score": comb_acc[j][0]}
                    classify_ret.append(dic_cat)

        return classify_ret
