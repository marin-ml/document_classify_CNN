
import cv2
from classify import FuncMl
import sys
import os
import shutil
import time
import numpy as np


def image_rotate_save(img_src, img_name):
    """
        rotate the img_src 4 times and save as bmp file
    """
    for rot_time in range(4):
        rot_img = np.rot90(img_src, rot_time)
        cv2.imwrite(img_name + str(rot_time) + '.bmp', rot_img)
        os.rename(img_name + str(rot_time) + '.bmp', img_name + str(rot_time))


def target_clean(target_folder, cat_folders):
    """
        cleaning and set of target folder
    """
    shutil.rmtree(target_folder, ignore_errors=True)
    time.sleep(1)
    os.makedirs(target_folder)
    dir_list = list(set(cat_folders))
    for dir_name in dir_list:
        os.makedirs(dir_name)


def convert_image(train_path, src_file, src_join):
    """
        load and convert image
    """
    for i in range(len(src_file)):
        if src_file[i][-4:].upper() == '.PDF':
            img_src = 'temp.jpg'
            func_class.pdf2jpg(src_join[i], img_src, True)
        else:
            img_src = src_join[i]

        img = func_class.load_convert_image(img_src)

        if img is not None:
            print "Converting", src_join[i]

            # Rotate original image
            image_rotate_save(img, train_path[i] + '/' + src_file[i][:-4] + str(i) + '_0_')

        else:
            print "Converting", src_join[i], "Invalid Image Format!"

if __name__ == '__main__':
    """ -------------------------- Input argument process ---------------------------- """
    if len(sys.argv) >= 2:
        para_path_in = sys.argv[1]
    else:
        para_path_in = "image_real"

    if len(sys.argv) >= 3:
        para_path_out = sys.argv[2]
    else:
        para_path_out = "training"

    """ --------- get file lists in para_path_in and load categorization list --------- """
    func_class = FuncMl()
    _, list_file, list_join = func_class.get_file_list(para_path_in)
    cat_folder, _, _ = func_class.load_standard_base_documents_json()

    """ --------------------------- get matching file lists ---------------------------- """
    img_path_train, img_file, img_join = \
        func_class.get_match_files(cat_folder, list_file, list_join, para_path_in, para_path_out)

    """ ---------------------- cleaning and set of target folder ----------------------- """
    target_clean(para_path_out, img_path_train)

    """ ----------------------------- load and convert image --------------------------- """
    convert_image(img_path_train, img_file, img_join)

    if os.path.isfile('temp.jpg'):
        os.remove('temp.jpg')
