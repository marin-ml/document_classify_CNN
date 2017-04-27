
from classify import ImageClassify
from classify import FuncMl
import sys


if __name__ == '__main__':

    if len(sys.argv) == 1:
        path_name = 'image_real'
    else:
        path_name = sys.argv[1]

    func_class = FuncMl()
    cnn_class = ImageClassify('models/classify_CNN')

    """ --------- get file lists in para_path_in and load categorization list --------- """
    _, list_file, list_join = func_class.get_file_list(path_name)
    cat_folder, _, _ = func_class.load_standard_base_documents_json()

    """ --------------------------- get matching file lists ---------------------------- """
    img_cat, _, img_join = func_class.get_match_files(cat_folder, list_file, list_join, path_name, '')

    """ ------------- classify matched files using OCR and get result data ------------- """
    ret_table = {}
    test_threshold = 70
    for i in range(len(img_join)):
        if ret_table.keys().count(img_cat[i]) == 0:         # add the new category item to result table
            ret_table.update({img_cat[i]: [0, 0, 0]})

        file_image = img_join[i]
        ret = cnn_class.classify(file_image)

        if ret:
            if ret[0]['name'] == img_cat[i] and ret[0]['score'] > test_threshold:
                ret_table[img_cat[i]][0] += 1
                print file_image, ret[0], "Right!"
            else:
                ret_table[img_cat[i]][1] += 1
                print file_image, ret[0], "Wrong!"
        else:
            ret_table[img_cat[i]][2] += 1
            print file_image, "Incorrect image format!"

    # print ret_table
    func_class.display_test_result(ret_table)
