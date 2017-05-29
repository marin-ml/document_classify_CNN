from classify import ImageClassify
import sys

""" ----------------------- Input argument process -------------------------- """
if __name__ == '__main__':

    if len(sys.argv) == 1:
        # src_name = './image_real/EAD_CARD/EAD Card Front.JPG'
        src_name = './image_real/3582/3582.pdf'
        # src_name = 'image_real/Birth_Certificate/7.Mexico/2.jpg'

    else:
        src_name = sys.argv[1]

    classify_class = ImageClassify('models/classify_CNN')

    # classify_class.classify_print(src_name)
    print(classify_class.classify(src_name))
