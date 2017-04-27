# README #

This document is updated 04/27/2017

The goal of this project is classify the images as 20 categories

### Necessary Packages ###

* Python 2.7
* Tensorflow 0.10.0rc0
* OpenCV 3
* Numpy
* libmagickwand-dev
* wand
* tesseract-ocr
* libtesseract-dev
* libleptonica-dev
* PIL(pillow)
* sklearn
* pickle


### Configuration ###
This project have 6 python files and 1 xls files.

* gen_train.py

    Convert images into training data (which size is 224*224 pixels) for machine learning.

    Format:

            python gen_train.py [path_in [path_out]]

    i.e.:

            python gen_train.py
            python gen_train.py image_real
            python gen_train.py e:/test/source e:/test/training

    default value:

            path_in     : image_real
            path_out    : training

* training_CNN.py

    Construct the model and train weights.

    Format:

            python training_CNN.py [train_path model_name step learning_rate para_conv para_hidden]

    i.e.:

            python training_CNN.py
            python training_CNN.py image_real
            python training_CNN.py image_real models/classify_CNN
            python training_CNN.py image_real models/classify_CNN 500 0.001
            python training_CNN.py image_real models/classify_CNN 500 0.001 0.8 0.5

    default value:

            train_path      : training              (training data path)
            model_name      : models/classify_CNN   (saved CNN model file name)
            step            : 1000                  (training step number)
            learning_rate   : 0.001                 (learning rate)
            para_conv       : 0.8                   (convolution coefficient)
            para_hidden     : 0.5                   (dropdown coefficient)

* image_classify.py

    Classify the individual image file or image group as folder using CNN.

    Format:

            python image_classify.py image_file_name
            python image_classify.py image_folder_name

    i.e.:

            python image_classify.py 1.jpg
            python image_classify.py ./source/Passports/1.jpg
            python image_classify.py ./source/Global\ Entry/1.jpg
            python image_classify.py ./source/SSC

