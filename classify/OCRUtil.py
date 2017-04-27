"""Utility functions for processing images for delivery to Tesseract"""

import os

class OCRUtil:
    def __init__(self):
        pass

    def image_to_scratch(self, im, scratch_image_name):
        """Saves image in memory to scratch file.  .bmp format will be read correctly by Tesseract"""
        im.save(scratch_image_name, dpi=(200, 200))


    def retrieve_text(self, scratch_text_name_root):
        inf = file(scratch_text_name_root + '.txt')
        text = inf.read()
        inf.close()
        return text


    def perform_cleanup(self, scratch_image_name, scratch_text_name_root):
        """Clean up temporary files from disk"""
        for name in (scratch_image_name, scratch_text_name_root + '.txt', "tesseract.log"):
            try:
                os.remove(name)
            except OSError:
                pass
