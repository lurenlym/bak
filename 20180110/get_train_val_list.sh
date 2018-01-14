#!/bin/bash

find `pwd`/image_train -name '*.bmp' | sort -V > image_train/file_list.txt
find `pwd`/image_val -name '*.bmp' | sort -V > image_val/file_list.txt

python add_label.py image_train/file_list.txt train_list.txt
python add_label.py image_val/file_list.txt val_list.txt
