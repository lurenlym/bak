import numpy as np
import os
import cv2
project_root = '/home/lzhpc/CODE/ProfileNet'
data_root = project_root + '/data3/image_val/ZD'
save_root = project_root + '/data3/image_val/ZD_F'
for file in os.listdir(data_root):
    img_name = data_root+'/'+file
    temp = cv2.imread(img_name)
    img_f = cv2.flip(temp,1)
    # cv2.imshow('a',temp)
    # cv2.imshow('f',img_f)
    # cv2.waitKey()
    cv2.imwrite(save_root+'/'+file[0:-4]+'_f'+'.bmp',img_f)


