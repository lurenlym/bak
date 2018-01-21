__author__ = 'lyming'
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
alex = open('./0119/0119alexnet_no_flip.log.test','r')
alex_data=[]
for line in alex:
    if line != '\n':
        alex_data.append(line[0:-2].split(','))
title = alex_data[0]
temp1 = np.array(alex_data[1:]).astype('float64')
alex = open('./0119/0119sketch_a.log.test','r')
alex_data=[]
for line in alex:
    if line != '\n':
        alex_data.append(line[0:-2].split(','))
temp2 = np.array(alex_data[1:]).astype('float64')
alex = open('./0119/20180119alexnet.log.test','r')
alex_data=[]
for line in alex:
    if line != '\n':
        alex_data.append(line[0:-2].split(','))
temp3 = np.array(alex_data[1:]).astype('float64')
plt.subplot(1,2,1)
plt.title('accucary')
plt.plot(temp1[:,0],temp1[:,3],'r', label='alexnet')
plt.plot(temp2[:,0],temp2[:,3],'b', label='sketch_a_net')
plt.plot(temp3[:,0],temp3[:,3],'g', label='alexnet_flip_data')
plt.legend(bbox_to_anchor=[0.15, 0.5])
plt.subplot(1,2,2)
plt.title('loss')
plt.plot(temp1[:,0],temp1[:,4],'r', label='alexnet')
plt.plot(temp2[:,0],temp2[:,4],'b', label='sketch_a_net')
plt.plot(temp3[:,0],temp3[:,4],'g', label='alexnet_flip_data')
plt.legend(bbox_to_anchor=[0.15, 0.5])
plt.savefig('result.png',dpi=200)
plt.show()