# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:21:58 2020

@author: jianqw
"""


import os
import cv2

imgs = os.listdir('frames/')

for i in range(len(imgs)):
    
    img = cv2.imread('frames/' + imgs[i], 1)
    
    with open('rtTxt_video_frame/' + imgs[i][:-3] + 'txt','r') as f:
        coords = f.read().splitlines()
        
    if coords != []:
        
        for coord in coords:
            
            coord = coord.split(',')
            
            left = int(coord[0])
            top = int(coord[1])
            
            right = int(coord[4])
            bottom = int(coord[5])
            
            cv2.rectangle(img, (left, top), (right, bottom), (0,0,255), 2)
    
    cv2.imwrite('frame_with_box/' + imgs[i], img)
    