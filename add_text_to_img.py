# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:46:34 2020

@author: jianqw
"""
import os
import cv2
import random
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 用于文字边框展示，传入draw,坐标x,y，字体，边框颜色和填充颜色
def text_border(draw, text, x, y, font, shadowcolor, fillcolor):
    # thin border
    draw.text((x - 1, y), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y), text, font=font, fill=shadowcolor)
    draw.text((x, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x, y + 1), text, font=font, fill=shadowcolor)
 
    # thicker border
    draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)
 
    # now draw the text over it
    draw.text((x, y), text, font=font, fill=fillcolor)

fontpath = './data/Fonts/Alibaba-PuHuiTi-Bold.ttf'
font = ImageFont.truetype(fontpath, 20)

with open('./data/3500常用汉字.txt', 'r', encoding='utf-8') as f:
    characters = f.read().splitlines()

random.shuffle(characters)


train_seg_file = open('./data/train_seg.txt', 'w')

# 读取视频
for j in range(1, 30):


    #获得视频的格式
    videoCapture = cv2.VideoCapture('./data/uc_200/' + str(j) + '.mp4')
    
    #获得码率及尺寸
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    #读帧
    success, frame = videoCapture.read()

    step = 1

    # 帧存放目录
    frames_dir = './data/frame/' + str(j) + '/'
    frames_with_mask_dir = './data/frame_with_mask/' + str(j) + '/'
    frames_with_new_char_dir = './data/frame_with_new_char/' + str(j) + '/'
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if not os.path.exists(frames_with_mask_dir):
        os.makedirs(frames_with_mask_dir)

    if not os.path.exists(frames_with_new_char_dir):
        os.makedirs(frames_with_new_char_dir)

    while success :
        cv2.imwrite(frames_dir + str(step) + '.jpg', frame)
        success, frame = videoCapture.read() #获取下一帧
        step += 1
        
    videoCapture.release()

    imgIds = os.listdir(frames_dir)

    chars = '初始值'

    for i in range(len(imgIds)):

        if i % 100 == 0:
            print(j, i)
        
        imgId = str(i + 1) + '.jpg'

        img = cv2.imread(frames_dir + imgId, 1)
        height, width, _ = img.shape
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if np.random.uniform() > 0.8:
            #绘制文字信息, 一行按30个字计算， 随机产生1-16个字符
            character_length = random.randint(1, 17)
            start_point = random.randint(1, 3400)
        
            chars_list = characters[start_point:(start_point +  character_length)]
            
            chars = ''.join(chars_list)
        
        # w, h
        chars_w, chars_h = font.getsize(chars)
        chars_x, chars_y = int((width - chars_w)/2), int(2*height/3)
        
        coords = [(chars_x, chars_y + 2), (chars_x, chars_y + chars_h),
                (chars_x + chars_w, chars_y + chars_h), (chars_x + chars_w, chars_y + 2)]
        
        valid_height = chars_y + chars_h + 15
        start_height = int(height / 5)
        start_width = int(width / 6)

        if np.random.uniform() > 0.15:
            
            text_border(draw, chars, chars_x, chars_y, font, (0, 0, 0), (240, 240, 240))
            
            img = np.array(img_pil)
            cv2.imwrite(frames_with_new_char_dir + imgId, img[start_height:valid_height, start_width:-start_width, :])
                
            cv2.rectangle(mask, (chars_x, chars_y + 2), (chars_x + chars_w, chars_y + chars_h), (255,255,255), -1) 
            cv2.imwrite(frames_with_mask_dir + imgId[:-3] + 'png', mask[start_height:valid_height, start_width:-start_width])

            train_seg_file.write(str(j) + '/' + imgId[:-4] + '\n')
        
        else:
            
            img = np.array(img_pil)
            cv2.imwrite(frames_with_new_char_dir + imgId, img[start_height:valid_height, start_width:-start_width, :])
            
            cv2.imwrite(frames_with_mask_dir + imgId[:-3] + 'png', mask[start_height:valid_height, start_width:-start_width])
    
train_seg_file.close()
