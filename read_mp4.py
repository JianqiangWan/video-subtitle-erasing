# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import cv2

#获得视频的格式
videoCapture = cv2.VideoCapture('./uc_200/15.mp4')
  
#获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

#读帧
success, frame = videoCapture.read()

step = 1

while success :
    cv2.imwrite('frames/' + str(step) + '.jpg', frame)
    success, frame = videoCapture.read() #获取下一帧
    step += 1
    
videoCapture.release()
