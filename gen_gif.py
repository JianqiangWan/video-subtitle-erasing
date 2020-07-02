import os
import cv2
import imageio
import numpy as np

test_imgs_dir = 'data/frame_with_new_char_test/30/'
origin_imgs = os.listdir(test_imgs_dir)

pred_masks_dir = 'test_debug_vis/30/'
pred_masks = os.listdir(pred_masks_dir)
pred_masks = [x.split('_')[0] for x in pred_masks if 'pred_mask' in x]

# for i in range(1, len(origin_imgs) + 1):

#     img = cv2.imread(test_imgs_dir + str(i) + '.jpg', 1)
#     img = cv2.resize(img, (480, 256), interpolation = cv2.INTER_LINEAR)
#     img_name = str(i)

#     if img_name in pred_masks:
#         pred_mask = cv2.imread(pred_masks_dir + img_name + '_pred_mask.png', 0)
#         pred_mask = (pred_mask > 128).astype(np.uint8)

        
#         contours = cv2.findContours(pred_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#         cnts = contours[0]

#         for cnt in cnts:
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
#     cv2.imwrite('gif_frames/' + img_name + '.png', img)

img_paths = []

for i in range(500, 700):
    img_paths.append('gif_frames/' + str(i) + '.png')

gif_images = []
for path in img_paths:
    gif_images.append(imageio.imread(path))

imageio.mimsave("test.gif",gif_images,fps=20)
