import sys
import scipy.io as sio
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import pylab as plt
from matplotlib import cm
import os
import torch.nn.functional as F

def label2color(label):

    label = label.astype(np.uint16)
    
    height, width = label.shape
    color3u = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(label)

    if unique_labels[-1] >= 2**24:       
        raise RuntimeError('Error: label overflow!')

    for i in range(len(unique_labels)):
    
        binary = '{:024b}'.format(unique_labels[i])
        # r g b 3*8 24
        r = int(binary[::3][::-1], 2)
        g = int(binary[1::3][::-1], 2)
        b = int(binary[2::3][::-1], 2)

        color3u[label == unique_labels[i]] = np.array([r, g, b])

    return color3u


def vis_pred_result(vis_image, gt_mask, pred_mask, save_dir):

    vis_image = vis_image.data.cpu().numpy()[0, ...]

    pred_mask = 255 * F.sigmoid(pred_mask.data.cpu()).numpy()[0,0,...]
    gt_mask = 255 * F.sigmoid(gt_mask.data.cpu()).numpy()[0,0,...]

    fig = plt.figure(figsize=(10, 3))

    ax0 = fig.add_subplot(131)
    ax0.imshow(vis_image[:,:,::-1])

    ax1 = fig.add_subplot(132)
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(gt_mask, cmap=cm.gray)


    ax1 = fig.add_subplot(133)
    ax1.set_autoscale_on(True)
    im1 = ax1.imshow(pred_mask, cmap=cm.gray)
    plt.colorbar(im1,shrink=0.5)

    plt.savefig(save_dir)
    plt.close(fig)