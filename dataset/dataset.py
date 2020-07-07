import cv2
import os
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms as T

class SegDataset(Dataset):

    def __init__(self, mode='train'):
        
        self.mode = mode
        file_dir = 'data/' + self.mode + '_seg.txt'

        with open(file_dir, 'r') as f:
            self.image_names = f.read().splitlines()

        self.dataset_length = len(self.image_names)

        self.normalize = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
    
    def __len__(self):

        return self.dataset_length

    def __getitem__(self, index):

        image_path = 'data/frame_with_new_char_' + self.mode + '/' + self.image_names[index] + '.jpg'
        label_path = 'data/frame_with_mask_' + self.mode + '/' + self.image_names[index] + '.png'   

        image = cv2.imread(image_path, 1)
        height, width, _ = image.shape

        label = cv2.imread(label_path, 0)

        image = cv2.resize(image, (480, 256), interpolation = cv2.INTER_LINEAR)
        vis_image = image.copy()
        label = cv2.resize(label, (480, 256), interpolation = cv2.INTER_NEAREST)
        # padding_h = height % 32
        # padding_w = width % 32

        # if padding_h > 0:
        #     padding_h = 32 - padding_h
        # if padding_w > 0:
        #     padding_w = 32 - padding_w

        # padding_value = (104, 116, 124)
        # image = cv2.copyMakeBorder(image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=padding_value)
        # label = cv2.copyMakeBorder(label, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)

        label = (label >0).astype(np.float32)

        image = image[:,:, ::-1]
        image = image.astype(np.float32)
        # normalize input image
        image = self.normalize(image)

        pos_sum = (label > 0).sum()
        neg_sum = (label == 0).sum()
        total_sum = pos_sum + neg_sum

        weight_matrix = (pos_sum / total_sum) * (1 - label) + (neg_sum / total_sum) * label
        weight_matrix = weight_matrix[np.newaxis, ...]

        label = label[np.newaxis, ...]

        return image, vis_image, label, weight_matrix, self.dataset_length, self.image_names[index]
