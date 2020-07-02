import argparse
import os
import cv2
import torch
import torch.nn as nn
from model.segnet import SegNet
from dataset.dataset import SegDataset
from torch.utils.data import Dataset, DataLoader
from utils import vis_pred_result

SNAPSHOT_DIR = './snapshots/'
TRAIN_DEBUG_VIS_DIR = './test_debug_vis/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Seg subtitle Network")
    parser.add_argument("--test-debug-vis-dir", type=str, default=TRAIN_DEBUG_VIS_DIR,
                        help="Directory for saving vis results during testing.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

args = get_arguments()


def main():

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if not os.path.exists(args.test_debug_vis_dir):
        os.makedirs(args.test_debug_vis_dir)

    model = SegNet(model='resnet50')
    model.load_state_dict(torch.load(args.snapshot_dir + '150000.pth'))

    # freeze bn statics
    model.eval()
    model.cuda()
    
    dataloader = DataLoader(SegDataset(mode='test'), batch_size=1, shuffle=False, num_workers=4)

    for i_iter, batch_data in enumerate(dataloader):
    
        Input_image, vis_image, gt_mask, weight_matrix, dataset_length, image_name = batch_data

        pred_mask = model(Input_image.cuda())

        print('i_iter/total {}/{}'.format(\
               i_iter, int(dataset_length[0].data)))

        if not os.path.exists(args.test_debug_vis_dir + image_name[0].split('/')[0]):
            os.makedirs(args.test_debug_vis_dir + image_name[0].split('/')[0])
        
        vis_pred_result(vis_image, gt_mask, pred_mask, args.test_debug_vis_dir + image_name[0] + '.png')
            

if __name__ == '__main__':
    main()