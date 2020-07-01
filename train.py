import argparse
import os
import torch
import torch.nn as nn
from model.segnet import SegNet
from dataset.dataset import SegDataset
from torch.utils.data import Dataset, DataLoader
from utils import vis_pred_result

INI_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHES = 10000

SNAPSHOT_DIR = './snapshots/'
TRAIN_DEBUG_VIS_DIR = './train_debug_vis/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Seg subtitle Network")
    parser.add_argument("--train-debug-vis-dir", type=str, default=TRAIN_DEBUG_VIS_DIR,
                        help="Directory for saving vis results during training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred_mask, gt_mask, weight_matrix):

    device_id = pred_mask.device
    gt_mask = gt_mask.cuda(device_id)
    weight_matrix = weight_matrix.cuda(device_id)

    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda(device_id)

    loss = weight_matrix * criterion(pred_mask, gt_mask)
    loss = loss.sum() / weight_matrix.sum()

    return loss

def get_params(model, key, bias=False):

    # for backbone 
    if key == "backbone":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        if m[1].bias != None:
                            yield m[1].bias
                # freeze bn
                elif isinstance(m[1], nn.BatchNorm2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        if m[1].bias != None:
                            yield m[1].bias
                    # m[1].weight.requires_grad = False
                    # m[1].bias.requires_grad = False
            
    # for added layer
    if key == "added":
        for m in model.named_modules():
            if "layer" not in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        yield m[1].bias
                elif isinstance(m[1], nn.BatchNorm2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        if m[1].bias != None:
                            yield m[1].bias

def adjust_learning_rate(optimizer, step):
    
    if step == 8e4:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def main():

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if not os.path.exists(args.train_debug_vis_dir):
        os.makedirs(args.train_debug_vis_dir)

    model = SegNet(model='resnet50')

    # freeze bn statics
    model.train()
    model.cuda()
    
    optimizer = torch.optim.SGD(
        params=[
            {
                "params": get_params(model, key="backbone", bias=False),
                "lr": INI_LEARNING_RATE
            },
            {
                "params": get_params(model, key="backbone", bias=True),
                "lr": 2 * INI_LEARNING_RATE
            },
            {
                "params": get_params(model, key="added", bias=False),
                "lr": 10 * INI_LEARNING_RATE  
            },
            {
                "params": get_params(model, key="added", bias=True),
                "lr": 20 * INI_LEARNING_RATE   
            },
        ],
        lr = INI_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    dataloader = DataLoader(SegDataset(), batch_size=8, shuffle=True, num_workers=4)

    global_step = 0

    for epoch in range(1, EPOCHES):

        for i_iter, batch_data in enumerate(dataloader):

            global_step += 1

            Input_image, vis_image, gt_mask, weight_matrix, dataset_length = batch_data

            optimizer.zero_grad()

            pred_mask = model(Input_image.cuda())

            loss = loss_calc(pred_mask, gt_mask, weight_matrix)

            loss.backward()

            optimizer.step()

            if global_step % 4 == 0:
                print('epoche {} i_iter/total {}/{} loss {:.2f}'.format(\
                       epoch, i_iter, int(dataset_length[0].data), loss))
            
            if global_step % 5 == 0:
                vis_pred_result(vis_image, gt_mask, pred_mask, args.train_debug_vis_dir + str(global_step) + '.png')
                
            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), args.snapshot_dir + str(global_step) + '.pth')

if __name__ == '__main__':
    main()