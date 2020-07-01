import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50

class SegNet(nn.Module):

    def __init__(self, model='resnet50', pretrained=True):
        super(SegNet, self).__init__()

        assert model in ['resnet50', 'resnet50']

        if model == 'resnet50':
            resnet = resnet50(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.aspp_conv1 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.aspp_conv2 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=4, dilation=4), nn.ReLU(inplace=True))
        self.aspp_conv3 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=8, dilation=8), nn.ReLU(inplace=True))
        self.aspp_conv4 = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=16, dilation=16), nn.ReLU(inplace=True))

        self.seg_conv = nn.Sequential(nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1))
        
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, stride=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1), nn.ReLU(inplace=True))

        nn.init.normal_(self.conv1[0].weight, mean=0, std=0.0001)
        nn.init.normal_(self.conv2[0].weight, mean=0, std=0.001)
        nn.init.normal_(self.conv3[0].weight, mean=0, std=0.01)
        nn.init.normal_(self.conv4[0].weight, mean=0, std=0.1)
        nn.init.normal_(self.conv5[0].weight, mean=0, std=0.1)
        
    def forward(self, x):

        x = self.layer0(x)

        x = self.layer1(x)
        stage1 = x
        
        x = self.layer2(x)
        stage2 = x

        x = self.layer3(x)
        stage3 = x

        x = self.layer4(x)
        stage4 = x

        aspp_feature1 = self.aspp_conv1(x)
        aspp_feature2 = self.aspp_conv2(x)
        aspp_feature3 = self.aspp_conv3(x)
        aspp_feature4 = self.aspp_conv4(x)

        aspp_feature = torch.cat((aspp_feature1, aspp_feature2, aspp_feature3, aspp_feature4), 1)

        stage5 = F.interpolate(self.conv5(aspp_feature), scale_factor=8, mode='bilinear')
        stage4 = F.interpolate(self.conv4(stage4), scale_factor=8, mode='bilinear')
        stage3 = F.interpolate(self.conv3(stage3), scale_factor=4, mode='bilinear')
        stage2 = F.interpolate(self.conv2(stage2), scale_factor=2, mode='bilinear')
        stage1 = self.conv1(stage1)

        concat_feature = torch.cat((stage1, stage2, stage3, stage4, stage5), 1)

        pred_seg = self.seg_conv(concat_feature)

        pred_seg = F.interpolate(pred_seg, scale_factor=4, mode='bilinear')
 
        return pred_seg
