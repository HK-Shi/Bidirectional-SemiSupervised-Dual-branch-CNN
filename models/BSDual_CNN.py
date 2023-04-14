from __future__ import print_function
from ast import Return
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from .stereo_focal_loss import StereoFocalLoss
from .smooth_l1_loss import DispSmoothL1Loss
from .conf_loss import ConfidenceLoss


class Network(nn.Module):
    def __init__(self, d):
        super(Network, self).__init__()
        self.branch1 = GwcNet(d, use_concat_volume=True)
        self.branch2 = GwcNet(d, use_concat_volume=True)

    def forward(self, left, right, leftDisp=None, step=1):

        if step == 1:
            return self.branch1(left, right, leftDisp)
        elif step == 2:
            return self.branch2(left, right, leftDisp)




class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X D X H X W) 主要输入
            --  y : input feature maps( B X D X C X H X W) 
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, D, height, width = x.size()
        y = x.contiguous().permute(0, 2, 1, 3, 4)  #B X D X C X H X W
        proj_query = y.contiguous().view(m_batchsize, D, C, -1)   # B × D X C X H*W
        proj_key = y.contiguous().view(m_batchsize, D, C, -1).permute(0, 1, 3, 2)  # B × D × H*W × C

        proj_query = proj_query.view(m_batchsize * D, C, height * width)
        proj_key = proj_key.view(m_batchsize * D, height * width, C)

        energy = torch.bmm(proj_query, proj_key)  #  B × D × C × C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.contiguous().view(m_batchsize, D, C, -1)  # B × D X C X H*W
        proj_value = proj_value.view(m_batchsize * D, C, height * width)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, D, C, height, width)

        out = self.gamma*out + y
        out = out.view(m_batchsize, C, D, height, width)
        return out



class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1), #32，64
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1), #64，64
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1), #64，128
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1), #128，128
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2)) #128，64

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels)) #64，32

        self.CAM = CAM_Module()

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x, use_CAM = False):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        if use_CAM:
            conv4 = self.CAM(conv4)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Conv3d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_normal_(m.weight,  mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()



        self.conf_est_net = nn.ModuleList([
            ConfidenceEstimation(in_planes=self.maxdisp) for i in range(3)])

        self.confidence_coefficient = 1.0
        self.confidence_init_value = 1.0   

        # calculate loss
        self.weights = (1.0, 0.7, 0.5, 0.5)
        self.sparse = True

        # distribution loss
        self.distribution_loss_evaluator = \
            StereoFocalLoss(max_disp=self.maxdisp, weights=self.weights, sparse=self.sparse)

        # disparity loss
        self.disparity_l1_loss_evaluator = DispSmoothL1Loss(self.maxdisp, weights=self.weights,
                                                  sparse=self.sparse)
        self.disparity_l1_loss_weight = 1 

        # confidence network loss
        self.conf_loss_evaluator = ConfidenceLoss(max_disp=self.maxdisp, weights=self.weights, sparse=self.sparse)
        self.conf_loss_weight = 8



    def forward(self, left, right, leftDisp=None):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        target = leftDisp

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)


        cost0 = self.classif0(cost0)
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2)
        cost3 = self.classif3(out3)

        cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost0 = torch.squeeze(cost0, 1)
        pred0 = F.softmax(cost0, dim=1)
        pred0 = disparity_regression(pred0, self.maxdisp)

        cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparity_regression(pred1, self.maxdisp)

        cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2, 1)
        pred2 = F.softmax(cost2, dim=1)
        pred2 = disparity_regression(pred2, self.maxdisp)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparity_regression(pred3, self.maxdisp)

        costs = [cost3, cost2, cost1, cost0]
        preds = [pred3, pred2, pred1, pred0]
        
        
        disps = [pred3.unsqueeze(dim=1), pred2.unsqueeze(dim=1), pred1.unsqueeze(dim=1), pred0.unsqueeze(dim=1)]
        confidence_costs = [cen(c) for c, cen in zip(costs, self.conf_est_net)]
        
        confidences = [torch.sigmoid(c) for c in confidence_costs]
        
        variances = [self.confidence_coefficient * (1 - conf) + self.confidence_init_value for conf in confidences]
        
        if self.training:
            losses = {}

            distribition_losses = self.distribution_loss_evaluator(costs, target, variances) 
            losses.update(distribition_losses)

            disparity_losses = self.disparity_l1_loss_evaluator(disps, target)
            disparity_losses = {k: v * self.disparity_l1_loss_weight for k, v in zip(disparity_losses.keys(), disparity_losses.values())}
            losses.update(disparity_losses)
            
            conf_losses = self.conf_loss_evaluator(confidence_costs, disps, target)
            conf_losses = {k : v * self.conf_loss_weight for k, v in zip(conf_losses.keys(), conf_losses.values())}
            losses.update(conf_losses)

            return losses, [pred3], confidences

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3], confidences


def Dual_CNN(d):
    return Network(d)
