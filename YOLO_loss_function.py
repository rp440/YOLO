#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(YoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        
        
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    
    def get_class_prediction_loss(self, classes_pred, classes_target):
        
        return F.mse_loss(classes_pred, classes_target, size_average=False)
    
    
    def get_regression_loss(self, box_pred_response, box_target_response):   
        
        pred_xy = box_pred_response[:,:2]
        pred_wh = torch.sqrt(box_pred_response[:,2:])
        target_xy = box_target_response[:,:2]
        target_wh = torch.sqrt(box_target_response[:,2:])
        loss_xy = F.mse_loss(pred_xy, target_xy, size_average=False)
        loss_wh = F.mse_loss(pred_wh, target_wh, size_average=False)       
        return (loss_xy+loss_wh)
    
    def get_contain_conf_loss(self, box_pred_response, box_target_response_iou):
        
        return F.mse_loss(box_pred_response[:,4], box_target_response_iou[:,4], size_average=False)
    
    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        
        
        no_object_prediction = pred_tensor[no_object_mask].view(-1, 30)
        no_object_target = target_tensor[no_object_mask].view(-1, 30)
        
        pred = torch.cat((no_object_prediction[:,4], no_object_prediction[:,9]), 0)
        target = torch.cat((no_object_target[:,4], no_object_target[:,9]), 0)

        return F.mse_loss(pred, target, size_average=False)
        
    
    
    def find_best_iou_boxes(self, box_target, box_pred):
        
        N = box_target.size()[0]
        M = box_pred.size()[0]
        box1 = torch.zeros((N, 4))
        box2 = torch.zeros((M, 4))

        box1[:,0] = box_target[:,0] / self.S - 0.5 * box_target[:,2]
        box1[:,1] = box_target[:,1] / self.S - 0.5 * box_target[:,3]
        box1[:,2] = box_target[:,0] / self.S + 0.5 * box_target[:,2]
        box1[:,3] = box_target[:,1] / self.S + 0.5 * box_target[:,3]

        box2[:,0] = box_pred[:,0] / self.S - 0.5 * box_pred[:,2]
        box2[:,1] = box_pred[:,1] / self.S - 0.5 * box_pred[:,3]
        box2[:,2] = box_pred[:,0] / self.S + 0.5 * box_pred[:,2]
        box2[:,3] = box_pred[:,1] / self.S + 0.5 * box_pred[:,3]

        iou = self.compute_iou(box1, box2)
        iou = torch.diag(iou, 0)
        contains_object_response_mask = torch.cuda.ByteTensor(box_target.size())
        contains_object_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0, iou.size()[0], 2):
            if iou[i] <= iou[i + 1]:
                contains_object_response_mask[i + 1] = 1
                box_target_iou[i + 1][4] = iou[i + 1]
            else:
                contains_object_response_mask[i] = 1
                box_target_iou[i][4] = iou[i]

        return box_target_iou, contains_object_response_mask
        
    
    
    
    def forward(self, pred_tensor,target_tensor):
        
        N = pred_tensor.size()[0]
        total_loss = None
        
        
        contains_object_mask = target_tensor[:,:,:,4] == 1
        no_object_mask = target_tensor[:,:,:,4] == 0

        contains_object_mask = contains_object_mask.unsqueeze(3).expand(N, self.S, self.S, 30)
        no_object_mask = no_object_mask.unsqueeze(3).expand(N, self.S, self.S, 30)

                        
        contains_object_pred = pred_tensor[contains_object_mask].view(-1, 30)
        bounding_box_pred = contains_object_pred[:,:10].contiguous().view(-1, 5)
        classes_pred = contains_object_pred[:,10:]        

        # Similarly as above create 2 tensors bounding_box_target and
        # classes_target.
        
        contains_object_target = target_tensor[contains_object_mask].view(-1, 30)
        bounding_box_target = contains_object_target[:,:10].contiguous().view(-1, 5)
        classes_target = contains_object_target[:,10:]

        # Compute the No object loss here
        
        no_object_loss = self.get_no_object_loss(target_tensor, pred_tensor, no_object_mask)

        # Compute the iou's of all bounding boxes and the mask for which bounding box 
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.
        
        box_target_iou, contains_object_response_mask = self.find_best_iou_boxes(bounding_box_target, bounding_box_pred)

        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou
        
        box_prediction_response = bounding_box_pred[contains_object_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[contains_object_response_mask].view(-1,5)
        box_target_response = bounding_box_target[contains_object_response_mask].view(-1,5)

        # Find the class_loss, containing object loss and regression loss
        
        class_loss = self.get_class_prediction_loss(classes_pred, classes_target)
        regression_loss = self.get_regression_loss(box_prediction_response, box_target_response)
        contain_loss = self.get_contain_conf_loss(box_prediction_response, box_target_response_iou)

        total_loss = (self.l_coord * regression_loss + contain_loss + self.l_noobj * no_object_loss + class_loss) / N        

        return total_loss

