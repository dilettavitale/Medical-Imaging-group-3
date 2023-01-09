import torch
import torch.nn as nn
import config as config
#from dice import DiceLoss
from DiceLoss import DiceLoss
#from torchgeometry.losses.dice import DiceLoss
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss,self).__init__()
        self.crossEntropy = nn.CrossEntropyLoss()
        self.binaryCrossEntropy = nn.BCEWithLogitsLoss()
        self.binaryCrossEntropy2=nn.BCEWithLogitsLoss()
        self.dice=DiceLoss()
    
    def forward(self,preds,mask,label,intensity):
        intensity = intensity.unsqueeze(1)
        #intensity = intensity.float()
        '''print("Intensity unsqueezata:",intensity.shape)
        print("preds:",preds.shape)
        print("preds[0]:",preds[0].shape)
        print("preds[1]:",preds[1].shape)
        print("preds[2]:",preds[2].shape)
        print("mask[0]:",mask[0].shape)'''
        loss0 = self.dice._dice_loss(preds[0],mask)
        loss1 = self.crossEntropy(preds[1],label)
        loss2 = self.binaryCrossEntropy2(preds[2],intensity) 

        return torch.stack([loss0,loss1,loss2])
