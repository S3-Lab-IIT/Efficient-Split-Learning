import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    source: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=1.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([ 0.69435484,  0.25656675,  0.87447337,  3.3516436 ,  1.19780503, 12.1584728 , 11.48567194,  4.62718949])
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# -----------------------------------------------------------
# class WeightedFocalLoss(_Loss):
#     """Weighted focal loss
#     See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
#     a good explanation
#     Attributes
#     ----------
#     alpha: torch.tensor of size 8, class weights
#     gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
#     the same as CE loss, increases gamma reduces the loss for the "hard to classify
#     examples"
#     """

#     def __init__(
#         self,
#         alpha=torch.tensor(
#             [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
#         ),
#         gamma=0.2, # 0.1, 0.2 use small value
#     ):
#         super(WeightedFocalLoss, self).__init__()
#         # self.alpha = alpha.to(torch.float)
#         self.alpha = torch.tensor(
#             [ 0.69435484,  0.25656675,  0.87447337,  3.3516436 ,  1.19780503, 12.1584728 , 11.48567194,  4.62718949]
#         )
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         """Weighted focal loss function
#         Parameters
#         ----------
#         inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
#         targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
#         """
#         targets = targets.view(-1, 1).type_as(inputs)
#         logpt = F.log_softmax(inputs, dim=1)
#         logpt = logpt.gather(1, targets.long())
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#         self.alpha = self.alpha.to(targets.device)
#         at = self.alpha.gather(0, targets.data.view(-1).long())
#         logpt = logpt * at
#         loss = -1 * (1 - pt) ** self.gamma * logpt

#         return loss.mean()
