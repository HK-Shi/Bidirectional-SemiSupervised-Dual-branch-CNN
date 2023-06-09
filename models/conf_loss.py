import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceLoss(object):
    """

        Args:
            weights (list of float or None): weight for each scale of estCost.
            max_disp (int): the max of Disparity. default: 192
            sparse (bool): whether the ground-truth disparity is sparse, for example, KITTI is sparse, but SceneFlow is not. default: False

        Inputs:
            estConf (Tensor or list of tensor): the estimated confidence mam, in (BatchSize, 1, Height, Width) layout.
            gtDisp (Tensor): the ground truth disparity map, in (BatchSize, 1, Height, Width) layout.

        Outputs:
            loss (dict), the loss of each level
    """

    def __init__(self, max_disp, weights=None, sparse=False):
        self.max_disp = max_disp
        self.weights = weights
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estConf, preds_per_lvl, gtDisp): 
        
        if gtDisp is not None:
            gtDisp = gtDisp.unsqueeze(dim = 1)
            label = (torch.abs(torch.sub(preds_per_lvl, gtDisp)) < 3).type(torch.FloatTensor)
            label = label.cuda()

            mask = (gtDisp > 0) & (gtDisp < (self.max_disp))
            mask = mask.detach_()

            bceloss = torch.nn.BCEWithLogitsLoss()

            conf_loss = bceloss(estConf[mask], label[mask])
            loss = conf_loss
            
        else:
            loss = None

        return  loss

    def __call__(self, estConf, preds, gtDisp):
        if not isinstance(estConf, (list, tuple)):
            estConf = [estConf]

        if self.weights is None:
            self.weights = [1.0] * len(estConf)

        # compute loss for per level
        if gtDisp is not None:
            loss_all_level = [
                self.loss_per_level(est_conf_per_lvl, preds_per_lvl, gtDisp) 
                for est_conf_per_lvl, preds_per_lvl in zip(estConf, preds)
            ]
            # re-weight loss per level
            weighted_loss_all_level = dict()
            for i, loss_per_level in enumerate(loss_all_level):
                name = "conf_loss_lvl{}".format(i)
                weighted_loss_all_level[name] = self.weights[i] * loss_per_level
        else:
            weighted_loss_all_level = dict()
            for i in range(len(estConf)):
                name = "conf_loss_lvl{}".format(i)
                weighted_loss_all_level[name] = self.weights[i] * 0.

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'ConfidenceLoss'