import torch
import torch.nn as nn
import torch.nn.functional as F

class GetSmoothLoss(object):
    """

        Args:
            max_disp, (int): the max of Disparity. default: 192
            start_disp, (int): the start searching disparity index, usually be 0
            weights, (list of float or None): weight for each scale of estCost.
            sparse, (bool): whether the ground-truth disparity is sparse, for example, KITTI is sparse, but SceneFlow is not. default: False

        Inputs:
            estDisp, (Tensor or list of Tensor): the estimated disparity map, in (BatchSize, 1, Height, Width) layout.
            gtDisp, (Tensor): the ground truth disparity map, in (BatchSize, 1, Height, Width) layout.

        Outputs:
            loss, (dict), the loss of each level
    """

    def __init__(self, max_disp, start_disp=0, weights=None, sparse=False):
        self.max_disp = max_disp
        self.weights = weights
        self.start_disp = start_disp
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, disp, img):
        N, C, H, W = disp.shape
        
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        #print(grad_disp_x.size())
        #print(grad_disp_y.size())


        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def __call__(self, estDisp, leftimg):
        if not isinstance(estDisp, (list, tuple)):
            estDisp = [estDisp]

        if self.weights is None:
            self.weights = [1.0] * len(estDisp)

        # compute loss for per level
        loss_all_level = []
        for est_disp_per_lvl in estDisp:
                loss_all_level.append(
                    self.loss_per_level(est_disp_per_lvl, leftimg)
                )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "smooth_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'SmoothL1Loss'