import torch.nn.functional as F
import torch


def model_loss(disp_ests, disp_gts, mask):
    weights = [1.0, 0.7, 0.5, 0.5]
    all_losses = []
    mask = mask.unsqueeze(dim=1)
    for disp_est,disp_gt, weight in zip(disp_ests, disp_gts, weights):
        # print(disp_est.size(), disp_gt.size(), mask.size())
        # disp_est = disp_est.squeeze()
        l1loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)
        # smoothloss = get_smooth_loss(disp_est, imgL)
        # loss = (l1loss + smoothloss)
        all_losses.append(weight * l1loss)
    return sum(all_losses)



def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    disp = torch.unsqueeze(disp,dim = 1)
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()