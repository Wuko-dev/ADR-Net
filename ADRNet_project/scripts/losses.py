import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.vgg_arch import VGGFeatureExtractor, Registry
from scripts.loss_utils import *
import torchvision.transforms as transforms
import numpy as np

_reduction_modes = ['none', 'mean', 'sum']

# GTMean亮度一致性损失（只对齐灰度均值）
class GTMeanLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GTMeanLoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def rgb_to_y(self, x):
        return 0.299 * x[:, 0:1, :, :] + \
               0.587 * x[:, 1:2, :, :] + \
               0.114 * x[:, 2:3, :, :]

    def forward(self, pred, target):
        # 转灰度
        y_pred = self.rgb_to_y(pred)
        y_target = self.rgb_to_y(target)

        # 计算全局均值（对应 cv2.mean）
        mean_pred = y_pred.mean(dim=[2, 3])   # [B,1]
        mean_target = y_target.mean(dim=[2, 3])

        # L1 loss
        loss = torch.abs(mean_pred - mean_target)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss
    
# 亮度一致性损失
class IlluminationConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(IlluminationConsistencyLoss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.reduction = reduction

    def rgb_to_y(self, x):
        return 0.299 * x[:, 0:1, :, :] + \
               0.587 * x[:, 1:2, :, :] + \
               0.114 * x[:, 2:3, :, :]

    def forward(self, pred, target):
        y_pred = self.rgb_to_y(pred)
        y_target = self.rgb_to_y(target)

        loss = torch.abs(y_pred - y_target)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss

# 普通TV—LOSS
class TVLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TVLoss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x):
        dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

        loss = dx.mean() + dy.mean()

        if self.reduction == 'sum':
            loss = dx.sum() + dy.sum()
        elif self.reduction == 'none':
            loss = dx + dy  # 不常用

        return self.loss_weight * loss

# 进阶版本的TV_loss
class EdgeAwareTVLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(EdgeAwareTVLoss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        
        self.loss_weight = loss_weight
        self.reduction = reduction

    def gradient(self, x):
        dx = x[:, :, :, :-1] - x[:, :, :, 1:]
        dy = x[:, :, :-1, :] - x[:, :, 1:, :]
        return dx, dy

    def forward(self, pred, input_img):
        pred_dx, pred_dy = self.gradient(pred)
        img_dx, img_dy = self.gradient(input_img)

        weight_x = torch.exp(-torch.abs(img_dx))
        weight_y = torch.exp(-torch.abs(img_dy))

        loss = (pred_dx.abs() * weight_x).mean() + \
               (pred_dy.abs() * weight_y).mean()

        if self.reduction == 'sum':
            loss = (pred_dx.abs() * weight_x).sum() + \
                   (pred_dy.abs() * weight_y).sum()

        return self.loss_weight * loss
            
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                            f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class GTMean_L1Loss(nn.Module):
    def __init__(self, module, loss_weight='dou_KL_2', reduction='mean', sigma_=0.1):
        super(GTMean_L1Loss, self).__init__()
        self.loss = module  
        self.sigma = sigma_
        self.reduction = reduction
        self.transform = transforms.Grayscale(num_output_channels=1)

        if loss_weight == 'dou_KL_2':
            self.iter_weight = self.double_KL_div_2
        else:
            raise ValueError('Currently only dou_KL_2 is supported')

    def double_KL_div_2(self, mu_1, mu_2, sigma):
        sigma_1 = sigma * mu_1
        sigma_2 = sigma * mu_2
        mu_M = 0.5 * (mu_1 + mu_2)
        sigma_M = torch.sqrt((sigma_1 ** 2 + sigma_2 ** 2) / 2)
        KL_2_weight = 0.5 * self._kl_div(mu_1, mu_M, sigma_1, sigma_M) + \
                      0.5 * self._kl_div(mu_2, mu_M, sigma_2, sigma_M)
        return KL_2_weight

    def _kl_div(self, mu_1, mu_2, sigma_1, sigma_2):
        return torch.log(sigma_2 / (sigma_1 + 1e-8)) + \
               0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2 + 1e-8) - 0.5

    def forward(self, pred, target, num_iter=None):
        # 灰度均值作为亮度代理
        E_y = torch.mean(self.transform(target), dim=(1, 2, 3))
        E_x = torch.mean(self.transform(pred), dim=(1, 2, 3))

        # 计算 GT-Mean 权重
        loss_weight_unclip = self.iter_weight(torch.abs(E_y), torch.abs(E_x), self.sigma)
        loss_weight = torch.clamp(loss_weight_unclip, 0, 1).detach()

        # 亮度对齐图像
        m = E_y / (E_x + 1e-8)
        pred_aligned = torch.clamp(m[:, None, None, None] * pred, 0, 1)

        # SSIM 损失
        ssim_original = self.loss(pred, target)
        ssim_aligned = self.loss(pred_aligned, target)

        # GT-Mean 组合
        total_loss = loss_weight * ssim_original + (1 - loss_weight) * ssim_aligned

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss        

        
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

        k = torch.tensor([[0.05, 0.25, 0.4, 0.25, 0.05]], dtype=torch.float32)
        kernel = torch.matmul(k.t(), k)             # [5, 5]
        kernel = kernel.unsqueeze(0).unsqueeze(0)   # [1, 1, 5, 5]
        kernel = kernel.repeat(3, 1, 1, 1)          # [3, 1, 5, 5]

        # 关键：注册为 buffer，这样 .to(device) 时会自动移动到 GPU
        self.register_buffer('kernel', kernel)

    def conv_gauss(self, img):
        n_channels = img.shape[1]

        # 如果输入不是 3 通道，则自动适配通道数
        if n_channels != self.kernel.shape[0]:
            kernel = self.kernel[:1].repeat(n_channels, 1, 1, 1)
        else:
            kernel = self.kernel

        _, _, kh, kw = kernel.shape
        img = F.pad(img, (kw // 2, kw // 2, kh // 2, kh // 2), mode='replicate')

        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]

        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4

        filtered = self.conv_gauss(new_filter)
        diff = current - filtered

        return diff

    def forward(self, x, y):
        loss = F.mse_loss(
            self.laplacian_kernel(x),
            self.laplacian_kernel(y),
            reduction=self.reduction
        )
        return loss * self.loss_weight



    
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                layer_weights,
                vgg_type='vgg19',
                use_input_norm=True,
                range_norm=True,
                perceptual_weight=1.0,
                style_weight=0.,
                criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight


class GTMean_SSIM(nn.Module):
    def __init__(self, ssim_module, loss_weight='dou_KL_2', reduction='mean', sigma_=0.1):
        super(GTMean_SSIM, self).__init__()
        self.ssim = ssim_module  # 传入一个 SSIM 实例
        self.sigma = sigma_
        self.reduction = reduction
        self.transform = transforms.Grayscale(num_output_channels=1)

        if loss_weight == 'dou_KL_2':
            self.iter_weight = self.double_KL_div_2
        else:
            raise ValueError('Currently only dou_KL_2 is supported')

    def double_KL_div_2(self, mu_1, mu_2, sigma):
        sigma_1 = sigma * mu_1
        sigma_2 = sigma * mu_2
        mu_M = 0.5 * (mu_1 + mu_2)
        sigma_M = torch.sqrt((sigma_1 ** 2 + sigma_2 ** 2) / 2)
        KL_2_weight = 0.5 * self._kl_div(mu_1, mu_M, sigma_1, sigma_M) + \
                      0.5 * self._kl_div(mu_2, mu_M, sigma_2, sigma_M)
        return KL_2_weight

    def _kl_div(self, mu_1, mu_2, sigma_1, sigma_2):
        return torch.log(sigma_2 / (sigma_1 + 1e-8)) + \
               0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2 ** 2 + 1e-8) - 0.5

    def forward(self, pred, target, num_iter=None):
        # 灰度均值作为亮度代理
        E_y = torch.mean(self.transform(target), dim=(1, 2, 3))
        E_x = torch.mean(self.transform(pred), dim=(1, 2, 3))

        # 计算 GT-Mean 权重
        loss_weight_unclip = self.iter_weight(torch.abs(E_y), torch.abs(E_x), self.sigma)
        loss_weight = torch.clamp(loss_weight_unclip, 0, 1).detach()

        # 亮度对齐图像
        m = E_y / (E_x + 1e-8)
        pred_aligned = torch.clamp(m[:, None, None, None] * pred, 0, 1)

        # SSIM 损失
        ssim_original = self.ssim(pred, target)
        ssim_aligned = self.ssim(pred_aligned, target)

        # GT-Mean 组合
        total_loss = loss_weight * ssim_original + (1 - loss_weight) * ssim_aligned

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


# def psnr_loss(pred, target, weight=None):
#     mse = F.mse_loss(pred, target, reduction='none').mean(dim=(1, 2, 3))  # 按样本计算MSE
#     psnr = 10 * torch.log10(1 / (mse + 1e-8))  # 避免除以零，保持张量
#     return psnr.mean()  # 保留梯度

# class PSNRLoss(nn.Module):
#     def __init__(self, loss_weight=1.0):
#         super().__init__()
#         self.loss_weight = loss_weight

#     def forward(self, pred, target, weight=None):
#         return -self.loss_weight * psnr_loss(pred, target, weight)


# # 原文章的公式
# class GTmeanLoss(nn.Module):
#     """GTmean loss.

#     Args:
#         loss_weight (float): Loss weight for GTmean loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """

#     def __init__(self, loss_weight='sigmoid', reduction='mean',sigma_=0.1):
#         super(GTmeanLoss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')


#         self.sigma = sigma_
#         print(f"current sigma is {self.sigma}")
#         if loss_weight == 'dou_KL_2':
#             self.iter_weight = self.double_KL_div_2
#         else:        
#             assert False, 'Currently weight is undefined'

            
#         self.reduction = reduction
#         self.transform = transforms.Grayscale(num_output_channels=1)

#     def linear(self, num_iter):
#         return (1.0*torch.tensor(num_iter) / 150000)

#     def KL_div(self, mu_1, mu_2,sigma_1, sigma_2):
#         return torch.log(sigma_2 / sigma_1)+0.5*(sigma_1**2+(mu_1-mu_2)**2)/sigma_2**2 - 0.5

#     def double_KL_div_2(self, mu_1, mu_2,sigma):
#         sigma_1 = sigma * mu_1
#         sigma_2 = sigma * mu_2
#         mu_M = 0.5*(mu_1 + mu_2)
#         sigma_M = torch.sqrt(((sigma_1)**2 + (sigma_2)**2)/2)
#         KL_2_weight = 0.5 * self.KL_div(mu_1, mu_M, sigma_1, sigma_M) + 0.5 * self.KL_div(mu_2, mu_M,sigma_2, sigma_M)

#         return KL_2_weight
    
#     def forward(self, pred, target,num_iter, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         E_y = torch.mean(self.transform(target), dim=(1,2,3))
#         E_x = torch.mean(self.transform(pred), dim=(1,2,3))
        
#         self.loss_weight_unclip =  self.iter_weight(torch.abs(E_y), torch.abs(E_x), self.sigma)
#         self.loss_weight = torch.clip(self.loss_weight_unclip, 0, 1).detach()
        
#         m = E_y / E_x
     
#         pred_clip = torch.clip(m[:,None,None,None]*pred, 0,1)
        
#         L1_loss = l1_loss(pred, target, self.loss_weight[:,None,None,None], reduction=self.reduction)
#         GT_loss = l1_loss(pred_clip, target,  (1-self.loss_weight)[:,None,None,None], reduction=self.reduction)
        
#         total_loss = GT_loss +L1_loss
        
#         return total_loss.mean()


# class L1Loss(nn.Module):
#     """L1 (mean absolute error, MAE) loss.

#     Args:
#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """

#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(L1Loss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction

#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         return self.loss_weight * l1_loss(
#             pred, target, weight, reduction=self.reduction)
    
# class MSELoss(nn.Module):
#     """MSE (L2) loss.

#     Args:
#         loss_weight (float): Loss weight for MSE loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """

#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(MSELoss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction

#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         return self.loss_weight * mse_loss(
#             pred, target, weight, reduction=self.reduction)

# class PSNRLoss(nn.Module):

#     def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
#         super(PSNRLoss, self).__init__()
#         assert reduction == 'mean'
#         self.loss_weight = loss_weight
#         self.scale = 10 / np.log(10)
#         self.toY = toY
#         self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
#         self.first = True

#     def forward(self, pred, target):
#         assert len(pred.size()) == 4
#         if self.toY:
#             if self.first:
#                 self.coef = self.coef.to(pred.device)
#                 self.first = False

#             pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
#             target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

#             pred, target = pred / 255., target / 255.
#             pass
#         assert len(pred.size()) == 4

#         return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

# class CharbonnierLoss(nn.Module):
#     """Charbonnier Loss (L1)"""

#     def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps

#     def forward(self, x, y):
#         diff = x - y
#         # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
#         loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
#         return loss

# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss