from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# import torchvision
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import copy
# (SAB 导入保持禁用)

def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer (3D VERSION)
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance': # InstanceNormalization
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """(此函数与 2D/3D 无关，保持不变)"""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.niter_decay + 1)
            return max(0.0, lr_l)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """(此函数与 2D/3D 无关，保持不变)"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier': 
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1: 
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    # [!!] 修正：init_weights 不返回任何东西

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """(此函数与 2D/3D 无关，保持不变)"""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_sab=False):
    """Create a generator (3D Version)"""
    net = None
    norm_layer = get_norm_layer(norm_type=norm) 

    if netG == 'resnet_9blocks':
        raise NotImplementedError('ResnetGenerator 尚未 3D 化')
    elif netG == 'resnet_6blocks':
        raise NotImplementedError('ResnetGenerator 尚未 3D 化')
    
    # [!!] 关键修正：'unet_128' 和 'unet_256' 均使用 num_downs=6
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
    
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator (3D Version)"""
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic': 
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers': 
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    """(此函数与 2D/3D 无关，保持不变)"""
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class FeatureExtractor(nn.Module):
    """(2D VGG - 按计划废弃，保持不变)"""
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator (3D VERSION)"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_sab=False):
        super(UnetGenerator, self).__init__()
        
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, inter=True)  
        
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, inter=True)
        
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer) 

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection (3D VERSION)."""
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, inter=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self_inter = inter # (2D 注意力标志，已禁用)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
            
        if input_nc is None:
            input_nc = outer_nc
        
        # 3D 模块
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False) 
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, False) # [!!] uprelu 在此定义
        upnorm = norm_layer(outer_nc)
        
        # (2D 注意力已按计划移除)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            
            # [!!! 战术修正（最终版）!!!]
            # 保持 2D 代码的 LeakyReLU，并添加 Tanh
            up = [upconv, uprelu, nn.Tanh()] 
            
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu]
            up = [upconv, uprelu, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downconv, downrelu, downnorm]
            up = [upconv, uprelu, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # (2D 注意力逻辑已按计划移除)
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator (3D VERSION)"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
 
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# (其他 2D ResNet/SRGAN 等模块保持未定义)

# (其他 2D ResNet/SRGAN 等模块保持未定义)

# from __future__ import print_function
# import torch
# import torch.nn as nn
# from torch.nn import init
# import functools
# from torch.optim import lr_scheduler
# # import torchvision
# import torch.nn.functional as F
# import torch.optim as optim
# from PIL import Image
# import matplotlib.pyplot as plt
# import copy
# # [改造点] 移除 2D 注意力导入，因为我们暂时禁用它们
# # from models.SAB import ChannelAttention,SpatialAttention,SpatioAttention, LocalsAwareAttention, GlobalAwareAttention, PixelAwareAttention
# # [!!] 战术决策：暂时禁用 SAB.py
# # (如果 SAB.py 存在，上面的导入会失败，所以我们注释掉它)
# # (当您创建 SAB_3D.py 时，我们再恢复它)


# def get_norm_layer(norm_type='instance'):
#     """
#     Return a normalization layer (3D VERSION)
#     [改造点 1] 将 2D 归一化替换为 3D
#     """
#     if norm_type == 'batch':
#         # 2D -> 3D
#         norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance': # InstanceNormalization
#         # 2D -> 3D
#         norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         norm_layer = None
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer

# def get_scheduler(optimizer, opt):
#     """Return a learning rate scheduler
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     if opt.lr_policy == 'linear':
#         def lambda_rule(epoch):
#             # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
#             lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.niter_decay + 1)
#             # return lr_l
#             return max(0.0, lr_l)  # <--- 修正后的代码：确保返回值永远不会小于0
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif opt.lr_policy == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
#     elif opt.lr_policy == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif opt.lr_policy == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#     return scheduler

# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier': # 
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         # [改造点 1.1] 确保 3D 归一化层也被正确初始化
#         elif classname.find('BatchNorm') != -1:  # 捕获 BatchNorm2d 和 BatchNorm3d
#             if m.weight is not None: # 修正：检查 weight 是否存在
#                 init.normal_(m.weight.data, 1.0, init_gain)
#             if m.bias is not None: # 修正：检查 bias 是否存在
#                 init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>

# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):# 定义神经网络（G和D）
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)
#     init_weights(net, init_type, init_gain=init_gain)
#     return net

# def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_sab=False):
#     """Create a generator
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm) # [改造点 1] 此处将自动获取 3D 归一化

#     if netG == 'resnet_9blocks':
#         # [!!] 战术提醒：ResnetGenerator 尚未 3D 化。
#         # (如果使用此选项，模型将失败)
#         raise NotImplementedError('ResnetGenerator 尚未 3D 化')
#     elif netG == 'resnet_6blocks':
#         # [!!] 战术提醒：ResnetGenerator 尚未 3D 化。
#         raise NotImplementedError('ResnetGenerator 尚未 3D 化')
#     elif netG == 'unet_128':
#         # (UnetGenerator 已在下方被 3D 化)
#         # [!!] 关键修正：'unet_128' (2D) 对应 7 次下采样
#         # 但我们的 3D Patch (64) 只能 6 次
#         net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
#     elif netG == 'unet_256':
#         # (UnetGenerator 已在下方被 3D 化)
#         # [!!] 关键修正：'unet_256' (2D) 对应 8 次下采样
#         # 我们的 3D Patch (64) 只能 6 次
#         net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
#     elif netG == 'srgan':
#         raise NotImplementedError('SRGAN 尚未 3D 化')
#     elif netG == 'VDSR':
#         raise NotImplementedError('VDSR 尚未 3D 化')
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
#     return init_net(net, init_type, init_gain, gpu_ids)

# def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Create a discriminator
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm) # [改造点 1] 此处将自动获取 3D 归一化

#     if netD == 'basic':  # default PatchGAN classifier
#         # (NLayerDiscriminator 已在下方被 3D 化)
#         net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
#     elif netD == 'n_layers':  # more options
#         # (NLayerDiscriminator 已在下方被 3D 化)
#         net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
#     elif netD == 'pixel':     # classify if each pixel is real or fake
#         raise NotImplementedError('PixelDiscriminator 尚未 3D 化')
#     elif netD == 'srgan':
#         raise NotImplementedError('SRDiscriminator 尚未 3D 化')
#     else:
#         raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
#     return init_net(net, init_type, init_gain, gpu_ids)

# class GANLoss(nn.Module):
#     """Define different GAN objectives.
#     (此函数与 2D/3D 无关，保持不变)
#     """

#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
#         """ Initialize the GANLoss class.
#         ... (保持不变) ...
#         """
#         super(GANLoss, self).__init__()
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#         self.gan_mode = gan_mode
#         if gan_mode == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif gan_mode == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif gan_mode in ['wgangp']:
#             self.loss = None
#         # elif gan_mode == 'style_texture':
#             # self.loss = get_style_texture_algorithm()
#         else:
#             raise NotImplementedError('gan mode %s not implemented' % gan_mode)

#     def get_target_tensor(self, prediction, target_is_real):
#         """Create label tensors with the same size as the input.
#         ... (保持不变) ...
#         """

#         if target_is_real:
#             target_tensor = self.real_label
#         else:
#             target_tensor = self.fake_label
#         return target_tensor.expand_as(prediction)

#     def __call__(self, prediction, target_is_real):
#         """Calculate loss given Discriminator's output and ground truth labels.
#         ... (保持不变) ...
#         """
#         if self.gan_mode in ['lsgan', 'vanilla']:
#             target_tensor = self.get_target_tensor(prediction, target_is_real)
#             loss = self.loss(prediction, target_tensor)
#         elif self.gan_mode == 'wgangp': #WGAN损失函数为什么时求均值？
#             if target_is_real:
#                 loss = -prediction.mean()
#             else:
#                 loss = prediction.mean()
#         return loss

# # [改造点 4] FeatureExtractor (2D VGG) 按计划保持不变。
# # 我们将在 pix2pix_model.py 中禁用它。
# class FeatureExtractor(nn.Module):
#     def __init__(self, cnn, feature_layer=11):
#         super(FeatureExtractor, self).__init__()
#         self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

#     def forward(self, x):
#         return self.features(x)


# class UnetGenerator(nn.Module):
#     """Create a Unet-based generator (3D VERSION)"""

#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_sab=False):
#         """Construct a Unet generator
#         Parameters:
#             ... (保持不变) ...
#         We construct the U-Net from the innermost layer to the outermost layer.
#         """
#         super(UnetGenerator, self).__init__()
#         # [改造点 1] 确保 norm_layer 是 3D 的 (它是由 define_G 传入的)
        
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, inter=True)  # add the innermost layer

#         if type(norm_layer) == functools.partial:
#             # 2D -> 3D
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             # 2D -> 3D
#             use_bias = norm_layer == nn.InstanceNorm3d


#         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, inter=True)

#         # [!!] 战术修正：确保 num_downs=6 时这些层仍然被构建
#         # num_downs=6: range(1) -> i=0
#         # num_downs=7: range(2) -> i=0, 1
#         # num_downs=8: range(3) -> i=0, 1, 2
#         # (原始 2D 代码逻辑保持不变)

#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)

# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection (3D VERSION)."""

#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, inter=False):
#         """Construct a 3D Unet submodule with skip connections."""
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.innermost = innermost
#         self.outermost = outermost
#         self.inter = inter # (这个 'inter' 标志与 2D 注意力相关)

#         if type(norm_layer) == functools.partial:
#             # 2D -> 3D
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             # 2D -> 3D
#             use_bias = norm_layer == nn.InstanceNorm3d
            
#         if input_nc is None:
#             input_nc = outer_nc
#         self.input_nc = input_nc
#         self.outer_nc = outer_nc
        
#         # [改造点 2.1] 2D -> 3D
#         downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, False)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.LeakyReLU(0.2, False)
#         upnorm = norm_layer(outer_nc)
        
#         # [改造点 2.2] 战术决策：移除 2D 注意力模块
#         # localatt = LocalAwareAttention()
#         # self.pixelatt = PixelAwareAttention(inner_nc)

#         if outermost:
#             # [改造点 2.1] 2D -> 3D
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
            
#             # [!!! 战术修正 !!!]
#             # 添加 nn.Tanh() 以匹配 [-1, 1] 的数据归一化
#             # up = [upconv, nn.Tanh()] 
#             up = [upconv, uprelu, nn.Tanh()]
            
#             model = down + [submodule] + up

#         elif innermost:
#             # [改造点 2.1] 2D -> 3D
#             upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downconv, downrelu] # 原始 2D 逻辑是 [downconv, downrelu]
#             # 修正：保持 [downconv, downrelu]
#             up = [upconv, uprelu, upnorm]
            
#             # [改造点 2.2] 移除 2D 注意力 (localatt)
#             # up = [uprelu, upconv, upnorm] # 原始 2D 逻辑
            
#             model = down + up
#         else:
#             # [改造点 2.1] 2D -> 3D
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downconv, downrelu, downnorm]

#             # [改造点 2.2] 移除 2D 注意力 (localatt)
#             # up = [uprelu, upconv, upnorm] # 原始 2D 逻辑
#             up = [upconv, uprelu, upnorm]

#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up

#         self.model = nn.Sequential(*model)
        
#         # [改造点 2.2] 移除 2D 注意力模块
#         # self.pa = PixelAwareAttention(input_nc)


#     def forward(self, x):
#         # [改造点 2.3] 简化 forward，移除 2D 注意力 (self.pa) 相关的逻辑
#         if self.outermost:
#             return self.model(x)
#         else:
#             # 简化为 3D U-Net 标准跳跃连接：
#             return torch.cat([x, self.model(x)], 1)


# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator (3D VERSION)"""

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a 3D PatchGAN discriminator."""
#         super(NLayerDiscriminator, self).__init__()
        
#         # [改造点 1] 确保 norm_layer 是 3D 的 (它是由 define_D 传入的)
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm has affine parameters
#             # 2D -> 3D
#             use_bias = norm_layer.func != nn.BatchNorm3d
#         else:
#             # 2D -> 3D
#             use_bias = norm_layer != nn.BatchNorm3d

#         kw = 4
#         padw = 1
        
#         # [改造点 3.1] 2D -> 3D
#         sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 # [改造点 3.1] 2D -> 3D
#                 nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
 
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             # [改造点 3.1] 2D -> 3D
#             nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
        
#         # [改造点 3.1] 2D -> 3D
#         sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)

# [!!] 战术提醒：以下 2D 模块尚未 3D 化，如果使用它们会报错
# (根据战略，我们目前不使用它们)
#
# class ResnetGenerator(nn.Module):
# ... (2D) ...
#
# class ResnetBlock(nn.Module):
# ... (2D) ...
#
# class PixelDiscriminator(nn.Module):
# ... (2D) ...
#
# class SRGenerator(nn.Module):
# ... (2D) ...
#
# class SRDiscriminator(nn.Module):
# ... (2D) ...
#
# class VDCGAN(nn.Module):
# ... (2D) ...



# from __future__ import print_function
# import torch
# import torch.nn as nn
# from torch.nn import init
# import functools
# from torch.optim import lr_scheduler
# # import torchvision
# import torch.nn.functional as F
# import torch.optim as optim
# from PIL import Image
# import matplotlib.pyplot as plt
# import copy
# # [改造点] 移除 2D 注意力导入，因为我们暂时禁用它们
# # from models.SAB import ChannelAttention,SpatialAttention,SpatioAttention, LocalAwareAttention, GlobalAwareAttention, PixelAwareAttention
# # [!!] 战术决策：暂时禁用 SAB.py
# # (如果 SAB.py 存在，上面的导入会失败，所以我们注释掉它)
# # (当您创建 SAB_3D.py 时，我们再恢复它)


# def get_norm_layer(norm_type='instance'):
#     """
#     Return a normalization layer (3D VERSION)
#     [改造点 1] 将 2D 归一化替换为 3D
#     """
#     if norm_type == 'batch':
#         # 2D -> 3D
#         norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance': # InstanceNormalization
#         # 2D -> 3D
#         norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         norm_layer = None
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer

# def get_scheduler(optimizer, opt):
#     """Return a learning rate scheduler
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     if opt.lr_policy == 'linear':
#         def lambda_rule(epoch):
#             # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
#             lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.niter_decay + 1)
#             # return lr_l
#             return max(0.0, lr_l)  # <--- 修正后的代码：确保返回值永远不会小于0
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif opt.lr_policy == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
#     elif opt.lr_policy == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif opt.lr_policy == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#     return scheduler

# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier': # 
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         # [改造点 1.1] 确保 3D 归一化层也被正确初始化
#         elif classname.find('BatchNorm') != -1:  # 捕获 BatchNorm2d 和 BatchNorm3d
#             if m.weight is not None: # 修正：检查 weight 是否存在
#                 init.normal_(m.weight.data, 1.0, init_gain)
#             if m.bias is not None: # 修正：检查 bias 是否存在
#                 init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>

# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):# 定义神经网络（G和D）
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)
#     init_weights(net, init_type, init_gain=init_gain)
#     return net

# def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], use_sab=False):
#     """Create a generator
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm) # [改造点 1] 此处将自动获取 3D 归一化

#     if netG == 'resnet_9blocks':
#         # [!!] 战术提醒：ResnetGenerator 尚未 3D 化。
#         # (如果使用此选项，模型将失败)
#         raise NotImplementedError('ResnetGenerator 尚未 3D 化')
#     elif netG == 'resnet_6blocks':
#         # [!!] 战术提醒：ResnetGenerator 尚未 3D 化。
#         raise NotImplementedError('ResnetGenerator 尚未 3D 化')
#     elif netG == 'unet_128':
#         # (UnetGenerator 已在下方被 3D 化)
#         # [!!] 关键修正：'unet_128' (2D) 对应 7 次下采样
#         # 但我们的 3D Patch (64) 只能 6 次
#         net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
#     elif netG == 'unet_256':
#         # (UnetGenerator 已在下方被 3D 化)
#         # [!!] 关键修正：'unet_256' (2D) 对应 8 次下采样
#         # 我们的 3D Patch (64) 只能 6 次
#         net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_sab=use_sab)
#     elif netG == 'srgan':
#         raise NotImplementedError('SRGAN 尚未 3D 化')
#     elif netG == 'VDSR':
#         raise NotImplementedError('VDSR 尚未 3D 化')
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
#     return init_net(net, init_type, init_gain, gpu_ids)

# def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Create a discriminator
#     (此函数与 2D/3D 无关，保持不变)
#     """
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm) # [改造点 1] 此处将自动获取 3D 归一化

#     if netD == 'basic':  # default PatchGAN classifier
#         # (NLayerDiscriminator 已在下方被 3D 化)
#         net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
#     elif netD == 'n_layers':  # more options
#         # (NLayerDiscriminator 已在下方被 3D 化)
#         net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
#     elif netD == 'pixel':     # classify if each pixel is real or fake
#         raise NotImplementedError('PixelDiscriminator 尚未 3D 化')
#     elif netD == 'srgan':
#         raise NotImplementedError('SRDiscriminator 尚未 3D 化')
#     else:
#         raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
#     return init_net(net, init_type, init_gain, gpu_ids)

# class GANLoss(nn.Module):
#     """Define different GAN objectives.
#     (此函数与 2D/3D 无关，保持不变)
#     """

#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
#         """ Initialize the GANLoss class.
#         ... (保持不变) ...
#         """
#         super(GANLoss, self).__init__()
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#         self.gan_mode = gan_mode
#         if gan_mode == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif gan_mode == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif gan_mode in ['wgangp']:
#             self.loss = None
#         # elif gan_mode == 'style_texture':
#             # self.loss = get_style_texture_algorithm()
#         else:
#             raise NotImplementedError('gan mode %s not implemented' % gan_mode)

#     def get_target_tensor(self, prediction, target_is_real):
#         """Create label tensors with the same size as the input.
#         ... (保持不变) ...
#         """

#         if target_is_real:
#             target_tensor = self.real_label
#         else:
#             target_tensor = self.fake_label
#         return target_tensor.expand_as(prediction)

#     def __call__(self, prediction, target_is_real):
#         """Calculate loss given Discriminator's output and ground truth labels.
#         ... (保持不变) ...
#         """
#         if self.gan_mode in ['lsgan', 'vanilla']:
#             target_tensor = self.get_target_tensor(prediction, target_is_real)
#             loss = self.loss(prediction, target_tensor)
#         elif self.gan_mode == 'wgangp': #WGAN损失函数为什么时求均值？
#             if target_is_real:
#                 loss = -prediction.mean()
#             else:
#                 loss = prediction.mean()
#         return loss

# # [改造点 4] FeatureExtractor (2D VGG) 按计划保持不变。
# # 我们将在 pix2pix_model.py 中禁用它。
# class FeatureExtractor(nn.Module):
#     def __init__(self, cnn, feature_layer=11):
#         super(FeatureExtractor, self).__init__()
#         self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

#     def forward(self, x):
#         return self.features(x)


# class UnetGenerator(nn.Module):
#     """Create a Unet-based generator (3D VERSION)"""

#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, use_sab=False):
#         """Construct a Unet generator
#         Parameters:
#             ... (保持不变) ...
#         We construct the U-Net from the innermost layer to the outermost layer.
#         """
#         super(UnetGenerator, self).__init__()
#         # [改造点 1] 确保 norm_layer 是 3D 的 (它是由 define_G 传入的)
        
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, inter=True)  # add the innermost layer

#         if type(norm_layer) == functools.partial:
#             # 2D -> 3D
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             # 2D -> 3D
#             use_bias = norm_layer == nn.InstanceNorm3d


#         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, inter=True)

#         # [!!] 战术修正：确保 num_downs=6 时这些层仍然被构建
#         # num_downs=6: range(1) -> i=0
#         # num_downs=7: range(2) -> i=0, 1
#         # num_downs=8: range(3) -> i=0, 1, 2
#         # (原始 2D 代码逻辑保持不变)

#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer) 
#         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)

# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection (3D VERSION)."""

#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, inter=False):
#         """Construct a 3D Unet submodule with skip connections."""
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.innermost = innermost
#         self.outermost = outermost
#         self.inter = inter # (这个 'inter' 标志与 2D 注意力相关)

#         if type(norm_layer) == functools.partial:
#             # 2D -> 3D
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             # 2D -> 3D
#             use_bias = norm_layer == nn.InstanceNorm3d
            
#         if input_nc is None:
#             input_nc = outer_nc
#         self.input_nc = input_nc
#         self.outer_nc = outer_nc
        
#         # [改造点 2.1] 2D -> 3D
#         downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, False)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.LeakyReLU(0.2, False)
#         upnorm = norm_layer(outer_nc)
        
#         # [改造点 2.2] 战术决策：移除 2D 注意力模块
#         # localatt = LocalAwareAttention()
#         # self.pixelatt = PixelAwareAttention(inner_nc)

#         if outermost:
#             # [改造点 2.1] 2D -> 3D
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
            
#             # [!!! 战术修正 !!!]
#             # 添加 nn.Tanh() 以匹配 [-1, 1] 的数据归一化
#             up = [uprelu, upconv, nn.Tanh()] 
#             # (原始 2D 代码是 [uprelu, upconv]，这在技术上是错的，
#             #  但 3D-GANs 必须有 Tanh)
#             # 修正顺序：激活函数应在卷积之后
#             up = [upconv, nn.ReLU(True), nn.Tanh()] # 标准 U-Net 使用 ReLU
#             # 再次修正：原始代码使用 LeakyReLU
#             up = [upconv, uprelu, nn.Tanh()] 
#             # 最终修正：原始代码 uprelu 在 upconv 之前...
#             # ... 这似乎不对。标准 pix2pix 结构是：
#             # ReLU, ConvTranspose, BatchNorm
#             # 我们的 2D 原始代码是：
#             # upconv, uprelu
#             # 让我们遵循原始代码的激活函数，但添加 Tanh
#             # [最终决定]：
#             up = [uprelu, upconv, nn.Tanh()] # 保持 2D 逻辑，添加 Tanh
            
#             model = down + [submodule] + up

#         elif innermost:
#             # [改造点 2.1] 2D -> 3D
#             upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv] # 原始 2D 逻辑是 [downconv, downrelu]
#             # 修正：保持 [downconv, downrelu]
#             down = [downconv, downrelu]
            
#             # [改造点 2.2] 移除 2D 注意力 (localatt)
#             up = [uprelu, upconv, upnorm] # 原始 2D 逻辑
            
#             model = down + up
#         else:
#             # [改造点 2.1] 2D -> 3D
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downconv, downrelu, downnorm]

#             # [改造点 2.2] 移除 2D 注意力 (localatt)
#             up = [uprelu, upconv, upnorm] # 原始 2D 逻辑

#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up

#         self.model = nn.Sequential(*model)
        
#         # [改造点 2.2] 移除 2D 注意力模块
#         # self.pa = PixelAwareAttention(input_nc)


#     def forward(self, x):
#         # [改造点 2.3] 简化 forward，移除 2D 注意力 (self.pa) 相关的逻辑
#         if self.outermost:
#             return self.model(x)
#         else:
#             # 简化为 3D U-Net 标准跳跃连接：
#             return torch.cat([x, self.model(x)], 1)


# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator (3D VERSION)"""

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a 3D PatchGAN discriminator."""
#         super(NLayerDiscriminator, self).__init__()
        
#         # [改造点 1] 确保 norm_layer 是 3D 的 (它是由 define_D 传入的)
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm has affine parameters
#             # 2D -> 3D
#             use_bias = norm_layer.func != nn.BatchNorm3d
#         else:
#             # 2D -> 3D
#             use_bias = norm_layer != nn.BatchNorm3d

#         kw = 4
#         padw = 1
        
#         # [改造点 3.1] 2D -> 3D
#         sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 # [改造点 3.1] 2D -> 3D
#                 nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
 
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             # [改造点 3.1] 2D -> 3D
#             nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
        
#         # [改造点 3.1] 2D -> 3D
#         sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)

# # [!!] 战术提醒：以下 2D 模块尚未 3D 化，如果使用它们会报错
# # (根据战略，我们目前不使用它们)
# #
# # class ResnetGenerator(nn.Module):
# # ... (2D) ...
# #
# # class ResnetBlock(nn.Module):
# # ... (2D) ...
# #
# # class PixelDiscriminator(nn.Module):
# # ... (2D) ...
# #
# # class SRGenerator(nn.Module):
# # ... (2D) ...
# #
# # class SRDiscriminator(nn.Module):
# # ... (2D) ...
# #
# # class VDCGAN(nn.Module):
# # ... (2D) ...