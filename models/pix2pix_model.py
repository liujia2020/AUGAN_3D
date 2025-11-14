import torch
from .base_model import BaseModel
from . import network
from thop import profile


class Pix2PixModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # [!!] 战术提醒：这里的 'unet_128' 已经被 network.py 内部修改为
        #               使用 6-downsamplings (适用于 64x64x64)。
        parser.set_defaults(norm='instance', netG='unet_128', netD='basic', use_sab=False, name='unet_b002')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            # [改造点 2] L1 -> L2
            parser.add_argument('--lambda_L2', type=float, default=1, help='weight for L2 loss')
            # [改造点 5] (隐含) 添加 lr_D_ratio
            # parser.add_argument('--lr_D_ratio', type=float, default=0.1, help='ratio of D learning rate to G learning rate')


        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        
        # [改造点 3] 移除 contentLoss
        self.loss_names = ['generator_adversarial_loss', 'generator_pixelwise_l2_loss', 'discriminator_loss_on_real', 'discriminator_loss_on_fake']

        # [改造点 4] 移除 2D VGG 和 feature_extractor
        # vgg = torchvision.models.vgg19(pretrained=True)
        # vgg.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # self.feature_extractor = network.FeatureExtractor(vgg).to(self.device)
        # self.feature_extractor.eval() 
        
        
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            
        # [!!] 关键依赖：
        # 这里的 network.define_G 和 network.define_D
        # 必须是上一阶段 3D 化改造后的版本。
        
        # define networks (both generator and discriminator)
        self.netG = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.use_sab)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = network.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = network.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss() # (L1 损失函数本身保留，以防未来使用)
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            # [改造点 5] 移除魔法数字 0.1
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * opt.lr_D_ratio, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, low_quality_image, high_quality_image):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        (此函数与 2D/3D 无关，保持不变)
        """
        self.low_quality_image = low_quality_image.to(self.device) 
        self.high_quality_image = high_quality_image.to(self.device) 

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # [!!] 关键依赖：netG 必须是 3D U-Net
        self.generated_image = self.netG(self.low_quality_image)  # G(A)

    def backward_D(self):
        """
        Calculate GAN loss for the discriminator
        (此函数逻辑与 2D/3D 无关，保持不变)
        """
        "学习识别假图像"
        concatenated_input_and_generated = torch.cat((self.low_quality_image, self.generated_image), 1)  
        pred_fake = self.netD(concatenated_input_and_generated.detach())
        self.discriminator_loss_on_fake = self.criterionGAN(pred_fake, False)
        "学习识别假图像"
        concatenated_input_and_target = torch.cat((self.low_quality_image, self.high_quality_image), 1)
        pred_real = self.netD(concatenated_input_and_target)
        self.discriminator_loss_on_real = self.criterionGAN(pred_real, True)
        
        # combine loss and calculate gradients
        self.loss_D = (self.discriminator_loss_on_fake + self.discriminator_loss_on_real) *0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L2 loss for the generator"""
        
        concatenated_input_and_generated = torch.cat((self.low_quality_image, self.generated_image), 1)
        pred_fake = self.netD(concatenated_input_and_generated)
        self.generator_adversarial_loss = self.criterionGAN(pred_fake, True) # 对抗损失
        
        # [改造点 6] L1 -> L2
        self.generator_pixelwise_l2_loss = self.criterionL2(self.generated_image, self.high_quality_image) * self.opt.lambda_L2 # 像素损失

        
        # [改造点 7] 移除 2D VGG Content Loss
        # self.target_image_features = self.feature_extractor(self.high_quality_image)
        # self.generated_image_features = self.feature_extractor(self.generated_image)
        # pred_fake1 = torch.cat((self.target_image_features,self.generated_image_features),1)
        # self.contentLoss = self.criterionGAN(pred_fake1,True) # 内容损失

        # [改造点 8] 更新总损失
        self.loss_G = self.generator_adversarial_loss + self.generator_pixelwise_l2_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Update network weights
        (此函数逻辑与 2D/3D 无关，保持不变)
        """
        self.forward() # compute fake image G(A)
        
        # update D
        self.set_requires_grad(self.netD, True) # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradient to zero
        self.backward_D()   # calculate gradient for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G

        self.optimizer_G.step()             # udpate G's weights