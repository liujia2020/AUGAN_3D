# from .base_model import BaseModel
# from . import network


# class TestModel(BaseModel):
#     """ This TesteModel can be used to generate CycleGAN results for only one direction.
#     This model will automatically set '--dataset_mode single', which only loads the images from one collection.

#     See the test instruction for more details.
#     """
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.

#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

#         Returns:
#             the modified parser.

#         The model can only be used during test time. It requires '--dataset_mode single'.
#         You need to specify the network using the option '--model_suffix'.
#         """
#         assert not is_train, 'TestModel cannot be used during training time'
#         parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

#         return parser

#     def __init__(self, opt):
#         """Initialize the pix2pix class.

#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         assert(not opt.isTrain)
#         BaseModel.__init__(self, opt)
#         # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
#         self.loss_names = []
#         # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
#         self.visual_names = ['real', 'fake']
#         # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
#         self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
#         self.netG = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
#                                       opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

#         # assigns the model to self.netG_[suffix] so that it can be loaded
#         # please see <BaseModel.load_networks>
#         setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

#     def set_input(self, data, target):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.

#         Parameters:
#             input : low resolution image
#             target : high resulotion image
#         """
#         self.real_A = data.to(self.device)
#         self.real_B = target.to(self.device)

#     def forward(self):
#         """Run forward pass."""
#         self.fake = self.netG(self.real_A)  # G(real)

#     def optimize_parameters(self):
#         """No optimization for test model."""
#         pass
    
    
    
from .base_model import BaseModel
from . import network


class TestModel(BaseModel):
    """ 
    此 TestModel 仅用于 CycleGAN 结果的单向生成。
    该模型会自动设置 '--dataset_mode single'，仅从一个集合加载图像。
    
    (注意：原始注释与 pix2pix 不符，但我们保留它)
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used during training time'
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        return parser

    def __init__(self, opt):
        """初始化 TestModel 类。"""
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        
        # 命名修正：定义清晰的可视化变量名
        self.visual_names = ['low_quality_image', 'generated_image', 'high_quality_image']
        
        # 损失名称列表为空，因为测试模式不计算损失
        self.loss_names = []
        
        # 模型名称
        self.model_names = ['G' + opt.model_suffix]
        
        # 定义生成器 G
        self.netG = network.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # 将 netG 赋值给 self.netG_[suffix] 以便加载权重
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, low_quality_image, high_quality_image):
        """
        从 dataloader 解包输入数据。
        
        参数:
            low_quality_image : 低质量（单角度）输入图像
            high_quality_image : 高质量（复合）目标图像 (用于对比)
        """
        # 命名修正：使用清晰的成员变量名
        self.low_quality_image = low_quality_image.to(self.device)
        self.high_quality_image = high_quality_image.to(self.device)

    def forward(self):
        """运行前向传播。"""
        # 命名修正：使用清晰的成员变量名
        self.generated_image = self.netG(self.low_quality_image)  # G(A)

    def optimize_parameters(self):
        """TestModel 不需要优化。"""
        pass