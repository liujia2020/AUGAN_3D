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