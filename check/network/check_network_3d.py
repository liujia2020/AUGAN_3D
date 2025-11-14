import torch
import logging
import argparse
import models.network as network # [!!] 导入我们刚刚改造的文件

def check_network_shapes(opt: argparse.Namespace):
    """
    隔离测试 models/network.py 中 3D 网络的实例化和形状流。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- [3D 网络架构冒烟测试] 开始 ---")
    
    # 1. 实例化 3D 生成器 (U-Net)
    # ----------------------------------------------------
    logging.info(f"步骤 1: 实例化 3D 生成器 (netG='{opt.netG}')...")
    try:
        # [!!] 关键测试点 1：
        # define_G 是否能正确调用 get_norm_layer(3D) 
        # 并且 UnetGenerator 是否能正确构建 3D 模块
        netG = network.define_G(
            input_nc=opt.input_nc, 
            output_nc=opt.output_nc, 
            ngf=opt.ngf, 
            netG=opt.netG, 
            norm=opt.norm, 
            use_dropout=not opt.no_dropout, 
            init_type=opt.init_type, 
            init_gain=opt.init_gain, 
            gpu_ids=opt.gpu_ids, 
            use_sab=opt.use_sab
        )
        logging.info("步骤 1 成功: 3D 生成器 (G) 实例化完毕。")
    except Exception as e:
        logging.error(f"步骤 1 失败: 实例化 3D 生成器时出错: {e}", exc_info=True)
        logging.error("请检查: 1. models/network.py 中 UnetGenerator 和 UnetSkipConnectionBlock 的 3D 改造。 2. get_norm_layer 是否返回 3D 归一化。")
        return

    # 2. 实例化 3D 判别器 (PatchGAN)
    # ----------------------------------------------------
    logging.info(f"步骤 2: 实例化 3D 判别器 (netD='{opt.netD}')...")
    try:
        # [!!] 关键测试点 2：
        # NLayerDiscriminator 是否能正确构建 3D 卷积
        netD = network.define_D(
            input_nc=opt.input_nc + opt.output_nc, # Pix2Pix 中 D 的输入是 (LQ + HQ)
            ndf=opt.ndf, 
            netD=opt.netD,
            n_layers_D=opt.n_layers_D, 
            norm=opt.norm, 
            init_type=opt.init_type, 
            init_gain=opt.init_gain, 
            gpu_ids=opt.gpu_ids
        )
        logging.info("步骤 2 成功: 3D 判别器 (D) 实例化完毕。")
    except Exception as e:
        logging.error(f"步骤 2 失败: 实例化 3D 判别器时出错: {e}", exc_info=True)
        logging.error("请检查: 1. models/network.py 中 NLayerDiscriminator 的 3D 改造。")
        return

    # 3. 创建模拟 3D 输入批次 (来自 data_process.py)
    # ----------------------------------------------------
    logging.info("步骤 3: 创建模拟 3D 输入批次 (Batch)...")
    
    BATCH_SIZE = 2 # 模拟一个批次
    PATCH_D, PATCH_H, PATCH_W = (64, 64, 64) # 必须与 data_process.py 一致
    
    # (B, C, D, H, W)
    dummy_input = torch.randn(BATCH_SIZE, opt.input_nc, PATCH_D, PATCH_H, PATCH_W)
    logging.info(f"  模拟输入形状 (B,C,D,H,W): {dummy_input.shape}")
    
    # 4. 测试生成器 (G) 的前向传播
    # ----------------------------------------------------
    logging.info("步骤 4: 测试生成器 (G) 的前向传播...")
    try:
        # [!!] 关键测试点 3：
        # 3D 块能否无错地流过 3D U-Net？
        fake_output = netG(dummy_input)
        
        expected_shape = dummy_input.shape
        logging.info(f"  [成功] G 输出形状: {fake_output.shape}")
        
        # 验证：U-Net 的输出形状必须与输入形状完全相同
        if fake_output.shape != expected_shape:
            logging.error(f"  [失败] G 输出形状 ({fake_output.shape}) 与输入形状 ({expected_shape}) 不匹配！")
            return
        logging.info(f"  [成功] G 形状验证通过。")
        
    except Exception as e:
        logging.error(f"步骤 4 失败: 3D 生成器 (G) 前向传播时出错: {e}", exc_info=True)
        logging.error("请检查: UnetSkipConnectionBlock 中的 3D 卷积/反卷积参数是否正确。")
        return

    # 5. 测试判别器 (D) 的前向传播
    # ----------------------------------------------------
    logging.info("步骤 5: 测试判别器 (D) 的前向传播...")
    
    # 模拟 Pix2Pix 的 D 输入：(B, C_in + C_out, D, H, W)
    d_input = torch.cat((dummy_input, fake_output), 1) # 沿通道维度 (dim=1) 拼接
    logging.info(f"  模拟 D 输入形状: {d_input.shape}")
    
    try:
        # [!!] 关键测试点 4：
        # 3D 块能否无错地流过 3D PatchGAN？
        pred = netD(d_input)
        
        # 3D PatchGAN (n_layers=3) 的输出形状计算：
        # Input: 64x64x64
        # L1 (s=2): 32x32x32
        # L2 (s=2): 16x16x16
        # L3 (s=2): 8x8x8
        # L4 (s=1): 7x7x7  (N-F+2P)/S + 1 = (8-4+2)/1 + 1 = 7
        # L5 (s=1): 6x6x6  (N-F+2P)/S + 1 = (7-4+2)/1 + 1 = 6
        expected_D_shape = (BATCH_SIZE, 1, 6, 6, 6)
        
        logging.info(f"  [成功] D 输出形状: {pred.shape}")
        
        if pred.shape != expected_D_shape:
            logging.error(f"  [失败] D 输出形状 ({pred.shape}) 与期望的 3D PatchGAN 形状 ({expected_D_shape}) 不匹配！")
            return
        logging.info(f"  [成功] D 形状验证通过。")

    except Exception as e:
        logging.error(f"步骤 5 失败: 3D 判别器 (D) 前向传播时出错: {e}", exc_info=True)
        logging.error("请检查: NLayerDiscriminator 中的 3D 卷积参数是否正确。")
        return

    logging.info("--- [3D 网络架构冒烟测试] 成功完成 ---")
    logging.info("所有 3D 模块均已成功实例化，并且前向传播形状正确。")


if __name__ == "__main__":
    # 模拟一个 'opt' 对象，包含运行网络所需的所有参数
    # (这必须与您在 train.py 中使用的参数相匹配)
    
    mock_opt = argparse.Namespace()
    
    # --- 关键 G/D 架构参数 ---
    mock_opt.input_nc = 1       # 灰度 3D 块
    mock_opt.output_nc = 1      # 灰度 3D 块
    mock_opt.ngf = 64         # 生成器滤波器数量 (同 2D)
    mock_opt.ndf = 64         # 判别器滤波器数量 (同 2D)
    
    mock_opt.netG = 'unet_128'  # [!!] 必须是 3D 化的 'unet_...'
    mock_opt.netD = 'basic'     # [!!] 必须是 3D 化的 'basic' 或 'n_layers'
    mock_opt.n_layers_D = 3   # (netD='basic' 默认值)
    
    mock_opt.norm = 'instance'  # 'batch' 或 'instance'
    mock_opt.no_dropout = True
    
    # [!!] 战术决策：必须禁用 2D 注意力
    mock_opt.use_sab = False 
    
    # --- 其他辅助参数 ---
    mock_opt.init_type = 'normal'
    mock_opt.init_gain = 0.02
    mock_opt.gpu_ids = []     # [!!] 在 CPU 上测试，不依赖 GPU
    
    # 运行测试
    check_network_shapes(mock_opt)