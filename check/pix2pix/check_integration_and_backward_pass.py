import torch
import logging
import argparse
import sys
from torch.utils.data import DataLoader

# [!!] 关键导入：
# 我们现在导入的是 *真实* 的 options, model 和 data
from options.train_options import TrainOptions
from models import create_model
from data_process import load_dataset

def check_full_integration_and_backward_pass():
    """
    集成测试：
    1. 加载 3D 兼容的 Options (train_options.py)
    2. 加载 3D 兼容的 Data (data_process.py)
    3. 加载 3D 兼容的 Model (pix2pix_model.py)
    4. 执行一次完整的 optimize_parameters() (前向 + 反向传播)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- [3D 集成与反向传播测试] 开始 ---")

    # --- [!!] 请修改为您 NIfTI 数据集的根目录 [!!] ---
    DATA_ROOT_PATH = "project_assets/Ultrasound_NIfTI_Dataset_Z185"
    
    # 1. 步骤 1: 加载 *真实* 的训练选项
    # ----------------------------------------------------
    logging.info("步骤 1: 正在解析 TrainOptions...")
    
    # 我们必须伪造 'sys.argv' 来覆盖默认值，
    # 确保它使用我们 3D 化的设置在 CPU 上运行一次。
    sys.argv = [
        'check_integration_and_backward_pass.py', # 脚本名 (占位)
        '--dataroot', DATA_ROOT_PATH,
        '--name', '3d_integration_test',
        '--model', 'pix2pix',          # [!!] 使用我们改造的 pix2pix
        '--netG', 'unet_128',          # [!!] 使用 3D U-Net (6-downsamples)
        '--netD', 'basic',             # [!!] 使用 3D PatchGAN
        '--gpu_ids', '-1',             # [!!] 强制在 CPU 上测试
        '--batch_size', '1',           # 保持 B=1 以便调试
        '--phase', 'train',
        '--no_flip'            # (可选) 禁用翻转，加快测试
    ]
    
    try:
        opt = TrainOptions().parse()
        logging.info("步骤 1 成功: TrainOptions 解析完毕。")
        logging.info(f"  > lambda_L2: {opt.lambda_L2}")
        logging.info(f"  > lr_D_ratio: {opt.lr_D_ratio}")
    except Exception as e:
        logging.error(f"步骤 1 失败: 解析 TrainOptions 时出错: {e}", exc_info=True)
        return

    # 2. 步骤 2: 加载 *真实* 的 3D 数据加载器
    # ----------------------------------------------------
    logging.info("步骤 2: 正在加载 3D 数据集 (data_process.py)...")
    try:
        dataset = load_dataset(opt, opt.phase)
        loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
        
        logging.info("  正在抓取一个 3D 数据批次...")
        lq_batch, hq_batch = next(iter(loader))
        
        expected_shape = (opt.batch_size, 1, 64, 64, 64)
        if lq_batch.shape != expected_shape:
             raise ValueError(f"LQ 批次形状 {lq_batch.shape} 与期望 {expected_shape} 不符")
             
        logging.info(f"步骤 2 成功: 抓取到 LQ 批次 {lq_batch.shape}")
        
    except Exception as e:
        logging.error(f"步骤 2 失败: 加载 3D 数据时出错: {e}", exc_info=True)
        logging.error("请检查: data_process.py 是否正确。")
        return

    # 3. 步骤 3: 创建 *真实* 的 3D 模型
    # ----------------------------------------------------
    logging.info("步骤 3: 正在创建 3D 模型 (pix2pix_model.py)...")
    try:
        model = create_model(opt)
        model.setup(opt) # 这将初始化 3D U-Net, 3D PatchGAN 和优化器
        logging.info(f"步骤 3 成功: 3D Pix2PixModel '{type(model).__name__}' 创建完毕。")
    except Exception as e:
        logging.error(f"步骤 3 失败: 创建模型时出错: {e}", exc_info=True)
        logging.error("请检查: network.py 和 pix2pix_model.py。")
        return

    # 4. 步骤 4: [!! 核心测试 !!] 执行一次完整的优化步骤
    # ----------------------------------------------------
    logging.info("步骤 4: 正在执行 model.optimize_parameters() (包含反向传播)...")
    try:
        # (1) 设置输入
        model.set_input(lq_batch, hq_batch)
        
        # (2) [!! 核心 !!]
        # 此函数将调用：
        #   model.forward()         (G 前向传播)
        #   model.backward_D()      (D 损失 + D 反向传播)
        #   model.optimizer_D.step()
        #   model.backward_G()      (G 损失 + G 反向传播) <--- 移除了 ContentLoss
        #   model.optimizer_G.step()
        model.optimize_parameters()
        
        logging.info("步骤 4 成功: 优化步骤 (optimize_parameters) 执行完毕，未抛出异常。")
        
    except Exception as e:
        logging.error(f"步骤 4 失败: 执行 optimize_parameters 时发生严重错误: {e}", exc_info=True)
        logging.error("请检查: pix2pix_model.py 中的 backward_G 和 backward_D 逻辑。")
        return

    # 5. 步骤 5: 验证损失计算
    # ----------------------------------------------------
    logging.info("步骤 5: 正在获取当前损失...")
    try:
        losses = model.get_current_losses()
        logging.info(f"  [成功] 获取到损失: {losses}")
        
        # 验证 contentLoss 已被移除
        if 'contentLoss' in losses:
             logging.error("  [失败] 'contentLoss' 仍存在于 get_current_losses() 中！")
             logging.error("  请检查: pix2pix_model.py 中的 self.loss_names")
             return
             
        # 验证 L2 损失存在
        if 'generator_pixelwise_l2_loss' not in losses:
             logging.error("  [失败] 未找到 'generator_pixelwise_l2_loss'！")
             return

        # 验证损失是否为 NaN (无效数字)
        for name, value in losses.items():
            if torch.isnan(torch.tensor(value)):
                logging.error(f"  [失败] 损失 '{name}' 计算结果为 NaN！")
                return
                
        logging.info("  [成功] 损失值均有效 (非 NaN)。")
        
    except Exception as e:
        logging.error(f"步骤 5 失败: 获取损失时出错: {e}", exc_info=True)
        return

    logging.info("--- [3D 集成与反向传播测试] 成功完成 ---")
    logging.info("所有 3D 模块已成功集成。模型可以接收数据并执行一次完整的训练步骤。")


if __name__ == "__main__":
    check_full_integration_and_backward_pass()