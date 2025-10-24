import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import time
try:
    from thop import profile
    
except Exception:
    profile = None  # optional; only used if FLOPs profiling is needed
    
from cubdl_master.example_picmus_torch import load_datasets,create_network,mk_img
from options.train_options import TrainOptions
from models import create_model
from data_process import load_dataset
from models.network import UnetGenerator
import math
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage,CenterCrop,Resize,RandomRotation,RandomVerticalFlip,RandomHorizontalFlip,Resize,ColorJitter
from metrics import image_evaluation

# def makedir(opt_name):
#     """Making the directory for saving pictures of loss change."""
#     train_base = os.path.join('./images', opt_name, 'train')
#     test_base = os.path.join('./images', opt_name, 'test')

#     # 一行代码搞定：如果目录已存在，exist_ok=True 会让它不报错
#     os.makedirs(train_base, exist_ok=True)
#     os.makedirs(test_base, exist_ok=True)

#     loss_path = os.path.join(train_base, 'loss_history.png')
#     return loss_path


def makedir(opt_name):
    """创建用于保存训练过程中损失图和测试图片的目录。"""
    # --- 修改基础路径 ---
    output_base_dir = os.path.join('./project_assets/output_visualizations', opt_name)
    # --- 修改结束 ---
    train_base = os.path.join(output_base_dir, 'train') # 例如 ./project_assets/output_visualizations/augan_run_2/train/
    test_base = os.path.join(output_base_dir, 'test')   # 例如 ./project_assets/output_visualizations/augan_run_2/test/

    os.makedirs(train_base, exist_ok=True)
    os.makedirs(test_base, exist_ok=True) # 保留 test 目录的创建，以防万一

    loss_path = os.path.join(train_base, 'loss_history.png')
    return loss_path


if __name__ == '__main__':
    # --- 1. 初始化设置 ---
    
    # 加载PICMUS数据仅用于获取可视化边界
    try:
        plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
        das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])
    except Exception as e:
        logging.warning("PICMUS dataset not available, skip visualization setup: %s", e)
        plane_wave_data = None
        xlims, zlims = [0, 1], [0, 1]

    opt = TrainOptions().parse() # 解析所有命令行参数
    model = create_model(opt)      # 创建模型 (pix2pix_model)
    model.setup(opt)               # 加载权重, 设置调度器
    total_iters = 0                # 总迭代次数
    total_epochs = opt.n_epochs + opt.niter_decay # 总训练轮数

    # --- 2. 日志和数据加载 ---
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:  %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device  {device}')
    
    loss_path = makedir(opt.name) # 创建用于保存损失图像的目录

    # 加载数据集
    img_dataset = load_dataset(opt, opt.phase, 0)
    dataset_len = img_dataset.len
    train_loader = DataLoader(dataset=img_dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    
    # --- 3. 创建用于历史记录的数组 ---
    # 我们将记录所有关键的损失值
    history_loss_G_Total = np.zeros(total_epochs)
    history_loss_D_Total = np.zeros(total_epochs)
    history_loss_G_GAN = np.zeros(total_epochs)
    history_loss_G_L2 = np.zeros(total_epochs)
    history_loss_G_Content = np.zeros(total_epochs)
    history_loss_D_Real = np.zeros(total_epochs)
    history_loss_D_Fake = np.zeros(total_epochs)
    
    logging.info(f'Starting training for {total_epochs} epochs. Dataset size: {dataset_len} images.')

    # --- 4. 训练主循环 ---
    for epoch in range(opt.epoch_count, total_epochs + 1):
        epoch_start_time = time.time() # 记录周期开始时间
        
        # --- 每周期的损失累加器 ---
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_G_GAN = 0.0
        epoch_loss_G_L2 = 0.0
        epoch_loss_G_Content = 0.0
        epoch_loss_D_Real = 0.0
        epoch_loss_D_Fake = 0.0
        
        epoch_iter = 0 # 当前周期的迭代次数
        
        # 使用tqdm创建进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", unit="image")

        for low_quality_image, high_quality_image in train_bar:
            iter_start_time = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # (已在 data_process.py 中处理好形状)
            
            # 将数据送入模型并执行优化
            model.set_input(low_quality_image, high_quality_image)
            model.optimize_parameters()  

            # --- 累加所有损失 ---
            epoch_loss_G += model.loss_G.item()
            epoch_loss_D += model.loss_D.item()
            epoch_loss_G_GAN += model.generator_adversarial_loss.item()
            epoch_loss_G_L2 += model.generator_pixelwise_l2_loss.item()
            epoch_loss_G_Content += model.contentLoss.item()
            epoch_loss_D_Real += model.discriminator_loss_on_real.item()
            epoch_loss_D_Fake += model.discriminator_loss_on_fake.item()

            # --- 更新Tqdm进度条描述 (更频繁地) ---
            if total_iters % opt.print_freq == 0:
                train_bar.set_description(
                    f"[Epoch {epoch}] "
                    f"G_loss: {model.loss_G.item():.4f} | "
                    f"D_loss: {model.loss_D.item():.4f}"
                )

        # --- 5. 周期（Epoch）结束 ---
        
        # 保存模型
        if epoch % opt.save_epoch_freq == 0:
            print(f'\nSaving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # --- 计算平均损失 ---
        avg_loss_G = epoch_loss_G / epoch_iter
        avg_loss_D = epoch_loss_D / epoch_iter
        avg_loss_G_GAN = epoch_loss_G_GAN / epoch_iter
        avg_loss_G_L2 = epoch_loss_G_L2 / epoch_iter
        avg_loss_G_Content = epoch_loss_G_Content / epoch_iter
        avg_loss_D_Real = epoch_loss_D_Real / epoch_iter
        avg_loss_D_Fake = epoch_loss_D_Fake / epoch_iter

        # --- 存储到历史记录 ---
        history_loss_G_Total[epoch-1] = avg_loss_G
        history_loss_D_Total[epoch-1] = avg_loss_D
        history_loss_G_GAN[epoch-1] = avg_loss_G_GAN
        history_loss_G_L2[epoch-1] = avg_loss_G_L2
        history_loss_G_Content[epoch-1] = avg_loss_G_Content
        history_loss_D_Real[epoch-1] = avg_loss_D_Real
        history_loss_D_Fake[epoch-1] = avg_loss_D_Fake

        # --- 更新学习率 ---
        model.update_learning_rate()
        
        # --- 6. 打印全面的周期总结日志 ---
        print('\n' + '-'*80)
        print(f'END OF EPOCH {epoch} / {total_epochs} \t Time Taken: {time.time() - epoch_start_time:.0f} sec')
        
        # 获取当前G和D的学习率
        lr_G = model.optimizers[0].param_groups[0]['lr']
        lr_D = model.optimizers[1].param_groups[0]['lr']
        print(f'  Learning Rates: \t G_lr = {lr_G:.7f} | D_lr = {lr_D:.7f}')
        
        print('  Average Losses:')
        print(f'    Generator (G): \t Total = {avg_loss_G:.4f}')
        print(f'      ├─ G_Adversarial: \t {avg_loss_G_GAN:.4f}')
        print(f'      ├─ G_Pixelwise (L2): \t {avg_loss_G_L2:.4f}')
        print(f'      └─ G_Content: \t\t {avg_loss_G_Content:.4f}')
        print(f'    Discriminator (D): \t Total = {avg_loss_D:.4f}')
        print(f'      ├─ D_Real_Loss: \t {avg_loss_D_Real:.4f}')
        print(f'      └─ D_Fake_Loss: \t {avg_loss_D_Fake:.4f}')
        print('-'*80 + '\n')
    
    # --- 7. 训练结束 - 绘制最终分析图表 ---
    print('Training finished. Plotting loss history...')
    plt.figure(figsize=(20, 10)) # 创建一个更大的图窗

    # 子图1: G 和 D 的总损失
    plt.subplot(2, 2, 1)
    plt.plot(history_loss_G_Total[:epoch], label="Generator (G) Total Loss")
    plt.plot(history_loss_D_Total[:epoch], label="Discriminator (D) Total Loss")
    plt.title("Total GAN Losses", fontsize=10)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 子图2: G 的损失分量
    plt.subplot(2, 2, 2)
    plt.plot(history_loss_G_GAN[:epoch], label="G Adversarial Loss")
    plt.plot(history_loss_G_L2[:epoch], label="G Pixelwise (L2) Loss")
    plt.plot(history_loss_G_Content[:epoch], label="G Content Loss")
    plt.title("Generator Loss Components", fontsize=10)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # 子图3: D 的损失分量
    plt.subplot(2, 2, 3)
    plt.plot(history_loss_D_Real[:epoch], label="D Real Loss")
    plt.plot(history_loss_D_Fake[:epoch], label="D Fake Loss")
    plt.title("Discriminator Loss Components", fontsize=10)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 添加总标题
    plt.suptitle(f"Training Loss History for: {opt.name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应总标题
    
    # 保存图表
    plt.savefig(loss_path)
    print(f'Loss history plot saved to {loss_path}')
    # plt.show() # 如果在服务器上运行，可能需要注释掉这行