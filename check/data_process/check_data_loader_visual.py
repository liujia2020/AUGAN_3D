import torch
import logging
import argparse
import os
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from data_process import load_dataset # 导入我们正在测试的
from typing import Tuple

def check_data_loader_visual(opt: argparse.Namespace, num_samples_to_save: int = 2):
    """
    通过保存 NIfTI 产物来进行视觉调试，以验证 data_process.py 的逻辑。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- [数据管线视觉调试] 开始 ---")
    
    # [!!] 输出目录
    output_dir = "project_assets/data_check_output3"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"产物将保存到: {output_dir}")

    # 1. 加载数据集
    logging.info(f"正在加载数据集 (phase: {opt.phase})...")
    dataset = load_dataset(opt, opt.phase)
    if not dataset or len(dataset) == 0:
        logging.error("数据集加载失败。请检查路径和 data_process.py。")
        return

    # 2. 创建 DataLoader (使用 batch_size=1 逐个提取)
    # [!!] 我们故意不使用 shuffle，以便可以多次运行并获得不同的随机结果
    loader = DataLoader(
        dataset=dataset,
        batch_size=1, # 每次只取一个
        shuffle=True, # [!!] 保持 True 来测试随机性！
        num_workers=0 
    )

    logging.info(f"步骤 3: 尝试从 DataLoader 中提取并保存 {num_samples_to_save} 个样本...")

    try:
        # [!!] 我们需要一个恒定的仿射矩阵 (Affine) 来保存 NIfTI
        # 我们假设体素大小为 (0.2, 0.2, 0.2) mm
        # 您可以根据需要修改此处的
        voxel_size = (0.2, 0.2, 0.2)
        # 创建一个标准的 4x4 仿射矩阵
        affine_matrix = np.array([
            [voxel_size[0], 0, 0, 0],
            [0, voxel_size[1], 0, 0],
            [0, 0, voxel_size[2], 0],
            [0, 0, 0, 1]
        ])

        for i, data in enumerate(loader):
            if i >= num_samples_to_save:
                break
                
            logging.info(f"--- 正在处理和保存样本 {i+1}/{num_samples_to_save} ---")

            lq_batch, hq_batch = data

            if lq_batch.nelement() == 0 or hq_batch.nelement() == 0:
                logging.warning(f"  样本 {i+1} 为空，跳过。")
                continue

            # 3. 从批次中提取张量 (B=1, C=1, D, H, W)
            lq_tensor = lq_batch[0] # (1, 64, 64, 64)
            hq_tensor = hq_batch[0] # (1, 64, 64, 64)

            # 4. 转换回 NumPy (C, D, H, W) -> (D, H, W)
            #    我们移除通道维度 C
            lq_numpy = lq_tensor.squeeze(0).cpu().numpy()
            hq_numpy = hq_tensor.squeeze(0).cpu().numpy()
            
            logging.info(f"  样本 {i+1} 形状 (D,H,W): {lq_numpy.shape}")
            logging.info(f"  样本 {i+1} LQ 均值 (归一化后): {lq_numpy.mean():.4f}")
            logging.info(f"  样本 {i+1} LQ 范围 (Min/Max): {lq_numpy.min():.4f} / {lq_numpy.max():.4f}")

            # 5. 保存为 NIfTI 文件
            # 注意：我们保存的是归一化后 [-1, 1] 范围内的值
            lq_img = nib.Nifti1Image(lq_numpy, affine_matrix)
            hq_img = nib.Nifti1Image(hq_numpy, affine_matrix)

            lq_filename = os.path.join(output_dir, f"check_sample_{i+1}_LQ.nii.gz")
            hq_filename = os.path.join(output_dir, f"check_sample_{i+1}_HQ.nii.gz")

            nib.save(lq_img, lq_filename)
            nib.save(hq_img, hq_filename)
            
            logging.info(f"  [成功] 已保存: {lq_filename}")
            logging.info(f"  [成功] 已保存: {hq_filename}")

    except Exception as e:
        logging.error(f"在提取或保存样本时发生严重错误: {e}", exc_info=True)
        return

    logging.info(f"--- [数据管线视觉调试] 成功完成 ---")
    logging.info(f"请检查在 '{output_dir}' 目录中生成的 .nii.gz 文件。")


if __name__ == "__main__":
    # [!!] 请修改此路径为您 NIfTI 数据集的根目录 [!!]
    DATA_ROOT_PATH = "project_assets/Ultrasound_NIfTI_Dataset_Z185"
    
    mock_opt = argparse.Namespace()
    mock_opt.dataroot = DATA_ROOT_PATH
    mock_opt.phase = 'train'
    mock_opt.no_flip = True # 确保测试 3D 翻转增强
    
    # 运行视觉调试
    check_data_loader_visual(mock_opt, num_samples_to_save=2)