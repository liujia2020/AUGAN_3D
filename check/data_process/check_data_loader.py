import torch
import logging
import argparse
from torch.utils.data import DataLoader
from data_process import load_dataset # 导入我们刚刚改造的函数
from typing import Tuple

def check_data_pipeline(opt: argparse.Namespace, batch_size: int = 2, num_batches_to_check: int = 3):
    """
    隔离测试 data_process.py 的 3D 数据加载和裁剪流程。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- [数据管线隔离测试] 开始 ---")

    # 1. 尝试调用 load_dataset
    # ----------------------------------------------------
    logging.info(f"步骤 1: 尝试从 {opt.dataroot} (phase: {opt.phase}) 查找 NIfTI 文件对...")
    try:
        dataset = load_dataset(opt, opt.phase)
    except Exception as e:
        logging.error(f"load_dataset 函数执行失败: {e}", exc_info=True)
        return

    if not dataset or len(dataset) == 0:
        logging.error("步骤 1 失败: load_dataset 返回了一个空的或无效的数据集。")
        logging.error("请检查: 1. `opt.dataroot` 路径是否正确。 2. 子目录是否为 'train_lq' 和 'train_hq'。 3. .nii 文件是否存在。")
        return
    logging.info(f"步骤 1 成功: 找到 {len(dataset)} 个文件对。")

    # 2. 尝试实例化 DataLoader
    # ----------------------------------------------------
    logging.info(f"步骤 2: 尝试创建 DataLoader (batch_size={batch_size})...")
    try:
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 # 保持为 0 以便在主进程中调试
        )
    except Exception as e:
        logging.error(f"DataLoader 实例化失败: {e}", exc_info=True)
        return
    logging.info(f"步骤 2 成功: DataLoader 创建完毕。")

    # 3. 尝试迭代加载数据块 (核心测试)
    # ----------------------------------------------------
    logging.info(f"步骤 3: 尝试从 DataLoader 中提取 {num_batches_to_check} 个批次的数据...")
    
    try:
        for i, data in enumerate(loader):
            if i >= num_batches_to_check:
                break # 我们只需要检查几个批次

            logging.info(f"--- 正在检查 Batch {i+1}/{num_batches_to_check} ---")
            
            # 检查数据是否为空 (我们在 __getitem__ 中设置了错误返回)
            if not data or len(data) != 2:
                logging.error(f"Batch {i+1} 加载失败: data 为空或格式不正确。")
                continue

            lq_batch, hq_batch = data

            # 检查是否因错误而返回了空张量
            if lq_batch.nelement() == 0 or hq_batch.nelement() == 0:
                logging.warning(f"Batch {i+1} 加载了空张量 (可能在 __getitem__ 中有错误被捕获)。跳过。")
                continue

            # [!!] 核心验证点：检查输出形状
            expected_shape = (batch_size, 1, 64, 64, 64) # (B, C, D, H, W)
            lq_shape = tuple(lq_batch.shape)
            hq_shape = tuple(hq_batch.shape)

            logging.info(f"  LQ 批次形状: {lq_shape} (期望: {expected_shape})")
            logging.info(f"  HQ 批次形状: {hq_shape} (期望: {expected_shape})")

            if lq_shape != expected_shape:
                logging.error(f"  [失败] LQ 形状不匹配！")
            if hq_shape != expected_shape:
                logging.error(f"  [失败] HQ 形状不匹配！")

            # 健全性检查：检查数据内容
            lq_mean = lq_batch.mean().item()
            hq_std = hq_batch.std().item()
            logging.info(f"  LQ 均值 (健全性检查): {lq_mean:.4f}")
            logging.info(f"  HQ 标准差 (健全性检查): {hq_std:.4f}")
            
            if torch.isnan(lq_batch).any() or torch.isnan(hq_batch).any():
                logging.error(f"  [失败] 数据中检测到 NaN！")

    except Exception as e:
        logging.error(f"步骤 3 失败: 在迭代 DataLoader 时发生严重错误: {e}", exc_info=True)
        logging.error("请重点检查: 1. data_process.py 的 __getitem__ 中的 NumPy 裁剪逻辑。 2. 3D 增强逻辑 (如 np.flip)。")
        return

    logging.info(f"--- [数据管线隔离测试] 成功完成 ---")


if __name__ == "__main__":
    # 我们不需要完整的 BaseOptions，只需要一个模拟的 'opt' 对象
    # (这是 argparse.Namespace 的标准用法，用于模拟解析)
    
    # [!!] 请修改此路径为您 NIfTI 数据集的根目录 [!!]
    DATA_ROOT_PATH = "project_assets/Ultrasound_NIfTI_Dataset_Z185"
    
    mock_opt = argparse.Namespace()
    mock_opt.dataroot = DATA_ROOT_PATH
    mock_opt.phase = 'train'      # 我们正在测试训练加载器
    mock_opt.no_flip = False      # 确保测试 3D 翻转增强
    
    check_data_pipeline(mock_opt)