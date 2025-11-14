import os
import torch
import logging
import numpy as np
import nibabel as nib  # [改造点 1] 导入新库，移除 'sio'
import random           # [改造点 1] 导入 random 用于裁剪
from torch.utils.data import Dataset
# from cubdl_master.example_picmus_torch import dispaly_img # [改造点 4] 移除 2D 可视化
from typing import List, Tuple # 用于类型提示

# --- [改造点 2] 重构 load_dataset 函数 ---
def load_dataset(opt, phase):
    """
    加载 3D NIfTI 数据集路径。
    此函数不再加载实际数据，只查找文件路径并将其配对。
    """
    # [改造点 2.1] 根据 opt.dataroot 和 phase (例如 'train') 构建 LQ 和 HQ 目录
    data_root_dir = opt.dataroot
    lq_dir = os.path.join(data_root_dir, f"{phase}_lq")
    hq_dir = os.path.join(data_root_dir, f"{phase}_hq")

    if not os.path.isdir(lq_dir):
        logging.error(f"低质量 (LQ) 目录未找到: {lq_dir}")
        return AugmentedImageDataset([], (0,0,0), opt) # 返回空数据集
    if not os.path.isdir(hq_dir):
        logging.error(f"高质量 (HQ) 目录未找到: {hq_dir}")
        return AugmentedImageDataset([], (0,0,0), opt) # 返回空数据集
        
    logging.info(f"正在从 {lq_dir} 和 {hq_dir} 查找 NIfTI 文件...")

    # [改造点 2.2] 扫描 LQ 目录并创建文件路径配对
    file_pairs: List[Tuple[str, str]] = []
    lq_filenames = sorted([f for f in os.listdir(lq_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and '_lq' in f])

    if not lq_filenames:
        logging.warning(f"在 {lq_dir} 中未找到包含 '_lq' 后缀的 NIfTI 文件。")

    for fname in lq_filenames:
        lq_path = os.path.join(lq_dir, fname)
        
        # [!!! 战术修正 !!!]
        # 动态构建 HQ 路径：假设 LQ 文件名包含 '_lq'，而 HQ 文件名包含 '_hq'
        if '_lq' not in fname:
            logging.warning(f"跳过 {fname}：文件名不含 '_lq'，无法推导 HQ 文件名。")
            continue
        
        hq_fname = fname.replace('_lq', '_hq')
        hq_path = os.path.join(hq_dir, hq_fname)
        # [!!! 修正结束 !!!]

        if os.path.exists(hq_path):
            file_pairs.append((lq_path, hq_path))
        else:
            # [!!] 这里的日志现在会显示正确的目标路径
            logging.warning(f"跳过 {lq_path}：未找到对应的 HQ 文件 {hq_path}")

    if not file_pairs:
        logging.error(f"在 {lq_dir} 和 {hq_dir} 中未找到任何匹配的 NIfTI 文件对。")
        # [!!] 修正：确保在失败时返回一个有效的空数据集
        return AugmentedImageDataset([], (0,0,0), opt) 

    logging.info(f"成功找到 {len(file_pairs)} 个 NIfTI 文件对。")

    # [改造点 2.3] 简化 test_type 逻辑，按计划硬编码 patch_size
    patch_size = (64, 64, 64) 
    
    # [改造点 2.4] 将路径列表传递给 Dataset
    image_dataset = AugmentedImageDataset(
        file_pairs=file_pairs,
        patch_size=patch_size,
        opt=opt
    )
    return image_dataset

# --- [改造点 3] 彻底改造 AugmentedImageDataset 类 ---
class AugmentedImageDataset(Dataset):
    """
    3D 数据集类，实现 Lazy Loading 和 3D 随机裁剪。
    """
    def __init__(self, file_pairs: List[Tuple[str, str]], patch_size: Tuple[int, int, int], opt):
        """
        [改造点 3.1] __init__ 
        不再接收数据，只接收文件路径列表和配置。
        """
        super(AugmentedImageDataset, self).__init__()
        
        self.file_pairs = file_pairs
        self.patch_size_D, self.patch_size_H, self.patch_size_W = patch_size
        self.opt = opt
        self.len = len(self.file_pairs)
        
        # [!!] 战略修正：移除 self.affine_headers 字典 (修复多进程漏洞)
        
        logging.info(f"AugmentedImageDataset 初始化完成，patch_size={patch_size}。")


    def normalize(self, volume_np):
        """[!!] 战略修正：添加 normalize 辅助函数"""
        """将 3D 容积从 [-60, 0] 裁剪并归一化到 [-1, 1]"""
        MIN_VAL = -60.0
        MAX_VAL = 0.0
        # 1. 裁剪 (Clip)
        volume_np = np.clip(volume_np, MIN_VAL, MAX_VAL)
        # 2. 归一化到 [0, 1]
        volume_np = (volume_np - MIN_VAL) / (MAX_VAL - MIN_VAL)
        # 3. 缩放到 [-1, 1]
        return (volume_np * 2.0) - 1.0


    def __getitem__(self, index: int):
        """
        [!!] 战略修正：__getitem__ (实现 Lazy Loading 和模式感知)
        """
        # 1. 根据索引获取文件路径
        lq_path, hq_path = self.file_pairs[index]

        try:
            # [!!] 战略修正：检查模式
            if self.opt.phase == 'train':
                # --- 训练模式：返回 64x64x64 随机块 ---
                lq_nii = nib.load(lq_path)
                lq_volume = lq_nii.get_fdata().astype(np.float32)
                hq_nii = nib.load(hq_path)
                hq_volume = hq_nii.get_fdata().astype(np.float32)

                D, H, W = lq_volume.shape
                if D < self.patch_size_D or H < self.patch_size_H or W < self.patch_size_W:
                    logging.error(f"文件 {lq_path} 尺寸 ({D},{H},{W}) 小于 patch_size ({self.patch_size_D},{self.patch_size_H},{self.patch_size_W})")
                    return torch.empty(0), torch.empty(0), "Error"

                d_start = random.randint(0, D - self.patch_size_D)
                h_start = random.randint(0, H - self.patch_size_H)
                w_start = random.randint(0, W - self.patch_size_W)

                # [!!] 战术修正：修复了您计划中的 ... 拼写错误
                lq_patch = lq_volume[
                    d_start : d_start + self.patch_size_D,
                    h_start : h_start + self.patch_size_H,
                    w_start : w_start + self.patch_size_W
                ]
                hq_patch = hq_volume[
                    d_start : d_start + self.patch_size_D,
                    h_start : h_start + self.patch_size_H,
                    w_start : w_start + self.patch_size_W
                ]

                # [!!] 战略修正：使用 self.normalize
                lq_patch = self.normalize(lq_patch)
                hq_patch = self.normalize(hq_patch)

                # 3D 数据增强 (翻转)
                if not self.opt.no_flip:
                    if random.random() > 0.5:
                        lq_patch = np.flip(lq_patch, axis=0).copy()
                        hq_patch = np.flip(hq_patch, axis=0).copy()
                    if random.random() > 0.5:
                        lq_patch = np.flip(lq_patch, axis=1).copy()
                        hq_patch = np.flip(hq_patch, axis=1).copy()
                    if random.random() > 0.5:
                        lq_patch = np.flip(lq_patch, axis=2).copy()
                        hq_patch = np.flip(hq_patch, axis=2).copy()

                lq_tensor = torch.from_numpy(lq_patch.copy()).float().unsqueeze(0)
                hq_tensor = torch.from_numpy(hq_patch.copy()).float().unsqueeze(0)

                # [!!] 战略修正：返回 3 个项目 (修复结构不一致漏洞)
                return lq_tensor, hq_tensor, "N/A" # (添加 "N/A" 占位符)

            elif self.opt.phase == 'test':
                # --- 测试模式：返回完整的 185x128x128 容积 ---
                lq_nii = nib.load(lq_path)
                lq_volume = lq_nii.get_fdata().astype(np.float32)
                hq_nii = nib.load(hq_path)
                hq_volume = hq_nii.get_fdata().astype(np.float32)

                # [!!] 战略修正：使用 self.normalize
                lq_volume_norm = self.normalize(lq_volume)
                hq_volume_norm = self.normalize(hq_volume)

                lq_tensor = torch.from_numpy(lq_volume_norm).float().unsqueeze(0)
                hq_tensor = torch.from_numpy(hq_volume_norm).float().unsqueeze(0)

                # [!!] 战略修正：返回 3 个项目 (包括路径，修复多进程漏洞)
                return lq_tensor, hq_tensor, lq_path

        except Exception as e:
            logging.error(f"加载或处理文件 {lq_path} 时出错: {e}")
            return torch.empty(0), torch.empty(0), "Error"


    def __len__(self) -> int:
        return self.len

# --- (旧的 'test_image' 函数已移除) ---

# import os
# import torch
# import logging
# import numpy as np
# import nibabel as nib  # [改造点 1] 导入新库，移除 'sio'
# import random           # [改造点 1] 导入 random 用于裁剪
# from torch.utils.data import Dataset
# # from cubdl_master.example_picmus_torch import dispaly_img # [改造点 4] 移除 2D 可视化
# from typing import List, Tuple # 用于类型提示

# # --- [改造点 2] 重构 load_dataset 函数 ---
# def load_dataset(opt, phase):
#     """
#     加载 3D NIfTI 数据集路径。
#     此函数不再加载实际数据，只查找文件路径并将其配对。
#     """
#     # [改造点 2.1] 根据 opt.dataroot 和 phase (例如 'train') 构建 LQ 和 HQ 目录
#     data_root_dir = opt.dataroot
#     lq_dir = os.path.join(data_root_dir, f"{phase}_lq")
#     hq_dir = os.path.join(data_root_dir, f"{phase}_hq")

#     if not os.path.isdir(lq_dir):
#         logging.error(f"低质量 (LQ) 目录未找到: {lq_dir}")
#         return AugmentedImageDataset([], (0,0,0), opt) # 返回空数据集
#     if not os.path.isdir(hq_dir):
#         logging.error(f"高质量 (HQ) 目录未找到: {hq_dir}")
#         return AugmentedImageDataset([], (0,0,0), opt) # 返回空数据集
        
#     logging.info(f"正在从 {lq_dir} 和 {hq_dir} 查找 NIfTI 文件...")

#     # [改造点 2.2] 扫描 LQ 目录并创建文件路径配对
#     file_pairs: List[Tuple[str, str]] = []
#     lq_filenames = sorted([f for f in os.listdir(lq_dir) if (f.endswith('.nii') or f.endswith('.nii.gz')) and '_lq' in f])

#     if not lq_filenames:
#         logging.warning(f"在 {lq_dir} 中未找到包含 '_lq' 后缀的 NIfTI 文件。")

#     for fname in lq_filenames:
#         lq_path = os.path.join(lq_dir, fname)
        
#         # [!!! 战术修正 !!!]
#         # 动态构建 HQ 路径：假设 LQ 文件名包含 '_lq'，而 HQ 文件名包含 '_hq'
#         if '_lq' not in fname:
#             logging.warning(f"跳过 {fname}：文件名不含 '_lq'，无法推导 HQ 文件名。")
#             continue
        
#         hq_fname = fname.replace('_lq', '_hq')
#         hq_path = os.path.join(hq_dir, hq_fname)
#         # [!!! 修正结束 !!!]

#         if os.path.exists(hq_path):
#             file_pairs.append((lq_path, hq_path))
#         else:
#             # [!!] 这里的日志现在会显示正确的目标路径
#             logging.warning(f"跳过 {lq_path}：未找到对应的 HQ 文件 {hq_path}")

#     if not file_pairs:
#         logging.error(f"在 {lq_dir} 和 {hq_dir} 中未找到任何匹配的 NIfTI 文件对。")
#         # [!!] 修正：确保在失败时返回一个有效的空数据集
#         return AugmentedImageDataset([], (0,0,0), opt) 

#     logging.info(f"成功找到 {len(file_pairs)} 个 NIfTI 文件对。")

#     # [改造点 2.3] 简化 test_type 逻辑，按计划硬编码 patch_size
#     patch_size = (64, 64, 64) 
    
#     # [改造点 2.4] 将路径列表传递给 Dataset
#     image_dataset = AugmentedImageDataset(
#         file_pairs=file_pairs,
#         patch_size=patch_size,
#         opt=opt
#     )
#     return image_dataset
# # --- [改造点 3] 彻底改造 AugmentedImageDataset 类 ---
# class AugmentedImageDataset(Dataset):
#     """
#     3D 数据集类，实现 Lazy Loading 和 3D 随机裁剪。
#     """
#     def __init__(self, file_pairs: List[Tuple[str, str]], patch_size: Tuple[int, int, int], opt):
#         """
#         [改造点 3.1] __init__ 
#         不再接收数据，只接收文件路径列表和配置。
#         """
#         super(AugmentedImageDataset, self).__init__()
        
#         self.file_pairs = file_pairs
#         self.patch_size_D, self.patch_size_H, self.patch_size_W = patch_size
#         self.opt = opt
#         self.len = len(self.file_pairs)

#         # [改造点 3.1] 移除所有预加载和预增强逻辑
#         # (旧代码中所有 torch.from_numpy, torch.cat, torch.flip 等均已移除)
#         logging.info(f"AugmentedImageDataset 初始化完成，patch_size={patch_size}。")


#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         [改造点 3.2] __getitem__ (实现 Lazy Loading)
#         """
#         # 1. 根据索引获取文件路径
#         lq_path, hq_path = self.file_pairs[index]

#         try:
#             # 2. 加载完整的 3D NIfTI 容积
#             # (185, 128, 128) (D, H, W)
#             lq_nii = nib.load(lq_path)
#             lq_volume = lq_nii.get_fdata().astype(np.float32)

#             hq_nii = nib.load(hq_path)
#             hq_volume = hq_nii.get_fdata().astype(np.float32)

#             # 3. 计算随机裁剪坐标
#             D, H, W = lq_volume.shape
            
#             # 确保容积足够大
#             if D < self.patch_size_D or H < self.patch_size_H or W < self.patch_size_W:
#                 logging.error(f"文件 {lq_path} 尺寸 ({D},{H},{W}) 小于 patch_size ({self.patch_size_D},{self.patch_size_H},{self.patch_size_W})")
#                 # 返回空值或错误，这里我们返回 None，DataLoader 应该能处理
#                 return torch.empty(0), torch.empty(0) 

#             d_start = random.randint(0, D - self.patch_size_D)
#             h_start = random.randint(0, H - self.patch_size_H)
#             w_start = random.randint(0, W - self.patch_size_W)

#             # 4. 执行配对裁剪 (必须使用相同的坐标)
#             lq_patch = lq_volume[
#                 d_start : d_start + self.patch_size_D,
#                 h_start : h_start + self.patch_size_H,
#                 w_start : w_start + self.patch_size_W
#             ]
#             hq_patch = hq_volume[
#                 d_start : d_start + self.patch_size_D,
#                 h_start : h_start + self.patch_size_H,
#                 w_start : w_start + self.patch_size_W
#             ]

#             # # [!!] 战术决策点：数据归一化
#             # # 战略计划要求一个统一的归一化策略。
#             # # 在您提供具体策略（例如，[0, 1] 或 [-1, 1]）之前，这里暂不应用归一化。
#             # # 您需要在这里添加您的归一化代码，例如：
#             # MIN_VAL = -60.0
#             # MAX_VAL = 0.0
#             # lq_patch = (lq_patch - MIN_VAL) / (MAX_VAL - MIN_VAL) * 2.0 - 1.0
#             # hq_patch = (hq_patch - MIN_VAL) / (MAX_VAL - MIN_VAL) * 2.0 - 1.0
#             # # 缩放到 [-1, 1] (GAN 常用)
#             # lq_patch = lq_patch * 2.0 - 1.0
#             # hq_patch = hq_patch * 2.0 - 1.0
            
#             # [!!] 战术决策点：数据归一化
#             # 根据 3D Slicer 确认，范围是 [-60, 0]
#             MIN_VAL = -60.0 
#             MAX_VAL = 0.0

#             # 1. 裁剪 (Clip) - 这是处理 [-3, 1] 问题的关键！
#             # 这一步确保所有值（包括异常值）都落在 [-60, 0] 范围内。
#             lq_patch = np.clip(lq_patch, MIN_VAL, MAX_VAL)
#             hq_patch = np.clip(hq_patch, MIN_VAL, MAX_VAL)

#             # 2. 归一化到 [0, 1]
#             # 裁剪后，这里的输入保证在 [0, 1] 范围内
#             lq_patch = (lq_patch - MIN_VAL) / (MAX_VAL - MIN_VAL)
#             hq_patch = (hq_patch - MIN_VAL) / (MAX_VAL - MIN_VAL)
            
#             # 3. 缩放到 [-1, 1]
#             # 裁剪后，这里的输入保证在 [-1, 1] 范围内
#             lq_patch = lq_patch * 2.0 - 1.0
#             hq_patch = hq_patch * 2.0 - 1.0

#             # 5. [改造点 3.3] 执行 3D 数据增强 (按需随机执行)
#             # 使用 .copy() 来避免 NumPy 负步幅 (negative stride) 问题
#             if not self.opt.no_flip:
#                 # 随机翻转 D 轴 (轴向)
#                 if random.random() > 0.5:
#                     lq_patch = np.flip(lq_patch, axis=0).copy()
#                     hq_patch = np.flip(hq_patch, axis=0).copy()
#                 # 随机翻转 H 轴 (高程)
#                 if random.random() > 0.5:
#                     lq_patch = np.flip(lq_patch, axis=1).copy()
#                     hq_patch = np.flip(hq_patch, axis=1).copy()
#                 # 随机翻转 W 轴 (横向)
#                 if random.random() > 0.5:
#                     lq_patch = np.flip(lq_patch, axis=2).copy()
#                     hq_patch = np.flip(hq_patch, axis=2).copy()

#             # 6. 转换为 Tensor
#             lq_tensor = torch.from_numpy(lq_patch).float()
#             hq_tensor = torch.from_numpy(hq_patch).float()

#             # 7. [改造点 3.4] 添加通道维度 (C)
#             # (D, H, W) -> (C, D, H, W)
#             lq_tensor = lq_tensor.unsqueeze(0) # (1, 64, 64, 64)
#             hq_tensor = hq_tensor.unsqueeze(0) # (1, 64, 64, 64)

#             return lq_tensor, hq_tensor

#         except Exception as e:
#             logging.error(f"加载或处理文件 {lq_path} 时出错: {e}")
#             # 返回空张量，以便 DataLoader 的 collate_fn 可以跳过这个样本
#             return torch.empty(0), torch.empty(0)


#     def __len__(self) -> int:
#         return self.len

# # --- [改造点 4] 移除 2D 'test_image' 函数 ---
# # (旧的 test_image 函数已完全移除)